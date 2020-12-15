import json
import args
import torch
import pickle
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import predict_data
from collections import OrderedDict
from tokenization import BertTokenizer
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForQuestionAnswering, BertConfig
from transformer.translator_gen import Translator

use_cuda = torch.cuda.is_available()

def convert_idx2text(example, tokenizer):
    words = []
    for i in example:
        if i.item() == 99:
            break
        words.append(tokenizer.ids_to_tokens[i.item()])
    return ''.join(words)

def find_best_answer_for_passage(start_probs, end_probs, passage_len, question, is_second):
    best_start, best_end, max_prob = -1, -1, 0

    start_probs, end_probs = start_probs.unsqueeze(0), end_probs.unsqueeze(0)
    if is_second:
        # prob_start, best_start = torch.max(start_probs, 1)
        prob_start, best_start = torch.max(start_probs, 1)
        prob_start_top2, best_start_top2 = torch.topk(start_probs, 2, dim=1, largest=True, sorted=True)
        prob_start_t1, prob_start = torch.split(prob_start_top2, 1, dim=-1)
        best_start_t1, best_start = torch.split(best_start_top2, 1, dim=-1)

        prob_end, best_end = torch.max(end_probs, 1)
        prob_end_top2, best_end_top2 = torch.topk(end_probs, 2, dim=1, largest=True, sorted=True)
        prob_end_t1, prob_end = torch.split(prob_end_top2, 1, dim=-1)
        best_end_t1, best_end = torch.split(best_end_top2, 1, dim=-1)
    else:
        prob_start, best_start = torch.max(start_probs, 1)
        prob_end, best_end = torch.max(end_probs, 1)
    num = 0
    while True:
        if num > 3:
            break
        if best_end >= best_start:
            break
        else:
            start_probs[0][best_start], end_probs[0][best_end] = 0.0, 0.0
            prob_start, best_start = torch.max(start_probs, 1)
            prob_end, best_end = torch.max(end_probs, 1)
        num += 1
    max_prob = prob_start * prob_end

    if best_start <= best_end:
        return best_start, best_end, max_prob
    else:
        return best_end, best_start, max_prob


def find_best_answer(sample, start_probs, end_probs, can_logits, is_second=False,
                     prior_scores=(0.44, 0.23, 0.15, 0.09, 0.07)):
    best_p_idx, best_span, best_score = None, None, 0
    best_answer = ''
    can_logit = 0
    for p_idx, passage in enumerate(sample['doc_tokens'][:args.max_para_num]):

        passage_len = min(args.max_seq_length, len(passage['doc_tokens']))

        start, end, score = find_best_answer_for_passage(start_probs[p_idx], end_probs[p_idx], passage_len,
                                                         sample['question_text'], is_second)

        answer_span = (start, end)
        # 如果他俩都为0，并且一直没有best score 那么认为没有答案，下面74行会判断，如果答案no，返回
        if answer_span[0].item() == 0 and answer_span[1].item() == 0 and best_score == 0:
            best_span = answer_span
            best_answer = 'no'
            best_p_idx = p_idx
            can_logit = can_logits[p_idx]
            continue
        # 如果有best score， 需要过滤掉这条数据，不要影响best_span
        elif answer_span[0].item() == 0 and answer_span[1].item() == 0 and best_score != 0:
            continue
        # 如果有span，则需要对span-1，因为预测计算的最大的那个index是加了1的，所以这里－1,
        else:
            start -= 1
            end -= 1
            answer_span = (start, end)
        score *= prior_scores[p_idx]

        answer = "p" + sample['question_text'] + "。" + sample['doc_tokens'][p_idx]['doc_tokens']
        best_answer = answer[answer_span[0]: answer_span[1] + 1]

        if score > best_score:
            best_score = score
            best_p_idx = p_idx
            best_span = answer_span

    if best_p_idx is None or best_span is None:
        best_answer = ''
        best_p_idx = 0
    elif best_answer == 'no':
        return best_answer, best_p_idx, can_logit
    else:
        para = "p" + sample['question_text'] + "。" + sample['doc_tokens'][best_p_idx]['doc_tokens']
        best_answer = ''.join(para[best_span[0]: best_span[1] + 1])
    # print(best_p_idx)
    return best_answer, best_p_idx, can_logits[best_p_idx]


def evaluate(model, result_file):
    print(args.predict_example_files)
    with open(args.predict_example_files, 'rb') as f:
        eval_examples = pickle.load(f)

    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('../roberta_wwm_ext', do_lower_case=True)
        # model.eval()
        pred_answers, ref_answers = [], []
        # fake = []
        # true = []
        # pred_count = 0
        # true_count = 0
        # pred_recall = 0
        # aa, bb = 0, 0
        # ref_yes1, ref_yes2, ref_no1, ref_no2 = 0, 0, 0, 0
        # # in all dataset
        # ref_yes11, ref_yes22, ref_no11, ref_no22 = 0, 0, 0, 0
        for step, example in enumerate(tqdm(eval_examples)):
            # start_probs, end_probs = [], []
            # scores, lines = [], []
            # can_logits = []
            question_text = example['question_text']
            if len(example['doc_tokens']) == 0:
                continue
            # ************************************
            # # 如果motivation 是multiple docs，那就要考虑下面每一篇文档了。否则看predicting_single.py
            # for p_num, doc_tokens in enumerate(example['doc_tokens'][:args.max_para_num]):
            #     (input_ids, input_ids_q, input_mask, attention_mask_q, segment_ids) = \
            #         predict_data.predict_data(question_text, doc_tokens['doc_tokens'], tokenizer, args.max_seq_length,
            #                                   args.max_query_length)
            #     # input_ids = torch.tensor(input_ids).unsqueeze(0)
            #     # input_mask = torch.tensor(input_mask).unsqueeze(0)
            #     # segment_ids = torch.tensor(segment_ids).unsqueeze(0)
            #     input_ids, input_mask, input_ids_q, attention_mask_q, segment_ids = \
            #         input_ids.cuda(), input_mask.cuda(), input_ids_q.cuda(), attention_mask_q.cuda(), \
            #         segment_ids.cuda()
            #     all_hyp, all_scores = \
            #         model.translate_batch(input_ids, input_ids_q, token_type_ids=segment_ids,
            #                               attention_mask=input_mask, attention_mask_q=attention_mask_q)
            #     for idx_seqs in all_hyp:    # batch_size=1, 就执行一次
            #         for idx_seq in idx_seqs:
            #             pred_line = convert_idx2text(idx_seq, tokenizer)
            #             print(pred_line)
            #             # print(all_scores)
            #             lines.append(pred_line)
            #             # output.write(pred_line + '\n')
            #     # lines += batch.src[0].size(0)
            #     # scores.append(all_scores[0] / len(idx_seq))
            #     scores.append(all_scores[0])
            #     # (input_ids, input_mask, segment_ids, can_answer) = \
            #     #     predict_data.predict_data(question_text, doc_tokens['doc_tokens'], tokenizer, args.max_seq_length,
            #     #                               args.max_query_length)
            #     # start_prob, end_prob = model(input_ids, segment_ids, can_answer,
            #     #                              attention_mask=input_mask)  # !!!!!!!!!!
            #
            #     # start_probs.append(start_prob.squeeze(0))
            #     # end_probs.append(end_prob.squeeze(0))
            #     can_logits.append(0)
            #
            # # 新加的
            # # print(example)
            # # print('example[answers] is : {}'.format(example['answers']))
            # best_score, best_index = torch.max(torch.tensor(scores), 0)
            # # print('scores is : {}, best_index is {}: '.format(scores, best_index))
            # best_answer = lines[best_index.item()]
            # print('best_answeris : {}'.format(best_answer))
            # print('example[answers] is : {}'.format(example['answers']))
            # ************************************

            (input_ids, input_ids_q, input_mask, attention_mask_q, segment_ids) = \
                predict_data.predict_data(question_text, example['doc_tokens'][:args.max_seq_length], tokenizer,
                                          args.max_seq_length, args.max_query_length)
            input_ids, input_mask, input_ids_q, attention_mask_q, segment_ids = \
                input_ids.cuda(), input_mask.cuda(), input_ids_q.cuda(), attention_mask_q.cuda(), \
                segment_ids.cuda()
            all_hyp, all_scores = \
                model.translate_batch(input_ids, input_ids_q, token_type_ids=segment_ids,
                                      attention_mask=input_mask, attention_mask_q=attention_mask_q)
            # print('inputs:', input_ids)
            for idx_seqs in all_hyp:    # batch_size=1, 就执行一次
                for idx_seq in idx_seqs:
                    pred_line = convert_idx2text(idx_seq, tokenizer)
                    print(pred_line)
                    # print(all_scores)
                    # lines.append(pred_line)
                    # output.write(pred_line + '\n')

            # best_answer, docs_index, can_logit = find_best_answer(example, start_probs, end_probs, can_logits)
            can_logit = 0
            can_logit = (float)(can_logit)
            # if can_logit >= 0.8 and best_answer == 'no':
            #     if len(example['answers']) == 0:
            #         aa += 1
            #     else:
            #         bb += 1
            #     best_answer, docs_index, can_logits = find_best_answer(example, start_probs,
            #                                                            end_probs, can_logits, is_second=True)

            # if can_logit <= 0.8 and best_answer != 'no':
                # if can_logit >= 0.9:
                #     aa += 1
                # else:
                #     bb += 1
                # aa += 1
                # best_answer = 'no'

            can_logit = (float)(can_logit)

            # if len(example['answers']) != 0 and can_logit >= 0.9:
            #     ref_yes11 += 1
            # if len(example['answers']) != 0 and can_logit <= 0.9:
            #     ref_yes22 += 1
            # if len(example['answers']) == 0 and can_logit >= 0.9:
            #     ref_no11 += 1
            # if len(example['answers']) == 0 and can_logit <= 0.9:
            #     ref_no22 += 1

            # if len(example['answers']) != 0 and can_logit >= 0.5:
            #     ref_yes11 += 1
            # if len(example['answers']) != 0 and can_logit <= 0.5:
            #     ref_yes22 += 1
            # if len(example['answers']) == 0 and can_logit >= 0.5:
            #     ref_no11 += 1
            # if len(example['answers']) == 0 and can_logit <= 0.5:
            #     ref_no22 += 1
            #
            # if len(example['answers']) == 0:
            #     true.append(example)
            #     print('ref no answer, score is:', can_logit)
            #     true_count += 1
            # if best_answer == 'no':
            #     # if len(example['answers']) != 0 and can_logit >= 0.9:
            #     #     ref_yes1 += 1
            #     # if len(example['answers']) != 0 and can_logit <= 0.9:
            #     #     ref_yes2 += 1
            #     # if len(example['answers']) == 0 and can_logit >= 0.9:
            #     #     ref_no1 += 1
            #     # if len(example['answers']) == 0 and can_logit <= 0.9:
            #     #     ref_no2 += 1
            #     if len(example['answers']) != 0 and can_logit >= 0.5:
            #         ref_yes1 += 1
            #     if len(example['answers']) != 0 and can_logit <= 0.5:
            #         ref_yes2 += 1
            #     if len(example['answers']) == 0 and can_logit >= 0.5:
            #         ref_no1 += 1
            #     if len(example['answers']) == 0 and can_logit <= 0.5:
            #         ref_no2 += 1
            #
            #     print('question_id is {}, question is {}, pre answers is {}, true answer is {}, can_prob is {}'.format(
            #         example['id'], example['question_text'], [best_answer], example['answers'], can_logit))
            #     fake.append({'question_id': example['id'],
            #                  'question': example['question_text'],
            #                  'question_type': example['question_type'],
            #                  'answers': [best_answer],
            #                  'can_answer': [can_logit]})
            #     fake.append({'question_id': example['id'],
            #                  'question_type': example['question_type'],
            #                  'answers': example['answers']})
            #     pred_count += 1
            #     if len(example['answers']) == 0:
            #         pred_recall += 1
                # best_answer = []
                # continue
            pred_answers.append({'question_id': example['id'],
                                 'question': example['question_text'],
                                 'question_type': example['question_type'],
                                 'answers': [pred_line],
                                 'entity_answers': [[]],
                                 'yesno_answers': [],
                                 'can_answer': [can_logit]})
            if len(example['answers']) == 0:
                example['answers'] = 'no'
            # print('ref is :', example['answers'])
            if 'answers' in example:
                ref_answers.append({'question_id': example['id'],
                                    'question_type': example['question_type'],
                                    'answers': example['answers'],
                                    'entity_answers': [[]],
                                    'yesno_answers': [],
                                    'can_answer': [can_logit]})

        with open(result_file, 'w', encoding='utf-8') as fout:
            for pred_answer in pred_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
        with open("../metric/ref_hype1_0.json", 'w', encoding='utf-8') as fout:
            for pred_answer in ref_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
        # print('in predict', ref_yes1, ref_yes2, ref_no1, ref_no2)
        # print('in all data', ref_yes11, ref_yes22, ref_no11, ref_no22)
        # print('no answer classifier -- acc is: ', pred_recall * 100 / pred_count)
        # print('no answer classifier -- recall is: ', pred_recall * 100 / true_count)
        # print('pred_recall is {}, pred_count is {}, acc_count is {}'.format(pred_recall, pred_count, true_count))
        # print('aa: max ', aa, 'bb: min ', bb)
        # with open("../metric/pre_onlyno_answer.json", 'w', encoding='utf-8') as fout:
        #     for i in fake:
        #         fout.write(json.dumps(i, ensure_ascii=False) + '\n')
        # with open("../metric/true_no_answer.json", 'w', encoding='utf-8') as fout:
        #     for i in true:
        #         fout.write(json.dumps(i, ensure_ascii=False) + '\n')


def eval_all():
    output_model_file = "../model_dir_/best_base_hype4_6"
    output_config_file = "../model_dir/bert_config.json"
    print('output_model_file is {}'.format(output_model_file))
    # config = BertConfig(output_config_file)
    # model = BertForQuestionAnswering(config)
    # # 针对多卡训练加载模型的方法：
    # state_dict = torch.load(output_model_file)
    # # 初始化一个空 dict
    # new_state_dict = OrderedDict()
    # # 修改 key，没有module字段则需要不上，如果有，则需要修改为 module.features
    # for k, v in state_dict.items():
    #     if 'module' not in k:
    #         k = k
    #     else:
    #         k = k.replace('module.', '')
    #     new_state_dict[k] = v
    # model.load_state_dict(new_state_dict)
    model = Translator(use_cuda, output_config_file, output_model_file)
    # model.load_state_dict(torch.load(output_model_file)) #, map_location='cpu'))
    evaluate(model, result_file="../metric/predicts_hype4_6.json")


eval_all()
