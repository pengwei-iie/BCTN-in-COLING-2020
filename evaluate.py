import torch
import args


def evaluate(model, dev_data, device_ids):
    total, losses = 0.0, []

    with torch.no_grad():
        model.eval()
        for batch in dev_data:
            input_ids, input_mask, input_ids_q, input_mask_q, answer_ids, answer_mask, \
            segment_ids, can_answer = \
                batch.input_ids, batch.input_mask, batch.input_ids_q, batch.input_mask_q, \
                batch.answer_ids, batch.answer_mask, batch.segment_ids, batch.can_answer

            answer_inputs = answer_ids[:, :-1]  # 去掉EOS
            answer_targets = answer_ids[:, 1:]  # 去掉BOS
            answer_len = answer_mask.sum(1) - 1

            input_ids, input_mask, input_ids_q, input_mask_q, answer_inputs, answer_len, answer_targets, \
            segment_ids, can_answer = \
                input_ids.cuda(), input_mask.cuda(), input_ids_q.cuda(), input_mask_q.cuda(), \
                answer_inputs.cuda(), answer_len.cuda(), answer_targets.cuda(), \
                segment_ids.cuda(), can_answer.cuda()

            # multi-gpu
            # input_ids, input_mask, input_ids_q, input_mask_q, \
            # segment_ids, can_answer, start_positions, end_positions = \
            #     input_ids.cuda(device=device_ids[0]), input_mask.cuda(device=device_ids[0]), \
            #     input_ids_q.cuda(device=device_ids[0]), input_mask_q.cuda(device=device_ids[0]), \
            #     segment_ids.cuda(device=device_ids[0]), can_answer.cuda(device=device_ids[0]), \
            #     start_positions.cuda(device=device_ids[0]), end_positions.cuda(device=device_ids[0])

            # flag = torch.ones(4).cuda(device=device_ids[0])

            loss = model(input_ids, input_ids_q, token_type_ids=segment_ids,
                         attention_mask=input_mask, attention_mask_q=input_mask_q,
                         dec_inputs=answer_inputs, dec_inputs_len=answer_len,
                         dec_targets=answer_targets, can_answer=can_answer)
            loss = loss.mean()
            loss = loss / args.gradient_accumulation_steps
            losses.append(loss.item())

        for i in losses:
            total += i
        with open("./log_hype4_6", 'a') as f:
            f.write("eval_loss: " + str(total / len(losses)) + "\n")

        return total / len(losses)
