import torch

seed = 42
device = torch.device("cuda", 0)
test_lines = 101250  # 多少条训练数据，即：len(features)

search_input_file = "../data/extracted/trainset/search.train.json"
zhidao_input_file = "../data/extracted/trainset/zhidao.train.json"
dev_zhidao_input_file = "../data/extracted/devset/zhidao.dev.json"
dev_search_input_file = "../data/extracted/devset/search.dev.json"

max_seq_length = 512
max_ans_length = 128
max_query_length = 60

output_dir = "./model_dir_"
predict_example_files='predict.data'

max_para_num=5  # 选择几篇文档进行预测
learning_rate = 5e-5
batch_size = 1
num_train_epochs = 4
gradient_accumulation_steps = 8   # 梯度累积
num_train_optimization_steps = int(test_lines / gradient_accumulation_steps / batch_size) * num_train_epochs
log_step = int(test_lines / batch_size / 4)  # 每个epoch验证几次，默认4次

# decoder
n_layers = 4
d_k = 64
d_v = 64
d_model = 256
d_ff = 1024
n_heads = 4
dropout = 0.1
tgt_vocab_size = 21128  # 记得修改
weighted_model = False  # decoder是否保留attention


# predict
beam_size = 5
pre_batch = 1
n_best = 1
max_decode_step = max_ans_length

