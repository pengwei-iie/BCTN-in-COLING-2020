import torch

seed = 42
device = torch.device("cuda", 1)
test_lines = 129932  # 多少条训练数据，即：len(features), 记得修改 !!!!!!!!!!

search_input_file = "../data/extracted/trainset/search.train.json"
zhidao_input_file = "../data/extracted/trainset/zhidao.train.json"
dev_zhidao_input_file = "../data/extracted/devset/zhidao.dev.json"
dev_search_input_file = "../data/extracted/devset/search.dev.json"

max_seq_length = 512
max_ans_length = 128
max_query_length = 60

pretrained_file = "./roberta_wwm_ext"
output_dir = "./model_dir"
predict_example_files='predict.data'

max_para_num=5  # 选择几篇文档进行预测
learning_rate = 5e-5
batch_size = 4
num_train_epochs = 15
gradient_accumulation_steps = 8   # 梯度累积
num_train_optimization_steps = int(test_lines / gradient_accumulation_steps / batch_size) * num_train_epochs
log_step = int(test_lines / batch_size / 4)  # 每个epoch验证几次，默认4次
display_freq = 100

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
