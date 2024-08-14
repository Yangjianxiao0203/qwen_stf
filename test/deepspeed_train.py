import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

tokenizer = AutoTokenizer.from_pretrained("/root/models/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("/root/models/Qwen1.5-0.5B")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_files = {
    "train": "/root/data/AdvertiseGen/train.json",
    "dev": "/root/data/AdvertiseGen/dev.json"
}
dataset = load_dataset('json', data_files=data_files, cache_dir="/root/data/AdvertiseGen/cache")
train_dataset = dataset['train']
eval_dataset = dataset['dev']

# 使用自定义数据整理器
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        texts = [item['content'] for item in batch]
        labels = [item['summary'] for item in batch]
        max_length = 512
        encoding = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(device)
        labels_encoding = self.tokenizer(labels, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(device)
        labels_encoding = labels_encoding['input_ids']
        labels_encoding[labels_encoding == tokenizer.pad_token_id] = -100
        return {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask'], 'labels': labels_encoding}

class DeepSpeedArgs:
    def __init__(self):
        self.deepspeed = True
        self.deepspeed_config = 'ds_config.json'  # Path to your DeepSpeed config file
        self.output_dir = './outputs'  # Output directory
        self.do_train = True

deepspeed_args = DeepSpeedArgs()


train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=CustomDataCollator(tokenizer))

# 初始化 DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=deepspeed_args, # 确保传入 DeepSpeed 配置的 args
    model=model,
    model_parameters=model.parameters(),
    # training_data=train_dataset
)

# 训练循环
for epoch in range(3):
    for batch in train_loader:
        model_engine.train()
        outputs = model_engine(**batch)
        model_engine.backward(outputs.loss)
        model_engine.step()
