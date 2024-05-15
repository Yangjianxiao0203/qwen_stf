from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
# 载入或准备你的数据集
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("/root/model/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("/root/model/Qwen1.5-0.5B")

data_files = {
    "train": "/root/data/AdvertiseGen/train.json",
    "dev": "/root/data/AdvertiseGen/dev.json"
}
dataset = load_dataset('json', data_files=data_files,cache_dir="/root/data/AdvertiseGen/cache")
train_dataset = dataset['train']
eval_dataset = dataset['dev']
print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(eval_dataset))


class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        texts = [item['content'] for item in batch]  # 获取每个批次中的文本
        encoding = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        labels = [item['summary'] for item in batch]
        labels_encoding = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True, max_length=512)
        return {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask'], 'labels': labels_encoding['input_ids']}


# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',              # 输出目录
    num_train_epochs=3,                  # 训练轮数
    per_device_train_batch_size=4,       # 每个设备的批量大小
    per_device_eval_batch_size=8,        # 每个设备的验证批量大小
    # warmup_steps=500,                    # 预热步数，用于学习率调度
    # weight_decay=0.01,                   # 权重衰减，帮助防止过拟合
    # logging_dir='./logs',                # 日志目录
    # logging_steps=10,                    # 每隔多少步打印一次日志
    # evaluation_strategy="steps",         # 在训练期间何时进行评估
    # eval_steps=500,                      # 每500步评估一次模型性能
    save_strategy="epoch",               # 模型保存策略
    save_total_limit=2,                   # 最多保存的模型数量
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=CustomDataCollator(tokenizer),
    tokenizer=tokenizer
)

# 开始训练
trainer.train()
