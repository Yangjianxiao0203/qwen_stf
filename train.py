from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import torch

tokenizer = AutoTokenizer.from_pretrained("/root/model/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("/root/model/Qwen1.5-0.5B")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = AdamW(model.parameters(), lr=5e-5)
model.to(device)

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
        labels = [item['summary'] for item in batch]  # 获取每个批次中的摘要

        # 对文本和摘要使用相同的最大长度限制确保它们长度一致
        max_length = 512
        encoding = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        labels_encoding = self.tokenizer(labels, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)

        # 确保标签长度与输入长度一致
        labels_encoding = labels_encoding['input_ids']
        labels_encoding[labels_encoding == self.tokenizer.pad_token_id] = -100  # 将pad部分的标签设置为-100，使其在损失计算中被忽略

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels_encoding
        }


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=CustomDataCollator(tokenizer))
eval_loader = DataLoader(eval_dataset, batch_size=4, collate_fn=CustomDataCollator(tokenizer))

# 训练参数
epochs = 3

model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        # 将数据移至相应设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 清除之前的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()

    # 计算平均损失
    avg_train_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")
