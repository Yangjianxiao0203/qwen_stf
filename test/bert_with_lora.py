from peft import LoraConfig, get_peft_model
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForMaskedLM

# 加载预训练的BERT模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# 定义 LoRA 配置
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["query", "key", "value"],  # 仅在这些模块上应用 LoRA
    lora_dropout=0.1,
    bias="none"
)

# 使用 peft 库的 get_peft_model 方法将 LoRA 配置应用到模型
lora_model = get_peft_model(model, lora_config)

# 检查模型结构
print(lora_model)

# 假设我们有一些训练数据
train_texts = ["Hello, how are you?", "What is your name?", "Tell me a joke."]
train_labels = train_texts  # 对于掩码语言模型，标签就是输入本身

# 预处理数据
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
train_labels = tokenizer(train_labels, truncation=True, padding=True, return_tensors="pt").input_ids

# 创建数据集
class MaskedLMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MaskedLMDataset(train_encodings, train_labels)

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    logging_dir='./logs',
)

# 创建 Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
)

# 训练模型
trainer.train()
