from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig
import torch

dataset = load_dataset("/root/autodl-tmp/data/alpaca-data-gpt4-chinese")

# 加载预训练的tokenizer
model_name = "/root/autodl-tmp/models/Llama3-8B-Chinese-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 加载预训练的模型
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
for name, layer in model.named_modules():
    print(f"Layer name: {name}, Layer type: {layer.__class__.__name__}")

def preprocess_function(examples):
    inputs = [f"{instr} {inp}" for instr, inp in zip(examples['instruction_zh'], examples['input_zh'])]
    outputs = examples['output_zh']
    texts = [f"{input} {output}" for input, output in zip(inputs, outputs)]
    return {'text': texts}

# 预处理数据集
tokenized_dataset = dataset.map(preprocess_function, batched=True)

lora_config = LoraConfig(
    r=2,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 根据模型结构调整
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)

def collate_fn(batch):
    inputs = tokenizer([x["text"] for x in batch], padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=8, 
    num_train_epochs=3,
    deepspeed="ds_config.json",
    fp16=True,
    report_to="none"
)
# 创建trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"],  # 通常你会有一个单独的验证集
    tokenizer=tokenizer,
    dataset_text_field="text",
    # data_collator=collate_fn,
)

# 开始训练
trainer.train()
