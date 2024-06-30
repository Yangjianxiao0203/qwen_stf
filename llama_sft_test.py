import torch
from datasets import load_dataset,DatasetDict,Dataset
from modelscope import snapshot_download
# from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import os
from peft import LoraConfig, TaskType, get_peft_model

# sft: 2-5 epoch, wandb

MAX_LENGTH = 612   

dataset = load_dataset("/root/autodl-tmp/code/test_data/alpaca-data-gpt4-chinese")
dataset = dataset["train"].select(range(10000))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def process_func(example):
     # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction_zh'] + example['input_zh']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    else:
        padding_length = MAX_LENGTH - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length  
        labels = labels + [-100] * padding_length  
    
    # Ensure lengths are consistent
    assert len(input_ids) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
    assert len(attention_mask) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
    assert len(labels) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }



# 预处理数据集
tokenized_dataset = dataset.map(process_func, batched=False,remove_columns=dataset.column_names)

# for i in tokenized_dataset:
#     print(i)
#     print("*" * 100)
#     break

def collate_fn(batch):
    inputs = tokenizer([x["text"] for x in batch], padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct', device_map="auto",torch_dtype=torch.bfloat16)
model.enable_input_require_grads() 
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, 
    lora_dropout=0.1
)

model = get_peft_model(model, config)

model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./output/llama3",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()