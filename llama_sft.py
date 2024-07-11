import torch
from datasets import load_dataset,load_metric
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import os
from peft import LoraConfig, TaskType, get_peft_model
import wandb

# 设置wandb为离线模式
wandb.init(mode="offline", project="llama_sft")

# sft: 2-5 epoch, wandb

MAX_LENGTH = 1024   

dataset = load_dataset("/root/autodl-tmp/data/alpaca-data-gpt4-chinese")

all_dataset = dataset['train'].select(range(8000))
#all_dataset = dataset['train']
columns_to_remove = ['instruction_zh', 'input_zh', 'output', 'input', 'output_zh','instruction']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B', use_fast=False, trust_remote_code=True)
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
    
    assert len(input_ids) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
    assert len(attention_mask) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
    assert len(labels) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 预处理数据集
tokenized_dataset = all_dataset.map(process_func, batched=False,remove_columns=columns_to_remove)
# for i in tokenized_dataset:
#     print(i)
#     print("*" * 100)
#     break

split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
# print(split_dataset)
test_dataset = split_dataset["test"]
train_dataset = split_dataset["train"]

def collate_fn(batch):
    inputs = tokenizer([x["text"] for x in batch], padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B', device_map="auto",torch_dtype=torch.bfloat16)
model.enable_input_require_grads() 
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, 
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.1
)

model = get_peft_model(model, config)

model.print_trainable_parameters()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE
    rouge = load_metric("rouge")
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_result = {key: value.mid.fmeasure * 100 for key, value in rouge_result.items()}

    # BLEU
    bleu = load_metric("bleu")
    decoded_preds_bleu = [pred.split() for pred in decoded_preds]
    decoded_labels_bleu = [[label.split()] for label in decoded_labels]
    bleu_result = bleu.compute(predictions=decoded_preds_bleu, references=decoded_labels_bleu)
    bleu_score = bleu_result["bleu"] * 100

    # Combine results
    result = {"bleu": bleu_score, **rouge_result}
    print("result: ",result)
    # Optional: Add additional metrics if needed
    # Example: Perplexity
    # cross_entropy_loss = torch.nn.CrossEntropyLoss()(torch.tensor(predictions), torch.tensor(labels))
    # result["perplexity"] = torch.exp(cross_entropy_loss).item()

    return result


args = TrainingArguments(
    output_dir="./output/llama3",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,  # 默认参数
    warmup_steps=int(0.5 * (len(tokenized_dataset) // (8 * 4))),  # 半个 epoch
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    # eval_dataset=test_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    # compute_metrics=compute_metrics
)

trainer.train(resume_from_checkpoint=True)

model_save_path = "./output/llama3/final_model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)