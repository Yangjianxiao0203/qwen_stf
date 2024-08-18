import argparse
import json
import logging

import torch
from datasets import load_dataset, load_metric
from modelscope import snapshot_download
from swanlab.integration.huggingface import SwanLabCallback
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    GenerationConfig, TrainerCallback
import os
from peft import LoraConfig, TaskType, get_peft_model

# 设置日志文件
logging.basicConfig(filename='training_log.txt', level=logging.INFO)
logger = logging.getLogger(__name__)


# 设置wandb为离线模式
# wandb.init(mode="offline", project="llama_sft")

MAX_LENGTH = 256

dataset = load_dataset("/root/autodl-tmp/data/alpaca-data-gpt4-chinese")
model_path = "/root/autodl-tmp/models/qwen/Qwen2-1___5B"
output_dir_prefix = "./output/qwen15-alpaca"

#all_dataset = dataset['train'].select(range(500))
all_dataset = dataset['train']
columns_to_remove = ['instruction_zh', 'input_zh', 'output', 'input', 'output_zh', 'instruction']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def process_func(example):
    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer \
        (f"You are a very helpful assistant. Please Answer the following question: \n{example['instruction'] + example['input']}\n",
         add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
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


def train(lora_num):
    # model_path = "/root/autodl-tmp/models/qwen/Qwen2-1___5B"
    output_dir = f"{output_dir_prefix}/r_{lora_num}"
    model_save_path = f"{output_dir_prefix}/final_model_r_{lora_num}"

    # 预处理数据集
    tokenized_dataset = all_dataset.map(process_func, batched=False, remove_columns=columns_to_remove)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    test_dataset = split_dataset["test"]
    train_dataset = split_dataset["train"]
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=lora_num,
        lora_alpha=2*lora_num,
        lora_dropout=0.1
    )

    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        logging_steps=3,
        num_train_epochs=2,
        save_steps=600,
        learning_rate=3e-5,
        weight_decay=0.01,  # 默认参数
        warmup_steps=int(0.33 * (len(tokenized_dataset) // (16 * 4))),
        # save_on_each_node=True,
        # gradient_checkpointing=True,
        # report_to="wandb",
        report_to="none",
    )

    # 创建自定义回调类
    class CustomLoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                try:
                    # 提取需要的日志信息
                    log_info = {key: logs[key] for key in ['loss', 'grad_norm', 'learning_rate', 'epoch'] if
                                key in logs}
                    # 记录到日志文件
                    logger.info(log_info)
                    # 保存到 JSON Lines 文件
                    with open(f'training_logs_r_{lora_num}.jsonl', 'a') as f:
                        json.dump(log_info, f)
                        f.write('\n')
                except Exception as e:
                    pass
    swanlab_callback = SwanLabCallback(
        project="Qwen2-mmlu-fintune",
        experiment_name=f"Qwen2-1.5B-Instruct-lora-alpaca-{lora_num}",
        description="使用通义千问Qwen2-1.5B-Instruct模型在zh_cls_fudan-news数据集上微调。",
        config={
            "model": f"qwen/Qwen2-1.5B-Instruct-lora-{lora_num}",
            "dataset": "mmlu",
        }
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        # eval_dataset=test_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback]
        # compute_metrics=compute_metrics
    )
    try:
        trainer.train(resume_from_checkpoint=True)
    except Exception as e:
        trainer.train()

    model_save_path = model_save_path
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

def main():
    parser = argparse.ArgumentParser(description='Run LoRA training')
    parser.add_argument('--lora_num', type=int, required=True, help='LoRA parameter value')
    args = parser.parse_args()

    lora_num = args.lora_num
    print(f"Starting training with r={lora_num}")
    train(lora_num)
    print(f"Finished training with r={lora_num}")


if __name__ == '__main__':
    # loras = [4,2,8,12,16,32,64]
    # for lora_num in loras:
    #     print(f"current processing r={lora_num}")
    #     train(lora_num)

    main()