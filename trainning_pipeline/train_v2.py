from evaluate import load
import numpy as np
import torch
from datasets import load_metric
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from data_pipeline.alpaca import build_alpaca_data
from accelerate import Accelerator
import wandb


# wandb.login(key="d7ff37534eeca77d459824f5c4be2958e279ec07")

# Initialize wandb project
# wandb.init(project="qwen_0.5b_sft")
wandb.init(mode="offline", project="qwen_0.5b_sft")
#wandb.init(project="qwen_0.5b_sft",mode="offline")

# 困惑度计算函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    valid_indices = labels != -100
    predictions = np.argmax(logits, axis=-1)
    predictions = predictions[valid_indices].flatten()
    references = labels[valid_indices].flatten()

    # 加载准确率度量
    accuracy_metric = load("accuracy")

    accuracy = accuracy_metric.compute(predictions=predictions, references=references)

    # return {"perplexity": perplexity, "accuracy": accuracy["accuracy"]}
    return {"accuracy": accuracy["accuracy"]}

def train(model_path):
    model =AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = build_alpaca_data(tokenizer,size=12)

    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=5e-5,
    #     per_device_train_batch_size=6,
    #     per_device_eval_batch_size=6,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    #     push_to_hub=False,
    #     logging_dir="./logs",
    #     report_to="wandb"
    # )
    
    training_args = TrainingArguments(
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
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save the model if needed
    trainer.save_model("output_model_path")
    tokenizer.save_pretrained("output_model_path")


if __name__ == "__main__":
    train("C:\jianxiao_codes\python\models\qwen/Qwen2-0.5B\qwen\Qwen2-0___5B")


