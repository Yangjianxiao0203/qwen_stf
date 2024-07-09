import os
import json
import argparse
from lm_eval.base import Task, rf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator, tasks

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model with local dataset")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B", help="Path to the model")
    parser.add_argument("--dataset_path", type=str, default="/root/autodl-tmp/data/wikitext", help="Path to the local dataset")
    return parser.parse_args()


args = parse_args()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和tokenizer
model_path = args.model_path
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)

# 定义一个函数包装模型和tokenizer
class CustomModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, context, max_length, eos_token_id):
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(device)
        output = self.model.generate(input_ids, max_length=max_length, eos_token_id=eos_token_id)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def __call__(self, prompt):
        return self.generate(prompt, max_length=1024, eos_token_id=self.tokenizer.eos_token_id)

# 包装模型
custom_model = CustomModel(model, tokenizer)

# 定义自定义任务类
class LocalWikiTextTask(Task):
    VERSION = 0
    DATASET_PATH = args.dataset_path  # 从命令行参数获取本地数据集路径

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        with open(self.DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

    def fewshot_examples(self, k, rnd):
        return []

    def doc_to_text(self, doc):
        return doc["prompt"]

    def doc_to_target(self, doc):
        return doc["completion"]

    def construct_requests(self, doc, ctx):
        continuation = rf.greedy_until(ctx, until=["\n"])
        return continuation

    def process_results(self, doc, results):
        target = self.doc_to_target(doc)
        prediction = results[0]
        return {
            "acc": int(prediction.strip() == target.strip())
        }

    def aggregation(self):
        return {
            "acc": rf.mean
        }

    def higher_is_better(self):
        return {
            "acc": True
        }

# 注册自定义任务
tasks.TASK_REGISTRY['local_wikitext'] = LocalWikiTextTask

# 定义要评测的任务
task_dict = tasks.get_task_dict(["local_wikitext"])

# 运行评测
results = evaluator.evaluate(
    model=custom_model,
    tasks=task_dict,
    batch_size=8,  # 可以根据需要调整
    device=device,
    num_fewshot=0  # 零样本学习
)

# 打印结果
print(results)
