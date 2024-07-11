import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def evaluate_hellaswag(model, tokenizer, device, dataset_split="validation"):
    # 加载 HellaSwag 数据集
    dataset = load_dataset("hellaswag", split=dataset_split)

    dataset = dataset.select(range(200))

    # 评估指标初始化
    total = 0
    correct = 0

    for example in dataset:
        context = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        # 编码上下文
        input_ids = tokenizer.encode(context, return_tensors="pt").to(device)

        # 对每个结尾进行评分
        scores = []
        for ending in endings:
            input_text = context + " " + ending
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

            # 进行推理
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()  # 获取损失值

            scores.append(loss)

        # 选择损失最低的结尾
        predicted_label = torch.argmin(torch.tensor(scores)).item()

        print(f"correct label: {label}, predicted label: {predicted_label}")

        # 计算准确率
        if predicted_label == label:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    # 加载模型
    # model_path = "C:\jianxiao_codes\python\models\qwen/Qwen2-0.5B\qwen\Qwen2-0___5B"
    model_path = "/root/autodl-tmp/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 评估模型
    accuracy = evaluate_hellaswag(model, tokenizer, device)
    print(f"Accuracy: {accuracy}")