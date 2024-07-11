import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def load_mmlu_data(data_path, dataset_split="train", sample_size=200):
    """
    加载并预处理 MMLU 数据集
    """
    # 加载数据集
    dataset = load_dataset("parquet", data_files=data_path, split=dataset_split)

    # # 选择子集进行测试
    # dataset = dataset.select(range(sample_size))
    return dataset

def evaluate_mmlu(model, tokenizer, device, dataset):
    """
    评估模型在 MMLU 数据集上的表现
    """
    # 评估指标初始化
    total = 0
    correct = 0

    for example in dataset:
        # print(example)
        # break
        
        #1. shot prompt
        context = example["question"]
        choices = example["choices"]
        label = int(example["answer"])
        
        #input: who is president?
        #output: A   I think A is Answer   B
        
        # encode: input + output -> input_ids
        # labels = input_ids
        
        # output = model(input_ids,labels)

        # 对每个选择进行评分
        scores = []
        for choice in choices:
            input_text = context + " " + choice
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

            # 进行推理
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()  # 获取损失值

            scores.append(loss)

        # 选择损失最低的选择
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
    #model_path = "/root/autodl-tmp/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B"
    model_path = "./output/llama3/final_model"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载数据
    data_path = {
        "validation": "/root/autodl-tmp/data/mmlu/all/validation-00000-of-00001.parquet"
    }
    dataset = load_mmlu_data(data_path["validation"])

    # 评估模型
    accuracy = evaluate_mmlu(model, tokenizer, device, dataset)
    print(f"Accuracy: {accuracy}")
