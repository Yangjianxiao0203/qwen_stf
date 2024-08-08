import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

"""
mmlu 采用cais/mmlu，huggingface上 all中有116k数据
这里面有源码构建：https://github.com/openai/evals/blob/main/examples/mmlu.ipynb
"""

def load_mmlu_data(data_path, dataset_split="train", sample_size=200):
    """
    加载并预处理 MMLU 数据集
    """
    # 加载数据集
    dataset = load_dataset("parquet", data_files=data_path, split=dataset_split)

    # # 选择子集进行测试
    # dataset = dataset.select(range(sample_size))
    return dataset

def create_mmlu_prompt(context, choices):
    """
    构建成这种形式的格式 <|start_header_id|>user<|end_header_id|>\n\n{example['instruction_zh'] + example['input_zh']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt = """
    <|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    indexs = ["A", "B", "C", "D"]
    user_prompt = f"{context}\n" + "\n".join([f"{index}. {choice}" for index, choice in zip(indexs,choices)])
    prompt = prompt.format(user_prompt)

    return prompt

def evaluate_mmlu(model, tokenizer, device, dataset):
    """
    评估模型在 MMLU 数据集上的表现
    """
    # 评估指标初始化
    total = 0
    correct = 0

    for example in dataset:
        context = example["question"]
        choices = example["choices"]
        label = int(example["answer"])

        # 构建prompt
        input_text = f"{context}\n"
        for i, choice in enumerate(choices):
            input_text += f"{i + 1}. {choice}\n"
        input_text += """
        Please provide your answer after #answer: \n#answer:
        """

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # 生成预测
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=256, pad_token_id=tokenizer.eos_token_id,temperature=0.1)
        
        # 解码生成的文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取预测的标签
        match = re.search(r"#answer:\s*(\d+)", generated_text)
        if match:
            predicted_label = int(match.group(1)) - 1
        else:
            predicted_label = -1

        print(f"Correct label: {label}, Predicted label: {predicted_label}, text: {generated_text}")
        print("*" * 20)

        # 计算准确率
        if predicted_label == label:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    # 加载模型
    model_path = "/root/autodl-tmp/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载数据
    data_path = {
        "validation": "/root/autodl-tmp/data/mmlu/astronomy/validation-00000-of-00001.parquet"
    }
    dataset = load_mmlu_data(data_path["validation"])

    # 评估模型
    accuracy = evaluate_mmlu(model, tokenizer, device, dataset)
    print(f"Accuracy: {accuracy}")
