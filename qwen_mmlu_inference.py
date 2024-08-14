import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json

base_model_name = "/root/autodl-tmp/models/qwen/Qwen2-1___5B"
base_model = True

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
    <|im_start|>system\nYou are an expert in the field of text classification. Please choose the most appropriate option from [A, B, C, D] based on the given context and output only one option, followed directly by "#Answer: " (e.g., "#Answer: A").<|im_end|>\n<|im_start|>user:\n{}<|im_end|>\n<|im_start|>assistant:\n
    """
    if base_model:
        prompt = """You are an expert in the field of text classification. Please choose the most appropriate option from [A, B, C, D] based on the given context and output only one option, followed directly by "#Answer: " (e.g., "#Answer: A"). \n {}"""

    indexs = ["A", "B", "C", "D"]
    user_prompt = f"{context}\n" + "\n".join(
        [f"{index}. {choice}" for index, choice in zip(indexs, choices)])
    prompt = prompt.format(user_prompt)

    return prompt

def form_prompt(dataset):
    """
    获取数据
    """
    data = []
    for example in dataset:
        context = example["question"]
        choices = example["choices"]
        label = example["answer"]
        # print("context: ", context)
        # print("choices: ", choices)
        # print("label: ", label)

        prompt = create_mmlu_prompt(context, choices)
        # print(prompt)
        return (prompt, label)
    return data

def form_prompts(dataset):
    """
    获取数据
    """
    data = []
    for example in dataset:
        context = example["question"]
        choices = example["choices"]
        label = example["answer"]

        prompt = create_mmlu_prompt(context, choices)
        data.append((prompt, label))
    return data


def load_model(adapter_model_name):
    # base_model_name = "/root/autodl-tmp/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B"
    #换成0.5B 了，看看行不行
    '''
    只需要存final models文件夹就可以了，checkpoint文件夹不需要
    '''
    # adapter_model_name = "./output/llama3/final_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    try:
        model = PeftModel.from_pretrained(model, adapter_model_name).to(device)
    except:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer, device


def model_inference(prompt, model, tokenizer, device):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]
    total_length = input_length + 10
    # with torch.no_grad():
    #     outputs = model.generate(input_ids, max_length=total_length, pad_token_id=tokenizer.eos_token_id, temperature=0.1)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, temperature=0.1)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):]
    # print(response)
    return response


def check_answer(answer, correct_label):
    label_map = {
        "0": "A",
        "1": "B",
        "2": "C",
        "3": "D"
    }
    if isinstance(correct_label, int):
        correct_label = str(correct_label)

    if correct_label in label_map:
        correct_label = label_map[correct_label]
    answer = re.search(r"Answer:\s*([A-D])", answer)
    if answer:
        answer = answer.group(1)
        return answer == correct_label, answer
    else:
        return False, answer

def main(model_lora_path):
    data_path = {
        "validation": "/root/autodl-tmp/data/mmlu/all/validation-00000-of-00001.parquet" #1530 rows
    }
    dataset = load_mmlu_data(data_path["validation"])
    # prompt, label = form_prompt(dataset)
    data = form_prompts(dataset)
    # model_lora_path = "./output/llama3/final_model_r_8"
    model, tokenizer, device = load_model(model_lora_path)
    correct = 0
    index = 1
    for prompt, label in data:
        response = model_inference(prompt,model,tokenizer,device)
        resp, answer = check_answer(response, label)
        print(f"{index}.response: {response}\ncorrect_label: {label}\nanswer:{answer} \ncheck answer: {resp}")
        print("*" * 20)
        if resp:
            correct += 1
        index += 1
    print("*" * 20)
    print(f"correct: {correct}, total: {len(data)}")
    accuracy = correct / len(data)
    print(f"accuracy: {accuracy}")
    result = {
        "model" : model_lora_path,
        "correct": correct,
        "total": len(data),
        "accuracy": accuracy
    }
    #存到json
    with open(f"mmlu_result.jsonl", "a+") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    model_lora_paths = [
        # "qwen2_1.5B_base_model"
        # "./output/qwen15/final_model_r_2",
        # "./output/qwen15/final_model_r_4",
        # "./output/qwen15/final_model_r_8",
        # "./output/qwen15/final_model_r_12",
        # "./output/qwen15/final_model_r_16",
        "./output/llama3/final_model_r_32",
        "./output/llama3/final_model_r_64",
    ]
    for model_lora_path in model_lora_paths:
        print("*" * 30)
        print(f"model_lora_path: {model_lora_path}")
        main(model_lora_path)
        print("*" * 20)
