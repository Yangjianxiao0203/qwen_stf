from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# 加载预训练的BERT模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

print("load complete")

# 定义对话函数
def generate_response(prompt, tokenizer, model, max_length=50):
    # 对输入进行编码
    inputs = tokenizer(prompt, return_tensors="pt")

    # 获取输入ID和attention mask
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    # 解码输出ID为文本
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 示例对话
prompt = "Hello, how are you?"
response = generate_response(prompt, tokenizer, model)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
