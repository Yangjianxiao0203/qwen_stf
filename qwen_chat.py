from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 载入模型和分词器
# tokenizer = AutoTokenizer.from_pretrained("/root/model/Qwen1.5-0.5B")
# model = AutoModelForCausalLM.from_pretrained("/root/model/Qwen1.5-0.5B")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

def chat_with_model(question, max_length=256):
    # 对问题进行编码，添加到批处理中
    inputs = tokenizer.encode(question, return_tensors="pt")
    
    # 生成回答
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    
    # 解码生成的回答
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    question = "who are you"
    resp = chat_with_model(question)
    print(resp)
