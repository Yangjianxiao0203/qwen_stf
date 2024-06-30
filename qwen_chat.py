from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from accelerate import Accelerator

from utils.download_model import modelscope_download_model

accelerator = Accelerator()

target_dir = "C:\jianxiao_codes\python\models"
model_name = "qwen/Qwen2-0.5B"
cache_file = modelscope_download_model(model_name,target_dir)

model = AutoModelForCausalLM.from_pretrained(
    cache_file,
    torch_dtype="auto",
    # device_map="auto"
)
model = accelerator.prepare(model)
tokenizer = AutoTokenizer.from_pretrained(cache_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

def chat_with_model(question, max_length=256):
    # 对问题进行编码，添加到批处理中
    inputs = tokenizer.encode(question, return_tensors="pt")
    inputs = inputs.to(accelerator.device)
    
    # 生成回答
    outputs = model.generate(inputs, max_length=max_length)
    
    # 解码生成的回答
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    question = "who are you"
    resp = chat_with_model(question)
    print(resp)
