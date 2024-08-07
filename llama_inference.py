import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "/root/autodl-tmp/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B"
adapter_model_name = "./output/llama3/final_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_model_name).to(device)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

inputs = tokenizer.encode("This movie was really", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))