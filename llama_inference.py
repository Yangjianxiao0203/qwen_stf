import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "/root/autodl-tmp/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B"
adapter_model_name = "./output/llama3/final_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_model_name).to(device)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
input_text = "this movie is very"
print(f"start asking {input_text}")
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=256, pad_token_id=tokenizer.eos_token_id, temperature=0.1)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))