import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "/root/autodl-tmp/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B"
'''
只需要存final models文件夹就可以了，checkpoint文件夹不需要
'''
# adapter_model_name = "./output/llama3/final_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
# model = PeftModel.from_pretrained(model, adapter_model_name).to(device)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
input_text = """
You are an expert in the field of text classification. Please choose the most appropriate option from [A, B, C, D] based on the given context and output only one option, followed directly by "#Answer: " (e.g., "#Answer: A").<|im_end|>

Tonsillar tissue is commonly found
A. on the posterior wall of the oropharynx.
B. under the mucosa of the ventral surface of the tongue.
C. between the palatoglossal and palatopharyngeal folds.
D. at all three sites.

"""
print(f"start asking {input_text}")
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=256, pad_token_id=tokenizer.eos_token_id, temperature=0.1)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 去掉原始输入的部分，保留生成的部分
generated_text_only = generated_text[len(input_text):].strip()

print("*" * 20)
print(generated_text_only)