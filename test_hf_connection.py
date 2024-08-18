from transformers import AutoModelForCausalLM, AutoTokenizer

model = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model)