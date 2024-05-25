from datasets import load_dataset
from transformers import AutoTokenizer


def load_dataset_from_huggingface(dataset_path, subset_name=None, **kwargs):
    dataset = load_dataset(dataset_path, subset_name)

    return dataset


def load_tokenizer_from_huggingface(model_path):
    return AutoTokenizer.from_pretrained(model_path)


if __name__ == '__main__':
    dataset = load_dataset_from_huggingface("nyu-mll/glue", "qnli")
    print(dataset)
    print("*" * 20)
    tokenizer = load_tokenizer_from_huggingface("bert-base-uncased")
    print(tokenizer)

