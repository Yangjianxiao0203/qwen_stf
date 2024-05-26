import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader,Subset
from transformers import AutoTokenizer, DataCollatorWithPadding
from data_pipeline.common_fc import load_tokenizer_from_huggingface, load_dataset_from_huggingface


class QNLIDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item['question']
        sentence = item['sentence']
        # 不padding，padding交给collator来做
        # 把question 和sentence合在一起，中间会加上tokenizer的分隔符
        inputs = self.tokenizer(question, sentence, truncation=True, padding=False, max_length=self.max_length,
                                return_tensors="pt")
        inputs['label'] = torch.tensor(item['label']) #分类任务
        return {k: v.squeeze(0) for k, v in inputs.items()}


def build_dataloader_from_dataset(
        dataset,
        tokenizer,
        max_length,
        batch_size,
        data_collator=None,
        **kwargs
):
    qnli_dataset = QNLIDataset(dataset, tokenizer, max_length)
    qnli_loader = DataLoader(
        qnli_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    return qnli_loader


def build_dataloaders(dataset, tokenizer, max_length=256, batch_size=32, **kwargs):
    if kwargs.get("data_collator"):
        data_collator = kwargs["data_collator"]
    else:
        data_collator = DataCollatorWithPadding(tokenizer,
                                                pad_to_multiple_of=8)  # padding到一个8的倍数的位置，256/512/1024 这种，nvidia gpu训练快
    train_loader = build_dataloader_from_dataset(dataset["train"], tokenizer, max_length, batch_size,
                                                 data_collator=data_collator)
    val_loader = build_dataloader_from_dataset(dataset["validation"], tokenizer, max_length, batch_size,
                                               data_collator=data_collator)
    test_loader = build_dataloader_from_dataset(dataset["test"], tokenizer, max_length, batch_size,
                                                data_collator=data_collator)
    return train_loader, val_loader, test_loader

def auto_get_glue_qnli_loaders(num_samples = None):
    dataset = load_dataset_from_huggingface("nyu-mll/glue", "qnli")
    tokenizer = load_tokenizer_from_huggingface("bert-base-uncased")
    max_length = 256
    batch_size = 32
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset, tokenizer,max_length, batch_size
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = auto_get_glue_qnli_loaders()
    for batch in train_loader:
        for key, value in batch.items():
            print(f"{key}: shape {value.shape}")
        break