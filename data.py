from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer  

def build_loaders(model_name, dataset_name, max_length=128, batch_size=32, num_workers=2):
    ds = load_dataset(dataset_name)  # splits: train/validation/test
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tok(batch):
        out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
        out["labels"] = batch["label"]
        return out

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(ds["validation"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(ds["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    id2label = {i: l for i, l in enumerate(ds["train"].features["label"].names)}
    label2id = {v:k for k,v in id2label.items()}
    return train_loader, val_loader, test_loader, tokenizer, id2label, label2id