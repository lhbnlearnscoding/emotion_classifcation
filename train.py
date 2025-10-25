import os, math, torch
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
from tqdm import tqdm # library for progress bars
from sklearn.metrics  import accuracy_score, f1_score, classification_report # for meansuring classification performance
from transformers import get_linear_schedule_with_warmup

from utils import load_config, set_seed, get_device, ensure_dir
from data import build_loaders
from model import build_model

def run_epoch(model, loader, optimizer, scheduler, device, train=True):
    model.train() if train else model.eval()
    losses, preds_all, labels_all = [], [], []

    for batch in tqdm(loader, disable=False, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = cross_entropy(out.logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler: scheduler.step()

        losses.append(loss.item())
        preds_all.extend(out.logits.argmax(dim=-1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())

    acc = accuracy_score(labels_all, preds_all)
    f1m = f1_score(labels_all, preds_all, average="macro")
    return sum(losses)/len(losses), acc, f1m

def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    ensure_dir(cfg["ckpt_dir"])

    # Ép kiểu an toàn
    lr = float(cfg.get("lr", 2e-5))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    epochs = int(cfg.get("epochs", 3))
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 2))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.1))

    # Dùng batch_size/num_workers đã ép kiểu
    train_loader, val_loader, test_loader, tokenizer, id2label, label2id = build_loaders(
        cfg["model_name"], cfg["dataset_name"], cfg["max_length"],
        batch_size, num_workers
    )

    model = build_model(
        cfg["model_name"],
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # Tính steps bằng biến đã ép kiểu
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    # Dùng lr/weight_decay đã là float
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val = 0.0
    best_path = os.path.join(cfg["ckpt_dir"], "best.pt")

    # Dùng epochs đã ép kiểu
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, optimizer=None, scheduler=None, device=device, train=False)
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(), "id2label": id2label, "label2id": label2id}, best_path)
            print(f"  🔸 Saved best to {best_path}")

    # Test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_acc, test_f1 = run_epoch(model, test_loader, optimizer=None, scheduler=None, device=device, train=False)
    print(f"==> Test: loss={test_loss:.4f} acc={test_acc:.4f} f1_macro={test_f1:.4f}")

if __name__ == "__main__":
    main()