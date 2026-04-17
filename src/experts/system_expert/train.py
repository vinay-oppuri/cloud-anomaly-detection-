import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from src.experts.system_expert.model import TransformerLogClassifier

# ── Paths ──────────────────────────────────────────────────────────
DATA_PATH  = "data/processed/hdfs_processed.pt"
MODEL_DIR  = "models"
MODEL_PATH = "models/system_expert_best.pth"
LAST_PATH  = "models/system_expert_last.pth"
META_PATH  = "models/system_expert_metrics.json"

# ── Config ─────────────────────────────────────────────────────────
CFG = {
    # Model
    "d_model"       : 192,
    "nhead"         : 6,
    "num_layers"    : 3,
    "dim_feedforward": 512,
    "dropout"       : 0.2,
    "max_len"       : 512,

    # Training
    "batch_size"    : 256,
    "epochs"        : 35,
    "lr"            : 3e-4,
    "weight_decay"  : 1e-4,
    "patience"      : 8,
    "seed"          : 42,
    "normal_class"  : 0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found - run: uv run prepare_system")

    bundle = torch.load(DATA_PATH, map_location="cpu")
    splits = bundle["splits"]

    def get_split(name):
        s = splits[name]
        X = torch.as_tensor(s.get("X", s.get("features")), dtype=torch.long)
        y = torch.as_tensor(s.get("y", s.get("labels")),   dtype=torch.long)
        return X, y

    train_X, train_y = get_split("train")
    val_X,   val_y   = get_split("val")
    test_X,  test_y  = get_split("test")

    # resolve class names
    raw = bundle.get("class_names")
    if isinstance(raw, (list, tuple)) and raw:
        class_names = [str(c) for c in raw]
    else:
        n = int(train_y.max().item()) + 1
        class_names = [f"class_{i}" for i in range(n)]

    # resolve vocab size
    vocab_size = bundle.get("vocab_size")
    if not (isinstance(vocab_size, int) and vocab_size >= 2):
        vocab_size = max(int(t.max().item()) for t in (train_X, val_X, test_X)) + 1

    print(f"[Data] Train:{len(train_X):,}  Val:{len(val_X):,}  Test:{len(test_X):,}")
    print(f"  Shape     : {tuple(train_X.shape)}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  classes   : {class_names}\n")

    return train_X, train_y, val_X, val_y, test_X, test_y, class_names, vocab_size


def make_loaders(train_X, train_y, val_X, val_y, test_X, test_y):
    pin = DEVICE.type == "cuda"
    kw  = dict(pin_memory=pin, num_workers=0)
    tr  = DataLoader(SequenceDataset(train_X, train_y),
                     batch_size=CFG["batch_size"], shuffle=True,  **kw)
    va  = DataLoader(SequenceDataset(val_X, val_y),
                     batch_size=CFG["batch_size"], shuffle=False, **kw)
    te  = DataLoader(SequenceDataset(test_X, test_y),
                     batch_size=CFG["batch_size"], shuffle=False, **kw)
    return tr, va, te


# ══════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        loss = criterion(model(X), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y); n += len(y)
    return total / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_labels, all_logits = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        all_logits.append(model(X).cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss   = nn.CrossEntropyLoss()(logits, labels).item()
    preds  = logits.argmax(dim=1).numpy()
    return loss, preds, labels.numpy()


def print_results(y_true, y_pred, class_names):
    print(f"\n{'='*55}")
    print("  System Expert - Test Results")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred,
                                target_names=class_names, digits=4))
    if len(class_names) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        f1  = f1_score(y_true, y_pred, zero_division=0)
        print(f"  F1={f1:.4f}")
        print(f"  Caught : {tp}/{tp+fn}  ({100*tp/(tp+fn+1e-9):.1f}%)")
        print(f"  False+ : {fp}")
        print(f"  Missed : {fn}")
        return {"f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn)}
    else:
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"  Macro-F1={macro_f1:.4f}")
        return {"macro_f1": macro_f1}


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 55)
    print("  System Expert - Transformer Log Anomaly Detection")
    print("  Dataset : HDFS")
    print("=" * 55)
    print(f"  Device  : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load
    (train_X, train_y, val_X, val_y, test_X, test_y,
     class_names, vocab_size) = load_data()

    tr_l, va_l, te_l = make_loaders(
        train_X, train_y, val_X, val_y, test_X, test_y)

    # Model
    model = TransformerLogClassifier(
        vocab_size     = max(vocab_size, 2),
        num_classes    = len(class_names),
        d_model        = CFG["d_model"],
        nhead          = CFG["nhead"],
        num_layers     = CFG["num_layers"],
        dim_feedforward= CFG["dim_feedforward"],
        dropout        = CFG["dropout"],
        max_len        = CFG["max_len"],
    ).to(DEVICE)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Params  : {total_p:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    # Train
    print(f"Training {CFG['epochs']} epochs (patience={CFG['patience']})...")
    print("-" * 55)

    best_f1   = 0.0
    no_improv = 0
    t0        = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss = train_epoch(model, tr_l, optimizer, criterion)

        val_loss, val_preds, val_labels = evaluate(model, va_l)
        vf = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        scheduler.step(val_loss)

        print(f"  Ep {epoch:2d}/{CFG['epochs']} | "
              f"Loss:{tr_loss:.4f} | "
              f"Val Loss:{val_loss:.4f} | "
              f"Val Macro-F1:{vf:.4f} | "
              f"LR:{optimizer.param_groups[0]['lr']:.1e}")

        # Save last checkpoint every epoch
        torch.save({
            "state_dict" : model.state_dict(),
            "epoch"      : epoch,
            "class_names": class_names,
            "vocab_size"  : vocab_size,
            "config"     : CFG,
        }, LAST_PATH)

        if vf > best_f1:
            best_f1   = vf
            no_improv = 0
            torch.save({
                "state_dict" : model.state_dict(),
                "epoch"      : epoch,
                "class_names": class_names,
                "vocab_size"  : vocab_size,
                "config"     : CFG,
            }, MODEL_PATH)
            print(f"    [saved] macro_f1={vf:.4f}")
        else:
            no_improv += 1
            if no_improv >= CFG["patience"]:
                print(f"\n  Early stop at epoch {epoch}")
                break

    print(f"\nDone in {(time.time()-t0)/60:.1f} min | Best Macro-F1={best_f1:.4f}")

    # Final test evaluation
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    _, test_preds, test_labels = evaluate(model, te_l)
    results = print_results(test_labels, test_preds, class_names)

    with open(META_PATH, "w") as f:
        json.dump({
            "class_names": class_names,
            "vocab_size" : vocab_size,
            "cfg"        : CFG,
            "test"       : results,
            "dataset"    : "HDFS",
        }, f, indent=2)

    print(f"\nModel -> {MODEL_PATH}")
    print(f"Meta  -> {META_PATH}")


if __name__ == "__main__":
    main()
