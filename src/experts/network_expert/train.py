"""
train.py
========
CNN-LSTM for binary network anomaly detection on CICIDS2018.
  Output: 0 = Benign, 1 = Anomaly

Input  : data/processed/  (from process.py)
Output : models/network_expert.pth
         models/network_meta.json

Run:
  uv run train_cicids
"""

import os
import json
import time
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, roc_auc_score,
                              classification_report,
                              confusion_matrix,
                              precision_recall_curve)

from src.experts.network_expert.model import CNNTransformerClassifier

# ── Paths ──────────────────────────────────────────────────────────
PRE_DIR    = "data/processed"
MODEL_DIR  = "models"
MODEL_PATH = "models/network_expert.pth"
META_PATH  = "models/network_meta.json"
PLOT_PATH  = "models/network_curves.png"

# ── Config ─────────────────────────────────────────────────────────
CFG = {
    # Model
    "conv_channels"       : 128,
    "conv_kernel_size"    : 3,
    "flow_embedding_dim"  : 128,
    "transformer_heads"   : 4,
    "transformer_layers"  : 2,
    "dim_feedforward"     : 256,
    "dropout"             : 0.3,

    # Training
    "batch_size"    : 512,
    "epochs"        : 4,
    "lr"            : 1e-3,
    "weight_decay"  : 1e-4,
    "grad_clip"     : 1.0,
    "patience"      : 2,
    "seed"          : 42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# ══════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════

class FlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.float()   # float for BCEWithLogitsLoss
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def load_data():
    def pt(name):
        p = os.path.join(PRE_DIR, name)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found - run: uv run prepare_cicids")
        return torch.load(p, weights_only=True)

    train_X, train_y = pt("train_X.pt"), pt("train_y.pt")
    val_X,   val_y   = pt("val_X.pt"),   pt("val_y.pt")
    test_X,  test_y  = pt("test_X.pt"),  pt("test_y.pt")

    with open(os.path.join(PRE_DIR, "meta.json")) as f:
        meta = json.load(f)

    pos_weight = torch.tensor([meta["pos_weight"]], dtype=torch.float32)

    print(f"[Data] Train:{len(train_X):,} Val:{len(val_X):,} "
          f"Test:{len(test_X):,}")
    print(f"  Shape   : {tuple(train_X.shape)}")
    print(f"  pos_weight = {pos_weight.item():.2f}  "
          f"(anomaly penalized {pos_weight.item():.1f}x more)\n")

    return (train_X, train_y, val_X, val_y, test_X, test_y,
            pos_weight, meta)


def make_loaders(train_X, train_y, val_X, val_y, test_X, test_y):
    nw  = 0 if platform.system() == "Windows" else 4
    pin = DEVICE.type == "cuda"
    kw  = dict(num_workers=nw, pin_memory=pin)

    tr = DataLoader(FlowDataset(train_X, train_y),
                    batch_size=CFG["batch_size"], shuffle=True,  **kw)
    va = DataLoader(FlowDataset(val_X, val_y),
                    batch_size=CFG["batch_size"], shuffle=False, **kw)
    te = DataLoader(FlowDataset(test_X, test_y),
                    batch_size=CFG["batch_size"], shuffle=False, **kw)
    return tr, va, te


# ══════════════════════════════════════════════════════════════════
#  MODEL: CNN-Transformer (binary)
# ══════════════════════════════════════════════════════════════════
# We now use the CNNTransformerClassifier imported from model.py
# and we squeeze the final dimension (-1) since num_classes=1.


# ══════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y   = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        loss   = criterion(model(X).squeeze(-1), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(y); n += len(y)
    return total / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    yl, yp = [], []
    for X, y in loader:
        X      = X.to(DEVICE, non_blocking=True)
        logits = model(X).squeeze(-1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        yl.extend(y.numpy())
        yp.extend(probs)
    return np.array(yl), np.array(yp)


def best_threshold(y_true, y_probs):
    """F1-maximising threshold on val set."""
    p, r, t = precision_recall_curve(y_true, y_probs)
    f1 = np.where((p+r) == 0, 0, 2*p*r/(p+r))
    idx = int(np.argmax(f1))
    return float(t[idx]) if idx < len(t) else 0.5


def print_results(y_true, y_probs, thr):
    y_pred = (y_probs >= thr).astype(int)
    print(f"\n{'='*55}")
    print(f"  Network Expert - Test Results (thr={thr:.4f})")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred,
                                target_names=["Benign", "Anomaly"],
                                digits=4))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    print(f"  F1={f1:.4f}  AUC={auc:.4f}")
    print(f"  Caught : {tp}/{tp+fn}  ({100*tp/(tp+fn+1e-9):.1f}%)")
    print(f"  False+ : {fp}")
    print(f"  Missed : {fn}")
    return {"f1":f1, "auc":auc, "threshold":thr,
            "tp":int(tp), "fp":int(fp), "fn":int(fn)}


def save_plot(tr_l, va_f1, path):
    ep = range(1, len(tr_l)+1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, tr_l, marker="o", ms=3)
    a1.set_title("Training Loss"); a1.grid(alpha=.3)
    a2.plot(ep, va_f1, color="green", marker="^", ms=3)
    a2.set_title("Val F1"); a2.set_ylim(0, 1); a2.grid(alpha=.3)
    plt.suptitle("Network Expert — CNN-LSTM Binary (CICIDS2018)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Plot -> {path}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 55)
    print("  Network Expert - CNN-LSTM Binary Anomaly Detection")
    print("  Dataset : CSE-CICIDS2018  (Binary: Benign / Anomaly)")
    print("=" * 55)
    print(f"  Device  : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load
    (train_X, train_y, val_X, val_y, test_X, test_y,
     pos_weight, meta) = load_data()

    n_features = meta["n_features"]
    seq_len    = meta["seq_len"]

    tr_l, va_l, te_l = make_loaders(
        train_X, train_y, val_X, val_y, test_X, test_y)

    # Model
    model = CNNTransformerClassifier(
        input_dim           = n_features,
        num_classes         = 1,
        conv_channels       = CFG["conv_channels"],
        conv_kernel_size    = CFG["conv_kernel_size"],
        flow_embedding_dim  = CFG["flow_embedding_dim"],
        transformer_heads   = CFG["transformer_heads"],
        transformer_layers  = CFG["transformer_layers"],
        dim_feedforward     = CFG["dim_feedforward"],
        dropout             = CFG["dropout"],
    ).to(DEVICE)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Params  : {total_p:,}\n")

    # Loss: BCEWithLogitsLoss + pos_weight for class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = CFG["lr"],
        steps_per_epoch = len(tr_l),
        epochs          = CFG["epochs"],
        pct_start       = 0.1,
        anneal_strategy = "cos",
    )

    # Train
    print(f"Training {CFG['epochs']} epochs (patience={CFG['patience']})...")
    print("-" * 55)

    best_f1   = 0.0
    no_improv = 0
    tr_losses = []
    va_f1s    = []
    t0        = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        tr = train_epoch(model, tr_l, optimizer, scheduler, criterion)

        vy, vp  = evaluate(model, va_l)
        vf      = f1_score(vy, (vp >= 0.5).astype(int), zero_division=0)

        try:
            vauc = roc_auc_score(vy, vp)
        except Exception:
            vauc = 0.0

        tr_losses.append(tr); va_f1s.append(vf)

        print(f"  Ep {epoch:2d}/{CFG['epochs']} | "
              f"Loss:{tr:.4f} | "
              f"Val F1:{vf:.4f} | AUC:{vauc:.4f} | "
              f"LR:{optimizer.param_groups[0]['lr']:.1e}")

        if vf > best_f1:
            best_f1   = vf
            no_improv = 0
            torch.save({
                "state_dict" : model.state_dict(),
                "epoch"      : epoch,
                "n_features" : n_features,
                "seq_len"    : seq_len,
                "config"     : CFG,
            }, MODEL_PATH)
            print(f"    [saved] f1={vf:.4f}")
        else:
            no_improv += 1
            if no_improv >= CFG["patience"]:
                print(f"\n  Early stop at epoch {epoch}")
                break

    print(f"\nDone in {(time.time()-t0)/60:.1f} min | Best F1={best_f1:.4f}")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    vy, vp = evaluate(model, va_l)
    thr    = best_threshold(vy, vp)
    print(f"Optimal threshold: {thr:.4f}")

    # Update checkpoint with threshold
    ckpt["threshold"] = thr
    torch.save(ckpt, MODEL_PATH)

    ty, tp_probs = evaluate(model, te_l)
    results      = print_results(ty, tp_probs, thr)

    save_plot(tr_losses, va_f1s, PLOT_PATH)

    with open(META_PATH, "w") as f:
        json.dump({
            "n_features" : n_features,
            "seq_len"    : seq_len,
            "threshold"  : thr,
            "cfg"        : CFG,
            "test"       : results,
            "dataset"    : "CSE-CICIDS2018 (binary)",
        }, f, indent=2)

    print(f"\nModel -> {MODEL_PATH}")
    print(f"Meta  -> {META_PATH}")


if __name__ == "__main__":
    main()
