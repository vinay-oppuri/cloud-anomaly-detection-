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

# ── Paths ──────────────────────────────────────────────────────────
PRE_DIR    = "data/processed"
MODEL_DIR  = "models"
MODEL_PATH = "models/network_expert.pth"
META_PATH  = "models/network_meta.json"
PLOT_PATH  = "models/network_curves.png"

# ── Config ─────────────────────────────────────────────────────────
CFG = {
    # Model
    "cnn_channels"  : 128,
    "cnn_kernel"    : 3,
    "lstm_hidden"   : 256,
    "lstm_layers"   : 2,
    "fc_hidden"     : 128,
    "dropout"       : 0.3,

    # Training
    "batch_size"    : 512,
    "epochs"        : 40,
    "lr"            : 1e-3,
    "weight_decay"  : 1e-4,
    "grad_clip"     : 1.0,
    "patience"      : 8,
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
#  MODEL: CNN-LSTM (binary)
# ══════════════════════════════════════════════════════════════════

class CNN_LSTM_Binary(nn.Module):
    """
    CNN-LSTM for binary anomaly detection.

    Input : (batch, seq_len, n_features)
    Output: (batch,) — raw logit, apply sigmoid for probability

    CNN:  extracts spatial correlations between the 80 flow features
          e.g. (bytes/s + packet_count + duration) together = DDoS signature
    LSTM: captures how the flow pattern evolves across seq_len timesteps
          e.g. escalating packet rate over 10 consecutive flows = attack
    """

    def __init__(self, n_features, cnn_channels=128, cnn_kernel=3,
                 lstm_hidden=256, lstm_layers=2,
                 fc_hidden=128, dropout=0.3):
        super().__init__()

        # 1D CNN block
        # Conv1d input: (batch, n_features, seq_len)  — features as channels
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels,
                      kernel_size=cnn_kernel,
                      padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels,
                      kernel_size=cnn_kernel,
                      padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # LSTM block
        self.lstm = nn.LSTM(
            input_size   = cnn_channels,
            hidden_size  = lstm_hidden,
            num_layers   = lstm_layers,
            dropout      = dropout if lstm_layers > 1 else 0.0,
            batch_first  = True,
        )

        # Binary classifier head — outputs single logit
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Linear(lstm_hidden, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

        # Weight init
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, seq_len, n_features)

        # CNN expects (B, n_features, seq_len)
        x = x.permute(0, 2, 1)   # → (B, F, L)
        x = self.cnn(x)           # → (B, C, L)
        x = x.permute(0, 2, 1)   # → (B, L, C)

        # LSTM — take final hidden state
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]               # (B, lstm_hidden)

        return self.head(h).squeeze(1)   # (B,) raw logit


# ══════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y   = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        loss   = criterion(model(X), y)
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
        logits = model(X)
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
    print(f"\n{'═'*55}")
    print(f"  Network Expert — Test Results (thr={thr:.4f})")
    print(f"{'═'*55}")
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
    print(f"Plot → {path}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 55)
    print("  Network Expert — CNN-LSTM Binary Anomaly Detection")
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
    model = CNN_LSTM_Binary(
        n_features  = n_features,
        cnn_channels = CFG["cnn_channels"],
        cnn_kernel   = CFG["cnn_kernel"],
        lstm_hidden  = CFG["lstm_hidden"],
        lstm_layers  = CFG["lstm_layers"],
        fc_hidden    = CFG["fc_hidden"],
        dropout      = CFG["dropout"],
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
    print("─" * 55)

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
                "model_state": model.state_dict(),
                "epoch"      : epoch,
                "n_features" : n_features,
                "seq_len"    : seq_len,
                "cfg"        : CFG,
            }, MODEL_PATH)
            print(f"    ✓ Saved (f1={vf:.4f})")
        else:
            no_improv += 1
            if no_improv >= CFG["patience"]:
                print(f"\n  Early stop at epoch {epoch}")
                break

    print(f"\nDone in {(time.time()-t0)/60:.1f} min | Best F1={best_f1:.4f}")

    # Threshold + final eval
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

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

    print(f"\n✓ Model → {MODEL_PATH}")
    print(f"✓ Meta  → {META_PATH}")


if __name__ == "__main__":
    main()
