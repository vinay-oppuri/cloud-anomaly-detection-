from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

from src.experts.system_expert.model import TransformerLogClassifier


DATA_PATH_DEFAULT = Path("data/processed/hdfs_processed.pt")
MODEL_DIR_DEFAULT = Path("models")
BEST_MODEL_PATH_DEFAULT = MODEL_DIR_DEFAULT / "system_expert_best.pth"
LAST_MODEL_PATH_DEFAULT = MODEL_DIR_DEFAULT / "system_expert_last.pth"
METRICS_PATH_DEFAULT = MODEL_DIR_DEFAULT / "system_expert_metrics.json"


CFG = {
    "d_model": 160,
    "nhead": 8,
    "num_layers": 3,
    "dim_feedforward": 384,
    "dropout": 0.15,
    "max_len": 256,
    "batch_size": 256,
    "epochs": 24,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "patience": 5,
    "seed": 42,
    "normal_class_index": 0,
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


@dataclass(slots=True)
class TrainConfig:
    data_path: Path
    best_model_path: Path
    last_model_path: Path
    metrics_path: Path


class SequenceDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features.long()
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the HDFS transformer anomaly detector.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH_DEFAULT)
    parser.add_argument("--best-model-path", type=Path, default=BEST_MODEL_PATH_DEFAULT)
    parser.add_argument("--last-model-path", type=Path, default=LAST_MODEL_PATH_DEFAULT)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH_DEFAULT)
    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        best_model_path=args.best_model_path,
        last_model_path=args.last_model_path,
        metrics_path=args.metrics_path,
    )


def main() -> None:
    config = parse_args()
    train_system_model(config)


def train_system_model(config: TrainConfig) -> None:
    seed_everything(CFG["seed"])
    print("=" * 64)
    print("  HDFS Transformer Anomaly Detection Training")
    print("=" * 64)
    print(f"  Device : {DEVICE}")

    (
        train_X,
        train_y,
        val_X,
        val_y,
        test_X,
        test_y,
        class_names,
        vocab_size,
        sequence_length,
    ) = load_data(config.data_path)

    train_loader, val_loader, test_loader = make_loaders(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y,
    )

    class_weights = compute_class_weights(train_y=train_y, num_classes=len(class_names)).to(DEVICE)
    model = TransformerLogClassifier(
        vocab_size=vocab_size,
        num_classes=len(class_names),
        d_model=CFG["d_model"],
        nhead=CFG["nhead"],
        num_layers=CFG["num_layers"],
        dim_feedforward=CFG["dim_feedforward"],
        dropout=CFG["dropout"],
        max_len=max(CFG["max_len"], sequence_length),
        padding_idx=0,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    config.best_model_path.parent.mkdir(parents=True, exist_ok=True)
    config.last_model_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, criterion, class_names)
        scheduler.step(val_metrics["macro_f1"])

        checkpoint = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "class_names": class_names,
            "vocab_size": vocab_size,
            "config": {
                "vocab_size": vocab_size,
                "num_classes": len(class_names),
                "d_model": CFG["d_model"],
                "nhead": CFG["nhead"],
                "num_layers": CFG["num_layers"],
                "dim_feedforward": CFG["dim_feedforward"],
                "dropout": CFG["dropout"],
                "max_len": max(CFG["max_len"], sequence_length),
                "padding_idx": 0,
            },
        }
        torch.save(checkpoint, config.last_model_path)

        print(
            f"  Epoch {epoch:02d}/{CFG['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_anomaly_f1={val_metrics['anomaly_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            patience_counter = 0
            checkpoint["threshold"] = float(val_metrics["best_threshold"])
            torch.save(checkpoint, config.best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= CFG["patience"]:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    elapsed_minutes = (time.time() - start_time) / 60.0
    print(f"\nTraining finished in {elapsed_minutes:.1f} min. Best epoch: {best_epoch}")

    best_checkpoint = torch.load(config.best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"])

    val_metrics = evaluate(model, val_loader, criterion, class_names)
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        class_names,
        threshold=float(best_checkpoint.get("threshold", val_metrics["best_threshold"])),
    )

    metrics_payload = {
        "dataset": "HDFS",
        "device": str(DEVICE),
        "best_epoch": best_epoch,
        "class_names": class_names,
        "vocab_size": vocab_size,
        "sequence_length": sequence_length,
        "config": CFG,
        "validation": serialise_metrics(val_metrics),
        "test": serialise_metrics(test_metrics),
    }
    config.metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Best checkpoint -> {config.best_model_path}")
    print(f"Last checkpoint -> {config.last_model_path}")
    print(f"Metrics         -> {config.metrics_path}")


def load_data(
    path: Path,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    int,
    int,
]:
    if not path.exists():
        raise FileNotFoundError(f"Processed HDFS data not found: {path}. Run `uv run prepare_hdfs` first.")

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    splits = bundle.get("splits")
    if not isinstance(splits, dict):
        raise TypeError("Processed HDFS bundle is missing split tensors.")

    def get_split(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        split = splits[name]
        features = torch.as_tensor(split["X"], dtype=torch.long)
        labels = torch.as_tensor(split["y"], dtype=torch.long)
        return features, labels

    train_X, train_y = get_split("train")
    val_X, val_y = get_split("val")
    test_X, test_y = get_split("test")
    class_names = [str(item) for item in bundle.get("class_names", ["Normal", "Anomaly"])]
    vocab_size = int(bundle.get("vocab_size", int(train_X.max().item()) + 1))
    sequence_length = int(bundle.get("sequence_length", train_X.shape[1]))

    print(f"[Data] Train={len(train_X):,} Val={len(val_X):,} Test={len(test_X):,}")
    print(f"       Sequence length={sequence_length} Vocab size={vocab_size}")
    return train_X, train_y, val_X, val_y, test_X, test_y, class_names, vocab_size, sequence_length


def make_loaders(
    *,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loader_kwargs = {
        "batch_size": CFG["batch_size"],
        "num_workers": 0,
        "pin_memory": DEVICE.type == "cuda",
    }
    train_loader = DataLoader(SequenceDataset(train_X, train_y), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(SequenceDataset(val_X, val_y), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(SequenceDataset(test_X, test_y), shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def compute_class_weights(train_y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(train_y, minlength=num_classes).float()
    counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    weights = counts.sum() / counts
    return weights / weights.mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(DEVICE, non_blocking=True)
        batch_y = batch_y.to(DEVICE, non_blocking=True)
        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        optimizer.step()

        batch_size = int(batch_y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(1, total_samples)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    class_names: list[str],
    threshold: float | None = None,
) -> dict[str, object]:
    model.eval()
    all_labels: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(DEVICE, non_blocking=True)
        batch_y = batch_y.to(DEVICE, non_blocking=True)
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        total_loss += float(loss.item()) * int(batch_y.shape[0])
        total_samples += int(batch_y.shape[0])
        all_logits.append(logits.cpu())
        all_labels.append(batch_y.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    probs = torch.softmax(logits_tensor, dim=-1).numpy()
    labels = labels_tensor.numpy()
    preds = probs.argmax(axis=1)

    normal_idx = class_names.index("Normal") if "Normal" in class_names else 0
    anomaly_scores = 1.0 - probs[:, normal_idx]
    y_true_anomaly = (labels != normal_idx).astype(np.int64, copy=False)

    if threshold is None:
        threshold = compute_best_threshold(y_true_anomaly, anomaly_scores)
    y_pred_anomaly = (anomaly_scores >= threshold).astype(np.int64, copy=False)

    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true_anomaly, y_pred_anomaly, labels=[0, 1]).ravel()

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "anomaly_precision": precision_score(y_true_anomaly, y_pred_anomaly, zero_division=0),
        "anomaly_recall": recall_score(y_true_anomaly, y_pred_anomaly, zero_division=0),
        "anomaly_f1": f1_score(y_true_anomaly, y_pred_anomaly, zero_division=0),
        "best_threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "classification_report": report,
    }


def compute_best_threshold(y_true: np.ndarray, anomaly_scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
    f1_values = np.where(
        (precision + recall) > 0,
        2.0 * precision * recall / (precision + recall),
        0.0,
    )
    best_index = int(np.argmax(f1_values))
    if best_index >= len(thresholds):
        return 0.5
    return float(thresholds[best_index])


def serialise_metrics(metrics: dict[str, object]) -> dict[str, object]:
    serialised: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            serialised[key] = value.item()
        else:
            serialised[key] = value
    return serialised


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
