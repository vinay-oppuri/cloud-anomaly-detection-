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
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.experts.network_expert.model import CNNTransformerClassifier


PRE_DIR_DEFAULT = Path("data/processed")
MODEL_DIR_DEFAULT = Path("models")
MODEL_PATH_DEFAULT = MODEL_DIR_DEFAULT / "network_expert.pth"
META_PATH_DEFAULT = MODEL_DIR_DEFAULT / "network_meta.json"


CFG = {
    "conv_channels": 128,
    "conv_kernel_size": 3,
    "flow_embedding_dim": 128,
    "transformer_heads": 4,
    "transformer_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.2,
    "batch_size": 256,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "patience": 5,
    "seed": 42,
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


@dataclass(slots=True)
class TrainConfig:
    preprocessed_dir: Path
    model_path: Path
    meta_path: Path


class FlowDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the CICIDS2018 network anomaly detector.")
    parser.add_argument("--preprocessed-dir", type=Path, default=PRE_DIR_DEFAULT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH_DEFAULT)
    parser.add_argument("--meta-path", type=Path, default=META_PATH_DEFAULT)
    args = parser.parse_args()
    return TrainConfig(
        preprocessed_dir=args.preprocessed_dir,
        model_path=args.model_path,
        meta_path=args.meta_path,
    )


def main() -> None:
    config = parse_args()
    train_network_model(config)


def train_network_model(config: TrainConfig) -> None:
    seed_everything(CFG["seed"])
    print("=" * 64)
    print("  CICIDS2018 CNN-Transformer Training")
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
        feature_cols,
        seq_len,
    ) = load_data(config.preprocessed_dir)

    train_loader, val_loader, test_loader = make_loaders(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y,
    )
    print(
        f"  Train batches/epoch : {len(train_loader):,} | "
        f"Val batches/epoch : {len(val_loader):,} | "
        f"Test batches : {len(test_loader):,}"
    )

    class_weights = compute_class_weights(train_y=train_y, num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = CNNTransformerClassifier(
        input_dim=train_X.shape[-1],
        num_classes=len(class_names),
        conv_channels=CFG["conv_channels"],
        conv_kernel_size=CFG["conv_kernel_size"],
        flow_embedding_dim=CFG["flow_embedding_dim"],
        transformer_heads=CFG["transformer_heads"],
        transformer_layers=CFG["transformer_layers"],
        dim_feedforward=CFG["dim_feedforward"],
        dropout=CFG["dropout"],
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    config.meta_path.parent.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    started_at = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch=epoch,
            total_epochs=CFG["epochs"],
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            class_names,
            progress_desc=f"Epoch {epoch:02d}/{CFG['epochs']} [val]",
        )
        scheduler.step(val_metrics["macro_f1"])

        checkpoint = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "class_names": class_names,
            "feature_cols": feature_cols,
            "threshold": float(val_metrics["best_threshold"]),
            "config": {
                "input_dim": int(train_X.shape[-1]),
                "num_classes": len(class_names),
                "conv_channels": CFG["conv_channels"],
                "conv_kernel_size": CFG["conv_kernel_size"],
                "flow_embedding_dim": CFG["flow_embedding_dim"],
                "transformer_heads": CFG["transformer_heads"],
                "transformer_layers": CFG["transformer_layers"],
                "dim_feedforward": CFG["dim_feedforward"],
                "dropout": CFG["dropout"],
                "seq_len": seq_len,
            },
        }

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
            torch.save(checkpoint, config.model_path)
        else:
            patience_counter += 1
            if patience_counter >= CFG["patience"]:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    elapsed_minutes = (time.time() - started_at) / 60.0
    print(f"\nTraining finished in {elapsed_minutes:.1f} min. Best epoch: {best_epoch}")

    best_checkpoint = torch.load(config.model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"])

    val_metrics = evaluate(
        model,
        val_loader,
        criterion,
        class_names,
        threshold=float(best_checkpoint["threshold"]),
        progress_desc="Final validation",
    )
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        class_names,
        threshold=float(best_checkpoint["threshold"]),
        progress_desc="Final test",
    )

    metrics_payload = {
        "dataset": "CICIDS2018",
        "device": str(DEVICE),
        "best_epoch": best_epoch,
        "class_names": class_names,
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "threshold": float(best_checkpoint["threshold"]),
        "best_threshold": float(best_checkpoint["threshold"]),
        "threshold_source": "validation_pr_curve",
        "config": CFG,
        "validation": serialise_metrics(val_metrics),
        "test": serialise_metrics(test_metrics),
    }
    config.meta_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Best checkpoint -> {config.model_path}")
    print(f"Metrics         -> {config.meta_path}")


def load_data(
    preprocessed_dir: Path,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
    int,
]:
    def load_tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
        path = preprocessed_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessing artifact: {path}. Run `uv run prepare_cicids` first.")
        tensor = torch.load(path, map_location="cpu", weights_only=False)
        return torch.as_tensor(tensor, dtype=dtype)

    train_X = load_tensor("train_X.pt", torch.float32)
    train_y = load_tensor("train_y.pt", torch.long)
    val_X = load_tensor("val_X.pt", torch.float32)
    val_y = load_tensor("val_y.pt", torch.long)
    test_X = load_tensor("test_X.pt", torch.float32)
    test_y = load_tensor("test_y.pt", torch.long)

    class_info_path = preprocessed_dir / "class_info.json"
    feature_cols_path = preprocessed_dir / "feature_cols.json"
    meta_path = preprocessed_dir / "meta.json"
    if not class_info_path.exists() or not feature_cols_path.exists():
        raise FileNotFoundError(
            "Missing class_info.json or feature_cols.json. Re-run `uv run prepare_cicids` with the new pipeline."
        )

    class_info = json.loads(class_info_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    class_names = [str(item) for item in class_info["class_names"]]
    feature_cols = [str(item) for item in json.loads(feature_cols_path.read_text(encoding="utf-8"))]
    seq_len = int(class_info.get("seq_len", meta.get("seq_len", train_X.shape[1])))

    print(f"[Data] Train={len(train_X):,} Val={len(val_X):,} Test={len(test_X):,}")
    print(f"       Sequence length={seq_len} Features={len(feature_cols)} Classes={class_names}")
    return train_X, train_y, val_X, val_y, test_X, test_y, class_names, feature_cols, seq_len


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
    train_loader = DataLoader(FlowDataset(train_X, train_y), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(FlowDataset(val_X, val_y), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(FlowDataset(test_X, test_y), shuffle=False, **loader_kwargs)
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
    *,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:02d}/{total_epochs} [train]",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    for batch_X, batch_y in progress:
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
        progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    return total_loss / max(1, total_samples)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    class_names: list[str],
    threshold: float | None = None,
    progress_desc: str | None = None,
) -> dict[str, object]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0

    iterator = loader
    if progress_desc is not None:
        iterator = tqdm(
            loader,
            desc=progress_desc,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

    for batch_X, batch_y in iterator:
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
    preds = probs.argmax(axis=1)
    labels = labels_tensor.numpy()

    benign_index = class_names.index("Benign") if "Benign" in class_names else 0
    anomaly_scores = 1.0 - probs[:, benign_index]
    y_true_anomaly = (labels != benign_index).astype(np.int64, copy=False)
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
    try:
        roc_auc = float(roc_auc_score(y_true_anomaly, anomaly_scores))
    except ValueError:
        roc_auc = None

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "anomaly_precision": precision_score(y_true_anomaly, y_pred_anomaly, zero_division=0),
        "anomaly_recall": recall_score(y_true_anomaly, y_pred_anomaly, zero_division=0),
        "anomaly_f1": f1_score(y_true_anomaly, y_pred_anomaly, zero_division=0),
        "roc_auc": roc_auc,
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
