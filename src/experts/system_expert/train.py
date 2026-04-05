from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

from src.experts.system_expert.model import TransformerLogClassifier
from src.training.checkpointing import (
    resolve_checkpoint_paths,
    save_metrics_report,
    save_model_checkpoint,
)
from src.training.data import SequenceDataset
from src.training.metrics import compute_classification_report
from src.training.runner import build_dataloader, evaluate_model, set_global_seed, train_one_epoch

DEFAULT_PROCESSED_PATH = Path("data/processed/hdfs_processed.pt")


@dataclass(slots=True)
class SystemTransformerTrainConfig:
    processed_data: Path
    output_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    seed: int
    device: str
    normal_class_index: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    max_len: int


def parse_args() -> SystemTransformerTrainConfig:
    parser = argparse.ArgumentParser(
        description="Train transformer system anomaly expert from hdfs_processed.pt."
    )
    parser.add_argument("--processed-data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normal-class-index", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-len", type=int, default=512)
    ns = parser.parse_args()
    return SystemTransformerTrainConfig(
        processed_data=ns.processed_data,
        output_dir=ns.output_dir,
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        learning_rate=ns.learning_rate,
        weight_decay=ns.weight_decay,
        patience=ns.patience,
        seed=ns.seed,
        device=ns.device,
        normal_class_index=ns.normal_class_index,
        d_model=ns.d_model,
        nhead=ns.nhead,
        num_layers=ns.num_layers,
        dim_feedforward=ns.dim_feedforward,
        dropout=ns.dropout,
        max_len=ns.max_len,
    )


def main() -> None:
    config = parse_args()
    summary = run_training(config)
    print(json.dumps(summary, indent=2))


def run_training(config: SystemTransformerTrainConfig) -> dict[str, Any]:
    if not config.processed_data.exists():
        raise FileNotFoundError(
            f"Processed HDFS dataset not found: {config.processed_data}. "
            "Run prepare-system first."
        )

    set_global_seed(config.seed)
    device = _resolve_device(config.device)
    print(_format_device_info(requested=config.device, resolved=device))
    bundle = torch.load(config.processed_data, map_location="cpu")

    train_X, train_y = _load_split(bundle, "train")
    val_X, val_y = _load_split(bundle, "val")
    test_X, test_y = _load_split(bundle, "test")
    class_names = _resolve_class_names(bundle, train_y)

    if train_X.ndim != 2:
        raise ValueError(f"Expected train tokens [N, seq_len], got {tuple(train_X.shape)}")

    inferred_vocab_size = _resolve_vocab_size(bundle, tensors=(train_X, val_X, test_X))
    model = TransformerLogClassifier(
        vocab_size=max(inferred_vocab_size, 2),
        num_classes=len(class_names),
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_len=config.max_len,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = build_dataloader(
        SequenceDataset(train_X, train_y),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = build_dataloader(
        SequenceDataset(val_X, val_y),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = build_dataloader(
        SequenceDataset(test_X, test_y),
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Friendly runtime summary for easier monitoring/debugging.
    print(
        "Training setup | "
        f"device={device} "
        f"train={int(train_X.shape[0])} "
        f"val={int(val_X.shape[0])} "
        f"test={int(test_X.shape[0])} "
        f"seq_len={int(train_X.shape[1])} "
        f"vocab={inferred_vocab_size}"
    )

    paths = resolve_checkpoint_paths(config.output_dir, prefix="system_expert")

    best_macro_f1 = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            input_dtype="long",
            progress_desc="train",
        )
        val_loss, val_logits, val_labels = evaluate_model(
            model,
            val_loader,
            criterion,
            device=device,
            input_dtype="long",
            progress_desc="val",
        )
        val_report = compute_classification_report(
            loss=val_loss,
            labels=val_labels,
            logits=val_logits,
            class_names=class_names,
            normal_class_index=config.normal_class_index,
        )
        scheduler.step(val_loss)
        print(
            "Epoch summary | "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_f1={val_report.macro_f1:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_report.to_dict(),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)

        save_model_checkpoint(
            paths.last_model_path,
            model_state_dict=model.state_dict(),
            class_names=list(class_names),
            config=_model_config_payload(config=config, vocab_size=inferred_vocab_size, num_classes=len(class_names)),
            epoch=epoch,
            metrics=val_report.to_dict(),
        )

        if val_report.macro_f1 > best_macro_f1:
            best_macro_f1 = val_report.macro_f1
            best_epoch = epoch
            best_metrics = val_report.to_dict()
            epochs_without_improvement = 0
            save_model_checkpoint(
                paths.best_model_path,
                model_state_dict=model.state_dict(),
                class_names=list(class_names),
                config=_model_config_payload(
                    config=config,
                    vocab_size=inferred_vocab_size,
                    num_classes=len(class_names),
                ),
                epoch=epoch,
                metrics=val_report.to_dict(),
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    checkpoint = torch.load(paths.best_model_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    test_loss, test_logits, test_labels = evaluate_model(
        model,
        test_loader,
        criterion,
        device=device,
        input_dtype="long",
        progress_desc="test",
    )
    test_report = compute_classification_report(
        loss=test_loss,
        labels=test_labels,
        logits=test_logits,
        class_names=class_names,
        normal_class_index=config.normal_class_index,
    )

    summary: dict[str, Any] = {
        "task": "system_expert_transformer_training",
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_model_path": str(paths.best_model_path),
        "last_model_path": str(paths.last_model_path),
        "metrics": {
            "best_val": best_metrics,
            "test": test_report.to_dict(),
            "history": history,
        },
        "config": {
            **asdict(config),
            "processed_data": str(config.processed_data),
            "output_dir": str(config.output_dir),
        },
    }
    save_metrics_report(paths.metrics_path, summary)
    return summary


def _load_split(bundle: dict[str, Any], split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    splits = bundle.get("splits")
    if not isinstance(splits, dict) or split_name not in splits:
        raise KeyError(f"Missing split '{split_name}' in processed bundle.")

    split = splits[split_name]
    if not isinstance(split, dict):
        raise TypeError(f"Split '{split_name}' payload must be a dict.")

    features = split.get("X", split.get("features"))
    labels = split.get("y", split.get("labels"))
    if features is None or labels is None:
        raise KeyError(f"Split '{split_name}' must contain X/y or features/labels.")

    return torch.as_tensor(features, dtype=torch.long), torch.as_tensor(labels, dtype=torch.long)


def _resolve_class_names(bundle: dict[str, Any], labels: torch.Tensor) -> list[str]:
    raw = bundle.get("class_names")
    if isinstance(raw, (list, tuple)) and raw:
        return [str(item) for item in raw]

    num_classes = int(labels.max().item()) + 1
    return [f"class_{idx}" for idx in range(num_classes)]


def _resolve_vocab_size(bundle: dict[str, Any], tensors: tuple[torch.Tensor, ...]) -> int:
    bundle_vocab = bundle.get("vocab_size")
    if isinstance(bundle_vocab, int) and bundle_vocab >= 2:
        return bundle_vocab

    max_id = 1
    for tensor in tensors:
        if tensor.numel() == 0:
            continue
        max_id = max(max_id, int(tensor.max().item()))
    return max_id + 1


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _format_device_info(*, requested: str, resolved: torch.device) -> str:
    if resolved.type == "cuda" and torch.cuda.is_available():
        device_index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        return f"Compute device: GPU (cuda:{device_index}) - {gpu_name}"

    if requested == "cuda" and resolved.type != "cuda":
        return "Compute device: CPU (CUDA requested but not available)"

    return "Compute device: CPU"


def _model_config_payload(
    *,
    config: SystemTransformerTrainConfig,
    vocab_size: int,
    num_classes: int,
) -> dict[str, Any]:
    return {
        "model_type": "TransformerLogClassifier",
        "vocab_size": vocab_size,
        "num_classes": num_classes,
        "d_model": config.d_model,
        "nhead": config.nhead,
        "num_layers": config.num_layers,
        "dim_feedforward": config.dim_feedforward,
        "dropout": config.dropout,
        "max_len": config.max_len,
        "padding_idx": 0,
    }


if __name__ == "__main__":
    main()
