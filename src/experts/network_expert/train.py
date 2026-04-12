from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler

from src.experts.network_expert.model import CNNLSTMClassifier
from src.training.checkpointing import (
    resolve_checkpoint_paths,
    save_metrics_report,
    save_model_checkpoint,
)
from src.training.data import SequenceDataset
from src.training.metrics import compute_classification_report
from src.training.runner import build_dataloader, evaluate_model, set_global_seed, train_one_epoch

DEFAULT_PROCESSED_PATH = Path("data/processed/cicids_processed.pt")
FALLBACK_PROCESSED_PATHS: tuple[Path, ...] = (
    Path("data/processed/cicids_processed_onefile.pt"),
    Path("D:/cloud-anomaly-artifacts/cicids_processed.pt"),
    Path("D:/cloud-anomaly-artifacts/cicids_processed_onefile.pt"),
)


@dataclass(slots=True)
class NetworkBundleTrainConfig:
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
    expected_feature_dim: int
    conv_channels: int
    conv_kernel_size: int
    flow_embedding_dim: int
    lstm_hidden_dim: int
    lstm_layers: int
    dropout: float
    bidirectional: bool
    disable_class_weights: bool
    use_balanced_sampler: bool


def parse_args() -> NetworkBundleTrainConfig:
    parser = argparse.ArgumentParser(
        description="Train CNN-LSTM network expert from cicids_processed.pt bundle."
    )
    parser.add_argument("--processed-data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normal-class-index", type=int, default=0)
    parser.add_argument("--expected-feature-dim", type=int, default=80)
    parser.add_argument("--conv-channels", type=int, default=96)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument("--flow-embedding-dim", type=int, default=128)
    parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a bidirectional LSTM over the flow sequence.",
    )
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weighting in CrossEntropyLoss.",
    )
    parser.add_argument(
        "--balanced-sampler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use WeightedRandomSampler to improve minority-class learning.",
    )
    ns = parser.parse_args()
    return NetworkBundleTrainConfig(
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
        expected_feature_dim=ns.expected_feature_dim,
        conv_channels=ns.conv_channels,
        conv_kernel_size=ns.conv_kernel_size,
        flow_embedding_dim=ns.flow_embedding_dim,
        lstm_hidden_dim=ns.lstm_hidden_dim,
        lstm_layers=ns.lstm_layers,
        dropout=ns.dropout,
        bidirectional=ns.bidirectional,
        disable_class_weights=ns.disable_class_weights,
        use_balanced_sampler=ns.balanced_sampler,
    )


def main() -> None:
    config = parse_args()
    summary = run_training(config)
    print(json.dumps(summary, indent=2))


def run_training(config: NetworkBundleTrainConfig) -> dict[str, Any]:
    processed_data_path = _resolve_processed_data_path(config.processed_data)
    if processed_data_path != config.processed_data:
        print(f"Using fallback processed dataset: {processed_data_path}")

    if not processed_data_path.exists():
        searched = [str(config.processed_data), *(str(path) for path in FALLBACK_PROCESSED_PATHS)]
        raise FileNotFoundError(
            "Processed CICIDS dataset not found. Checked: "
            + ", ".join(searched)
            + ". Run prepare_cicids first."
        )

    set_global_seed(config.seed)
    device = _resolve_device(config.device)
    print(_format_device_info(requested=config.device, resolved=device))
    bundle = torch.load(processed_data_path, map_location="cpu")

    train_X, train_y = _load_split(bundle, "train")
    val_X, val_y = _load_split(bundle, "val")
    test_X, test_y = _load_split(bundle, "test")
    class_names = _resolve_class_names(bundle, train_y)

    if train_X.ndim != 3:
        raise ValueError(
            f"Expected network features [N, seq_len, feature_dim], got {tuple(train_X.shape)}"
        )
    if config.expected_feature_dim > 0 and int(train_X.shape[-1]) != config.expected_feature_dim:
        raise ValueError(
            "Unexpected CICIDS feature dimension. "
            f"Expected {config.expected_feature_dim}, got {int(train_X.shape[-1])}. "
            "Re-run prepare_cicids with --target-feature-count matching this training config."
        )

    input_dim = int(train_X.shape[-1])
    model = CNNLSTMClassifier(
        input_dim=input_dim,
        num_classes=len(class_names),
        conv_channels=config.conv_channels,
        conv_kernel_size=config.conv_kernel_size,
        flow_embedding_dim=config.flow_embedding_dim,
        lstm_hidden_dim=config.lstm_hidden_dim,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    train_dataset = SequenceDataset(train_X, train_y)
    sampler = None
    class_weights_cpu = _inverse_frequency_weights(train_y, num_classes=len(class_names))
    if config.use_balanced_sampler:
        sample_weights = class_weights_cpu[train_y].to(dtype=torch.float64)
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(config.seed)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=int(sample_weights.shape[0]),
            replacement=True,
            generator=sampler_generator,
        )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
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
        f"features={int(train_X.shape[2])} "
        f"classes={len(class_names)}"
    )
    class_counts = _class_counts(train_y, num_classes=len(class_names))
    print("Train class distribution:")
    for idx, class_name in enumerate(class_names):
        print(f"  {class_name}: {class_counts[idx]}")
    class_weights = None
    if not config.disable_class_weights:
        class_weights = class_weights_cpu.to(device)
        weights_text = ", ".join(
            f"{class_names[idx]}={float(class_weights[idx]):.4f}" for idx in range(len(class_names))
        )
        print(f"Using class weights: {weights_text}")
    else:
        print("Using class weights: disabled")
    print(f"Using balanced sampler: {'enabled' if config.use_balanced_sampler else 'disabled'}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    paths = resolve_checkpoint_paths(config.output_dir, prefix="network_expert")

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
            input_dtype="float",
            progress_desc="train",
        )
        val_loss, val_logits, val_labels = evaluate_model(
            model,
            val_loader,
            criterion,
            device=device,
            input_dtype="float",
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
            config=_model_config_payload(config=config, input_dim=input_dim, num_classes=len(class_names)),
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
                    input_dim=input_dim,
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
        input_dtype="float",
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
        "task": "network_expert_training",
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
            "processed_data": str(processed_data_path),
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

    return torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long)


def _resolve_class_names(bundle: dict[str, Any], labels: torch.Tensor) -> list[str]:
    raw = bundle.get("class_names")
    if isinstance(raw, (list, tuple)) and raw:
        return [str(item) for item in raw]
    num_classes = int(labels.max().item()) + 1
    return [f"class_{idx}" for idx in range(num_classes)]


def _resolve_processed_data_path(requested: Path) -> Path:
    if requested.exists():
        return requested

    if requested != DEFAULT_PROCESSED_PATH:
        return requested

    for candidate in FALLBACK_PROCESSED_PATHS:
        if candidate.exists():
            return candidate
    return requested


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
    config: NetworkBundleTrainConfig,
    input_dim: int,
    num_classes: int,
) -> dict[str, Any]:
    return {
        "model_type": "CNNLSTMClassifier",
        "input_dim": input_dim,
        "num_classes": num_classes,
        "conv_channels": config.conv_channels,
        "conv_kernel_size": config.conv_kernel_size,
        "flow_embedding_dim": config.flow_embedding_dim,
        "lstm_hidden_dim": config.lstm_hidden_dim,
        "lstm_layers": config.lstm_layers,
        "dropout": config.dropout,
        "bidirectional": config.bidirectional,
    }


def _class_counts(labels: torch.Tensor, num_classes: int) -> list[int]:
    counts = torch.bincount(labels, minlength=num_classes)
    return [int(value) for value in counts.tolist()]


def _inverse_frequency_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    total = torch.sum(counts)
    return total / (counts * float(num_classes))


if __name__ == "__main__":
    main()
