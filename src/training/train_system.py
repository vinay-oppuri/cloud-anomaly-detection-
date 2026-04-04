from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

from src.experts.system_model import BiLSTMLogClassifier
from src.training.checkpointing import (
    resolve_checkpoint_paths,
    save_metrics_report,
    save_model_checkpoint,
)
from src.training.data import SequenceDataset, load_processed_dataset
from src.training.metrics import compute_classification_report
from src.training.runner import build_dataloader, evaluate_model, set_global_seed, train_one_epoch


@dataclass(slots=True)
class SystemTrainConfig:
    train_data: str
    val_data: str
    test_data: str | None
    output_dir: str
    class_names_path: str | None
    num_classes: int | None
    vocab_size: int | None
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    seed: int
    device: str
    normal_class_index: int
    embedding_dim: int
    hidden_dim: int
    lstm_layers: int
    dropout: float


def parse_args() -> SystemTrainConfig:
    parser = argparse.ArgumentParser(description="Train Bi-LSTM system-log anomaly expert.")
    parser.add_argument("--train-data", type=str, required=True, help="Path to train .npz/.pt")
    parser.add_argument("--val-data", type=str, required=True, help="Path to val .npz/.pt")
    parser.add_argument("--test-data", type=str, default=None, help="Path to test .npz/.pt")
    parser.add_argument("--output-dir", type=str, default="models", help="Artifact output directory")
    parser.add_argument("--class-names-path", type=str, default=None, help="Text file with one class per line")
    parser.add_argument("--num-classes", type=int, default=None, help="Override number of classes")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocab size")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normal-class-index", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    ns = parser.parse_args()
    return SystemTrainConfig(
        train_data=ns.train_data,
        val_data=ns.val_data,
        test_data=ns.test_data,
        output_dir=ns.output_dir,
        class_names_path=ns.class_names_path,
        num_classes=ns.num_classes,
        vocab_size=ns.vocab_size,
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        learning_rate=ns.learning_rate,
        weight_decay=ns.weight_decay,
        patience=ns.patience,
        seed=ns.seed,
        device=ns.device,
        normal_class_index=ns.normal_class_index,
        embedding_dim=ns.embedding_dim,
        hidden_dim=ns.hidden_dim,
        lstm_layers=ns.lstm_layers,
        dropout=ns.dropout,
    )


def main() -> None:
    config = parse_args()
    set_global_seed(config.seed)

    device = resolve_device(config.device)
    train_loaded = load_processed_dataset(config.train_data, feature_dtype=torch.long)
    val_loaded = load_processed_dataset(config.val_data, feature_dtype=torch.long)
    test_loaded = (
        load_processed_dataset(config.test_data, feature_dtype=torch.long)
        if config.test_data is not None
        else None
    )

    if train_loaded.features.ndim != 2:
        raise ValueError(
            f"System data must be [N, seq_len] token IDs, got {tuple(train_loaded.features.shape)}"
        )

    inferred_classes = int(train_loaded.labels.max().item()) + 1
    num_classes = config.num_classes or inferred_classes
    class_names = resolve_class_names(config.class_names_path, num_classes=num_classes)
    if len(class_names) != num_classes:
        raise ValueError("class_names length does not match num_classes.")

    inferred_vocab_size = int(train_loaded.features.max().item()) + 1
    vocab_size = config.vocab_size or max(inferred_vocab_size, 2)

    model = BiLSTMLogClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    criterion = nn.CrossEntropyLoss()

    train_dataset = SequenceDataset(train_loaded.features, train_loaded.labels)
    val_dataset = SequenceDataset(val_loaded.features, val_loaded.labels)
    test_dataset = (
        SequenceDataset(test_loaded.features, test_loaded.labels) if test_loaded is not None else None
    )

    train_loader = build_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = (
        build_dataloader(test_dataset, batch_size=config.batch_size, shuffle=False)
        if test_dataset is not None
        else None
    )

    paths = resolve_checkpoint_paths(config.output_dir, prefix="system_expert")

    best_macro_f1 = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            input_dtype="long",
        )
        val_loss, val_logits, val_labels = evaluate_model(
            model,
            val_loader,
            criterion,
            device=device,
            input_dtype="long",
        )
        val_report = compute_classification_report(
            loss=val_loss,
            labels=val_labels,
            logits=val_logits,
            class_names=class_names,
            normal_class_index=config.normal_class_index,
        )
        scheduler.step(val_loss)

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
            config=_model_config_payload(
                config=config,
                vocab_size=vocab_size,
                num_classes=num_classes,
            ),
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
                    vocab_size=vocab_size,
                    num_classes=num_classes,
                ),
                epoch=epoch,
                metrics=val_report.to_dict(),
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    test_metrics: dict[str, Any] | None = None
    if test_loader is not None:
        checkpoint = torch.load(paths.best_model_path, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
        test_loss, test_logits, test_labels = evaluate_model(
            model,
            test_loader,
            criterion,
            device=device,
            input_dtype="long",
        )
        test_report = compute_classification_report(
            loss=test_loss,
            labels=test_labels,
            logits=test_logits,
            class_names=class_names,
            normal_class_index=config.normal_class_index,
        )
        test_metrics = test_report.to_dict()

    summary = {
        "task": "system_expert_training",
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_model_path": str(paths.best_model_path),
        "last_model_path": str(paths.last_model_path),
        "metrics": {
            "best_val": best_metrics,
            "test": test_metrics,
            "history": history,
        },
        "config": asdict(config),
    }
    save_metrics_report(paths.metrics_path, summary)
    print(json.dumps(summary, indent=2))


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def resolve_class_names(path: str | None, num_classes: int) -> list[str]:
    if path is None:
        return [f"class_{idx}" for idx in range(num_classes)]

    file_path = Path(path)
    class_names = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not class_names:
        raise ValueError(f"No class names found in {file_path}")
    return class_names


def _model_config_payload(
    *,
    config: SystemTrainConfig,
    vocab_size: int,
    num_classes: int,
) -> dict[str, Any]:
    return {
        "model_type": "BiLSTMLogClassifier",
        "vocab_size": vocab_size,
        "num_classes": num_classes,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "lstm_layers": config.lstm_layers,
        "dropout": config.dropout,
    }


if __name__ == "__main__":
    main()

