from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.correlation.compatibility import compatibility_score
from src.correlation.events import EventExtractionConfig, extract_cross_layer_events
from src.correlation.model import AttentionCorrelationModel, PairSample
from src.correlation.pair_features import PAIR_FEATURE_DIM, build_pair_features
from src.correlation.schema import AnomalyEvent


DEFAULT_MODEL_PATH = Path("models/correlation_attention.pth")
DEFAULT_METRICS_PATH = Path("models/correlation_attention_metrics.json")
NEGATIVE_TO_POSITIVE_RATIO = 1.5


CFG = {
    "hidden_dim": 128,
    "dropout": 0.1,
    "batch_size": 512,
    "epochs": 40,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "patience": 8,
    "seed": 42,
    "temporal_window_minutes": 20,
}


@dataclass(slots=True)
class TrainConfig:
    train_split: str
    val_split: str
    device: str
    model_path: Path
    metrics_path: Path
    batch_size: int
    max_network_events: int | None
    max_system_events: int | None
    show_progress: bool


class PairDataset(Dataset):
    def __init__(self, samples: list[PairSample], event_lookup: dict[str, AnomalyEvent]) -> None:
        self.samples = samples
        self.event_lookup = event_lookup

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        left = torch.from_numpy(self.event_lookup[sample.left_id].embedding.astype(np.float32))
        right = torch.from_numpy(self.event_lookup[sample.right_id].embedding.astype(np.float32))
        pair_features = torch.from_numpy(sample.pair_features.astype(np.float32))
        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return left, right, pair_features, label


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the attention-based cross-layer correlation model.")
    parser.add_argument("--train-split", choices=("train", "val"), default="train")
    parser.add_argument("--val-split", choices=("val", "test"), default="val")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-network-events", type=int, default=3000)
    parser.add_argument("--max-system-events", type=int, default=3000)
    parser.add_argument("--hide-progress", action="store_true")
    args = parser.parse_args()
    return TrainConfig(
        train_split=args.train_split,
        val_split=args.val_split,
        device=args.device,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        batch_size=max(64, int(args.batch_size)),
        max_network_events=None if args.max_network_events in (None, 0) else int(args.max_network_events),
        max_system_events=None if args.max_system_events in (None, 0) else int(args.max_system_events),
        show_progress=not bool(args.hide_progress),
    )


def main() -> None:
    config = parse_args()
    train_correlation_model(config)


def train_correlation_model(config: TrainConfig) -> None:
    seed_everything(CFG["seed"])
    device = resolve_device(config.device)
    print("=" * 64)
    print("  Attention-Based Correlation Model Training")
    print("=" * 64)
    print(f"  Device : {device}")

    train_events = load_events_for_split(
        split=config.train_split,
        device=config.device,
        batch_size=config.batch_size,
        max_network_events=config.max_network_events,
        max_system_events=config.max_system_events,
        show_progress=config.show_progress,
    )
    val_events = load_events_for_split(
        split=config.val_split,
        device=config.device,
        batch_size=config.batch_size,
        max_network_events=config.max_network_events,
        max_system_events=config.max_system_events,
        show_progress=config.show_progress,
    )

    train_samples, train_lookup = build_pair_samples(train_events, temporal_window_minutes=CFG["temporal_window_minutes"])
    val_samples, val_lookup = build_pair_samples(val_events, temporal_window_minutes=CFG["temporal_window_minutes"])
    if not train_samples or not val_samples:
        raise ValueError("Could not build correlation pair samples from the extracted events.")

    train_loader = DataLoader(
        PairDataset(train_samples, train_lookup),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        PairDataset(val_samples, val_lookup),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = AttentionCorrelationModel(
        input_dim=104,
        hidden_dim=CFG["hidden_dim"],
        dropout=CFG["dropout"],
        pair_feature_dim=PAIR_FEATURE_DIM,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    started_at = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, CFG["epochs"], config.show_progress)
        val_metrics = evaluate(model, val_loader, criterion, device, config.show_progress, desc=f"Epoch {epoch:02d}/{CFG['epochs']} [val]")
        scheduler.step(val_metrics["f1"])

        print(
            f"  Epoch {epoch:02d}/{CFG['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = float(val_metrics["f1"])
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "threshold": float(val_metrics["best_threshold"]),
                    "config": {
                        "input_dim": 104,
                        "hidden_dim": CFG["hidden_dim"],
                        "dropout": CFG["dropout"],
                        "pair_feature_dim": PAIR_FEATURE_DIM,
                    },
                },
                config.model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= CFG["patience"]:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    elapsed_minutes = (time.time() - started_at) / 60.0
    best_model = torch.load(config.model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_model["state_dict"])
    final_val_metrics = evaluate(model, val_loader, criterion, device, config.show_progress, desc="Final correlation validation")

    metrics_payload = {
        "device": str(device),
        "best_epoch": best_epoch,
        "elapsed_minutes": round(elapsed_minutes, 3),
        "train_pairs": len(train_samples),
        "val_pairs": len(val_samples),
        "best_threshold": float(best_model.get("threshold", final_val_metrics["best_threshold"])),
        "config": CFG,
        "validation": final_val_metrics,
    }
    config.metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"\nBest checkpoint -> {config.model_path}")
    print(f"Metrics         -> {config.metrics_path}")


def load_events_for_split(
    *,
    split: str,
    device: str,
    batch_size: int,
    max_network_events: int | None,
    max_system_events: int | None,
    show_progress: bool,
) -> list[AnomalyEvent]:
    network_events, system_events = extract_cross_layer_events(
        EventExtractionConfig(
            split=split,
            device=device,
            batch_size=batch_size,
            max_network_events=max_network_events,
            max_system_events=max_system_events,
            show_progress=show_progress,
        )
    )
    return sorted([*network_events, *system_events], key=lambda event: event.timestamp)


def build_pair_samples(
    events: list[AnomalyEvent],
    *,
    temporal_window_minutes: int,
) -> tuple[list[PairSample], dict[str, AnomalyEvent]]:
    event_lookup = {event.event_id: event for event in events}
    samples: list[PairSample] = []
    positive_pairs: list[PairSample] = []
    negative_pairs: list[PairSample] = []
    window_seconds = float(temporal_window_minutes * 60)
    positive_pair_ids = build_positive_pair_ids(events)

    for left_index, left_event in enumerate(events):
        for right_index in range(left_index + 1, len(events)):
            right_event = events[right_index]
            delta_seconds = abs((right_event.timestamp - left_event.timestamp).total_seconds())
            if delta_seconds > window_seconds:
                break

            causal_score = float(
                max(
                    compatibility_score(left_event.label, right_event.label),
                    compatibility_score(right_event.label, left_event.label),
                )
            )
            pair_features = build_pair_features(
                left_event=left_event,
                right_event=right_event,
                delta_seconds=delta_seconds,
                temporal_window_seconds=window_seconds,
                causal_score=causal_score,
            )
            cross_modal = left_event.event_type != right_event.event_type
            sample = PairSample(
                left_id=left_event.event_id,
                right_id=right_event.event_id,
                pair_features=pair_features,
                label=1 if (left_event.event_id, right_event.event_id) in positive_pair_ids else 0,
            )
            if sample.label == 1:
                positive_pairs.append(sample)
            elif cross_modal and causal_score >= 0.18:
                negative_pairs.append(sample)
            elif (
                left_event.event_type == right_event.event_type
                and left_event.label == right_event.label
                and delta_seconds <= min(window_seconds, 300.0)
            ):
                negative_pairs.append(sample)

    random.Random(CFG["seed"]).shuffle(negative_pairs)
    negative_pairs = negative_pairs[: max(1, int(len(positive_pairs) * NEGATIVE_TO_POSITIVE_RATIO))]
    samples.extend(positive_pairs)
    samples.extend(negative_pairs)
    random.Random(CFG["seed"]).shuffle(samples)
    return samples, event_lookup


def build_positive_pair_ids(events: list[AnomalyEvent]) -> set[tuple[str, str]]:
    positive_pairs: set[tuple[str, str]] = set()

    # Direct cross-layer parent-child links are the primary causal relation.
    for event in events:
        if event.event_type != "log" or event.synthetic_parent_id is None:
            continue
        positive_pairs.add((event.synthetic_parent_id, event.event_id))

    # Mirror the graph's contextual edges: adjacent logs within the same chain.
    chain_to_logs: dict[str, list[AnomalyEvent]] = {}
    for event in events:
        if event.event_type != "log" or event.synthetic_chain_id is None:
            continue
        chain_to_logs.setdefault(event.synthetic_chain_id, []).append(event)
    for log_events in chain_to_logs.values():
        ordered = sorted(log_events, key=lambda event: event.timestamp)
        for left_event, right_event in zip(ordered, ordered[1:], strict=False):
            positive_pairs.add((left_event.event_id, right_event.event_id))

    # Mirror same-stream network context edges used during graph construction.
    stream_to_networks: dict[tuple[str | None, str], list[AnomalyEvent]] = {}
    for event in events:
        if event.event_type != "network":
            continue
        key = (event.metadata.get("source_file"), event.label)
        stream_to_networks.setdefault(key, []).append(event)
    for network_events in stream_to_networks.values():
        ordered = sorted(network_events, key=lambda event: event.timestamp)
        for left_event, right_event in zip(ordered, ordered[1:], strict=False):
            positive_pairs.add((left_event.event_id, right_event.event_id))

    return positive_pairs


def train_epoch(
    model: AttentionCorrelationModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    show_progress: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs} [train]", unit="batch", dynamic_ncols=True, leave=False)
    for left, right, pair_features, labels in iterator:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        pair_features = pair_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model.score_pairs(left, right, pair_features)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(1, total_items)


@torch.inference_mode()
def evaluate(
    model: AttentionCorrelationModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool,
    *,
    desc: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=False)

    for left, right, pair_features, labels in iterator:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        pair_features = pair_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.score_pairs(left, right, pair_features)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    probabilities = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    best_threshold = compute_best_threshold(labels, probabilities)
    predictions = (probabilities >= best_threshold).astype(np.int64)
    return {
        "loss": total_loss / max(1, total_items),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "best_threshold": float(best_threshold),
    }


def compute_best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    denominator = precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_values = np.where(
            denominator > 0,
            2.0 * precision * recall / denominator,
            0.0,
        )
    best_index = int(np.argmax(f1_values))
    if best_index >= len(thresholds):
        return 0.5
    return float(thresholds[best_index])


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested if requested != "cuda" else "cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
