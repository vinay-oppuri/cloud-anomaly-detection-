from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.correlation.compatibility import (
    build_common_embedding,
    compatibility_score,
    merge_embeddings,
    severity_score,
)
from src.correlation.schema import AnomalyEvent
from src.experts.network_expert.model import CNNTransformerClassifier
from src.experts.system_expert.classifier import classify_anomaly
from src.experts.system_expert.model import TransformerLogClassifier


NETWORK_PREPROCESSED_DIR = Path("data/processed")
NETWORK_MODEL_PATH = Path("models/network_expert.pth")
NETWORK_META_PATH = Path("models/network_meta.json")
SYSTEM_PROCESSED_PATH = Path("data/processed/hdfs_processed.pt")
SYSTEM_CACHE_PATH = Path("data/processed/hdfs_cache.json")
SYSTEM_MODEL_PATH = Path("models/system_expert_best.pth")
SYSTEM_METRICS_PATH = Path("models/system_expert_metrics.json")

NETWORK_AGGREGATION_GAP = timedelta(seconds=45)
SYSTEM_DELAY_BY_NETWORK_LABEL = {
    "DDoS": 25,
    "DoS": 40,
    "BruteForce": 180,
    "WebAttack": 240,
    "Botnet": 300,
    "OtherAttack": 210,
    "Benign": 0,
}


@dataclass(slots=True)
class EventExtractionConfig:
    split: str = "test"
    device: str = "cuda"
    network_preprocessed_dir: Path = NETWORK_PREPROCESSED_DIR
    network_model_path: Path = NETWORK_MODEL_PATH
    network_meta_path: Path = NETWORK_META_PATH
    system_processed_path: Path = SYSTEM_PROCESSED_PATH
    system_cache_path: Path = SYSTEM_CACHE_PATH
    system_model_path: Path = SYSTEM_MODEL_PATH
    system_metrics_path: Path = SYSTEM_METRICS_PATH
    batch_size: int = 1024
    max_network_events: int | None = None
    max_system_events: int | None = None
    show_progress: bool = True


class _TensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, indices: np.ndarray) -> None:
        self.features = features
        self.labels = labels
        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        source_index = int(self.indices[index])
        return self.features[source_index], self.labels[source_index], source_index


def extract_cross_layer_events(config: EventExtractionConfig) -> tuple[list[AnomalyEvent], list[AnomalyEvent]]:
    network_events = extract_network_events(config)
    system_events = extract_system_events(config)
    aligned_system_events = synthesize_system_timestamps(system_events, network_events)
    return network_events, aligned_system_events


def extract_network_events(config: EventExtractionConfig) -> list[AnomalyEvent]:
    device = _resolve_device(config.device)
    checkpoint = torch.load(config.network_model_path, map_location="cpu", weights_only=False)
    class_info = json.loads((config.network_preprocessed_dir / "class_info.json").read_text(encoding="utf-8"))
    metrics = json.loads(config.network_meta_path.read_text(encoding="utf-8"))
    class_names = [str(item) for item in checkpoint.get("class_names", class_info["class_names"])]
    threshold = float(metrics.get("best_threshold", metrics.get("threshold", 0.5)))
    split = str(config.split)

    features = torch.load(config.network_preprocessed_dir / f"{split}_X.pt", map_location="cpu", weights_only=False)
    labels = torch.load(config.network_preprocessed_dir / f"{split}_y.pt", map_location="cpu", weights_only=False)
    anomaly_labels_path = config.network_preprocessed_dir / f"{split}_y_binary.pt"
    if anomaly_labels_path.exists():
        anomaly_targets = torch.load(anomaly_labels_path, map_location="cpu", weights_only=False)
        candidate_indices = torch.nonzero(torch.as_tensor(anomaly_targets).long() == 1, as_tuple=False).squeeze(-1).numpy()
    else:
        benign_index = class_names.index("Benign") if "Benign" in class_names else 0
        candidate_indices = torch.nonzero(torch.as_tensor(labels).long() != benign_index, as_tuple=False).squeeze(-1).numpy()
    if candidate_indices.size == 0:
        return []
    if config.max_network_events is not None:
        candidate_indices = _limit_candidate_indices(candidate_indices, target_count=max_network_events_to_candidates(config.max_network_events))

    meta = torch.load(config.network_preprocessed_dir / f"{split}_meta.pt", map_location="cpu", weights_only=False)
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    label_tensor = torch.as_tensor(labels, dtype=torch.long)

    model_cfg = checkpoint.get("config", {})
    model = CNNTransformerClassifier(
        input_dim=int(model_cfg.get("input_dim", feature_tensor.shape[-1])),
        num_classes=int(model_cfg.get("num_classes", len(class_names))),
        conv_channels=int(model_cfg.get("conv_channels", 128)),
        conv_kernel_size=int(model_cfg.get("conv_kernel_size", 3)),
        flow_embedding_dim=int(model_cfg.get("flow_embedding_dim", 128)),
        transformer_heads=int(model_cfg.get("transformer_heads", 4)),
        transformer_layers=int(model_cfg.get("transformer_layers", 3)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 256)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        _TensorDataset(feature_tensor, label_tensor, candidate_indices),
        batch_size=max(32, config.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    extracted: list[dict[str, Any]] = []
    iterator: Iterable[Any] = loader
    if config.show_progress:
        iterator = tqdm(loader, desc=f"Extracting network events [{split}]", unit="batch", dynamic_ncols=True, leave=False)

    benign_index = class_names.index("Benign") if "Benign" in class_names else 0

    with torch.inference_mode():
        for batch_X, batch_y, batch_indices in iterator:
            batch_X = batch_X.to(device, non_blocking=True)
            logits = model(batch_X)
            embeddings = model.forward_features(batch_X)
            probabilities = torch.softmax(logits, dim=-1)
            anomaly_scores = 1.0 - probabilities[:, benign_index]
            predicted_indices = torch.argmax(probabilities, dim=-1)

            for row_idx, score, pred_idx, embedding, probs in zip(
                batch_indices.tolist(),
                anomaly_scores.detach().cpu().tolist(),
                predicted_indices.detach().cpu().tolist(),
                embeddings.detach().cpu().numpy(),
                probabilities.detach().cpu().numpy(),
                strict=False,
            ):
                if float(score) < threshold:
                    continue
                meta_row = meta[int(row_idx)]
                timestamp = _select_network_timestamp(meta_row)
                label = class_names[int(pred_idx)]
                confidence = float(probs[int(pred_idx)])
                extracted.append(
                    {
                        "row_index": int(row_idx),
                        "timestamp": timestamp,
                        "label": label,
                        "anomaly_score": float(score),
                        "confidence": confidence,
                        "embedding": build_common_embedding(
                            raw_embedding=np.asarray(embedding, dtype=np.float32),
                            label=label,
                            event_type="network",
                            anomaly_score=float(score),
                            severity=_network_severity(label, float(score)),
                        ),
                        "severity": _network_severity(label, float(score)),
                        "source_ref": f"{meta_row.get('source_file', 'unknown')}:{meta_row.get('start_index', 0)}-{meta_row.get('end_index', 0)}",
                        "metadata": {
                            "source_file": meta_row.get("source_file"),
                            "start_index": meta_row.get("start_index"),
                            "end_index": meta_row.get("end_index"),
                            "start_timestamp": meta_row.get("start_timestamp"),
                            "end_timestamp": meta_row.get("end_timestamp"),
                            "ground_truth_label": class_names[int(label_tensor[int(row_idx)].item())],
                            "window_label": meta_row.get("window_label"),
                        },
                    }
                )

    aggregated = aggregate_network_events(extracted)
    if config.max_network_events is not None:
        aggregated = _select_evenly_spaced_events(aggregated, int(config.max_network_events))
    else:
        aggregated.sort(key=lambda event: event.timestamp)
    return aggregated


def aggregate_network_events(rows: list[dict[str, Any]]) -> list[AnomalyEvent]:
    if not rows:
        return []
    rows.sort(key=lambda item: (item["metadata"].get("source_file"), item["timestamp"], item["label"]))
    clusters: list[list[dict[str, Any]]] = []

    current_cluster: list[dict[str, Any]] = [rows[0]]
    for item in rows[1:]:
        prev = current_cluster[-1]
        same_label = item["label"] == prev["label"]
        same_file = item["metadata"].get("source_file") == prev["metadata"].get("source_file")
        close_in_time = (item["timestamp"] - prev["timestamp"]) <= NETWORK_AGGREGATION_GAP
        if same_label and same_file and close_in_time:
            current_cluster.append(item)
        else:
            clusters.append(current_cluster)
            current_cluster = [item]
    clusters.append(current_cluster)

    events: list[AnomalyEvent] = []
    for cluster_idx, cluster_rows in enumerate(clusters, start=1):
        anchor = max(cluster_rows, key=lambda item: float(item["anomaly_score"]))
        timestamps = [item["timestamp"] for item in cluster_rows]
        event_id = f"network-{cluster_idx:05d}"
        metadata = dict(anchor["metadata"])
        metadata["window_count"] = len(cluster_rows)
        metadata["cluster_start"] = min(timestamps).isoformat()
        metadata["cluster_end"] = max(timestamps).isoformat()
        metadata["member_rows"] = [item["row_index"] for item in cluster_rows]
        events.append(
            AnomalyEvent(
                event_id=event_id,
                timestamp=min(timestamps),
                event_type="network",
                label=str(anchor["label"]),
                anomaly_score=float(max(item["anomaly_score"] for item in cluster_rows)),
                confidence=float(max(item["confidence"] for item in cluster_rows)),
                embedding=merge_embeddings([item["embedding"] for item in cluster_rows]),
                source_ref=str(anchor["source_ref"]),
                severity=str(anchor["severity"]),
                metadata=metadata,
            )
        )
    return events


def extract_system_events(config: EventExtractionConfig) -> list[AnomalyEvent]:
    device = _resolve_device(config.device)
    bundle = torch.load(config.system_processed_path, map_location="cpu", weights_only=False)
    cache = json.loads(config.system_cache_path.read_text(encoding="utf-8"))
    metrics = json.loads(config.system_metrics_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(config.system_model_path, map_location="cpu", weights_only=False)
    split = str(config.split)

    split_payload = bundle["splits"][split]
    features = torch.as_tensor(split_payload["X"], dtype=torch.long)
    labels = torch.as_tensor(split_payload["y"], dtype=torch.long)
    block_ids = [str(item) for item in split_payload["block_ids"]]
    lengths = torch.as_tensor(split_payload.get("lengths"), dtype=torch.long)
    threshold = float(metrics.get("validation", {}).get("best_threshold", 0.5))
    class_names = [str(item) for item in checkpoint.get("class_names", bundle.get("class_names", ["Normal", "Anomaly"]))]

    candidate_indices = torch.nonzero(labels == 1, as_tuple=False).squeeze(-1).numpy()
    if candidate_indices.size == 0:
        return []
    if config.max_system_events is not None:
        candidate_indices = _limit_candidate_indices(candidate_indices, target_count=max_system_events_to_candidates(config.max_system_events))

    vocab = cache["vocab"]
    inverse_vocab = {int(value): str(key).upper() for key, value in vocab.items()}

    model_cfg = checkpoint.get("config", {})
    model = TransformerLogClassifier(
        vocab_size=int(model_cfg.get("vocab_size", bundle.get("vocab_size", 2))),
        num_classes=int(model_cfg.get("num_classes", len(class_names))),
        d_model=int(model_cfg.get("d_model", 160)),
        nhead=int(model_cfg.get("nhead", 8)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 384)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        max_len=int(model_cfg.get("max_len", bundle.get("sequence_length", 128))),
        padding_idx=int(model_cfg.get("padding_idx", 0)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        _TensorDataset(features, labels, candidate_indices),
        batch_size=max(32, config.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    extracted: list[AnomalyEvent] = []
    iterator: Iterable[Any] = loader
    if config.show_progress:
        iterator = tqdm(loader, desc=f"Extracting system events [{split}]", unit="batch", dynamic_ncols=True, leave=False)

    normal_index = class_names.index("Normal") if "Normal" in class_names else 0
    with torch.inference_mode():
        for batch_X, _, batch_indices in iterator:
            batch_X = batch_X.to(device, non_blocking=True)
            logits = model(batch_X)
            embeddings = model.forward_features(batch_X)
            probabilities = torch.softmax(logits, dim=-1)
            anomaly_scores = 1.0 - probabilities[:, normal_index]
            predicted_indices = torch.argmax(probabilities, dim=-1)

            for row_idx, score, pred_idx, embedding, probs in zip(
                batch_indices.tolist(),
                anomaly_scores.detach().cpu().tolist(),
                predicted_indices.detach().cpu().tolist(),
                embeddings.detach().cpu().numpy(),
                probabilities.detach().cpu().numpy(),
                strict=False,
            ):
                if float(score) < threshold:
                    continue
                token_ids = features[int(row_idx)].tolist()
                event_names = [
                    inverse_vocab.get(int(token_id), "<UNK>")
                    for token_id in token_ids
                    if int(token_id) != 0
                ]
                classified = classify_anomaly(event_names, float(score))
                label = str(classified["anomaly_type"])
                confidence = float(probs[int(pred_idx)])
                extracted.append(
                    AnomalyEvent(
                        event_id=f"log-{len(extracted)+1:05d}",
                        timestamp=datetime(2018, 1, 1),
                        event_type="log",
                        label=label,
                        anomaly_score=float(score),
                        confidence=confidence,
                        embedding=build_common_embedding(
                            raw_embedding=np.asarray(embedding, dtype=np.float32),
                            label=label,
                            event_type="log",
                            anomaly_score=float(score),
                            severity=str(classified["severity"]),
                        ),
                        source_ref=str(block_ids[int(row_idx)]),
                        severity=str(classified["severity"]),
                        metadata={
                            "block_id": block_ids[int(row_idx)],
                            "event_names": event_names,
                            "length": int(lengths[int(row_idx)].item()) if lengths.numel() > 0 else len(event_names),
                            "predicted_class": class_names[int(pred_idx)],
                            "matched_rules": list(classified["matched_rules"]),
                            "description": str(classified["description"]),
                        },
                    )
                )

    extracted.sort(key=lambda event: (-event.anomaly_score, event.source_ref))
    if config.max_system_events is not None:
        extracted = extracted[: int(config.max_system_events)]
    extracted.sort(key=lambda event: (event.label, event.source_ref))
    return extracted


def synthesize_system_timestamps(
    system_events: list[AnomalyEvent],
    network_events: list[AnomalyEvent],
) -> list[AnomalyEvent]:
    if not system_events or not network_events:
        return system_events

    sorted_network = sorted(network_events, key=lambda event: event.timestamp)
    grouped_assignments: dict[str, list[tuple[AnomalyEvent, float, float]]] = defaultdict(list)
    system_groups: dict[str, list[AnomalyEvent]] = defaultdict(list)
    for system_event in system_events:
        system_groups[system_event.label].append(system_event)

    for label, label_events in system_groups.items():
        label_events.sort(key=lambda event: (event.source_ref, -event.anomaly_score))
        compatible_candidates = [
            (network_event, compatibility_score(network_event.label, label))
            for network_event in sorted_network
            if compatibility_score(network_event.label, label) >= 0.18
        ]
        if not compatible_candidates:
            compatible_candidates = [
                (network_event, compatibility_score(network_event.label, label))
                for network_event in sorted_network
            ]

        target_load = max(1, math.ceil(len(label_events) / max(1, len(compatible_candidates))))
        search_radius = min(18, max(3, len(compatible_candidates) // 25))
        parent_loads: dict[str, int] = defaultdict(int)
        max_label_rank = max(1, len(label_events) - 1)

        for label_rank, system_event in enumerate(label_events):
            anchor_index = int(round((label_rank / max_label_rank) * max(0, len(compatible_candidates) - 1)))
            left = max(0, anchor_index - search_radius)
            right = min(len(compatible_candidates), anchor_index + search_radius + 1)
            best_network: AnomalyEvent | None = None
            best_compatibility = 0.0
            best_score = -1.0

            for candidate_index in range(left, right):
                network_event, compat = compatible_candidates[candidate_index]
                distance = abs(candidate_index - anchor_index)
                local_rank_score = 1.0 - (distance / max(1, search_radius))
                severity_alignment = 1.0 - abs(severity_score(network_event.severity) - severity_score(system_event.severity))
                score_alignment = 1.0 - abs(network_event.anomaly_score - system_event.anomaly_score)
                load_penalty = min(1.0, parent_loads[network_event.event_id] / float(target_load))
                objective = (
                    0.42 * compat
                    + 0.23 * local_rank_score
                    + 0.15 * severity_alignment
                    + 0.12 * score_alignment
                    + 0.08 * (1.0 - load_penalty)
                )
                if objective > best_score:
                    best_score = float(objective)
                    best_network = network_event
                    best_compatibility = float(compat)

            if best_network is None:
                continue

            parent_loads[best_network.event_id] += 1
            grouped_assignments[best_network.event_id].append((system_event, best_compatibility, best_score))
            system_event.synthetic_parent_id = best_network.event_id
            system_event.synthetic_chain_id = f"chain-{best_network.event_id}"

    aligned_events: list[AnomalyEvent] = []
    for network_event in sorted_network:
        assignments = grouped_assignments.get(network_event.event_id, [])
        assignments.sort(key=lambda item: (item[0].source_ref, -item[2]))
        if assignments:
            network_event.synthetic_chain_id = f"chain-{network_event.event_id}"
            network_event.metadata["synthetic_children"] = len(assignments)
        for position, (system_event, compat, objective) in enumerate(assignments):
            base_delay = SYSTEM_DELAY_BY_NETWORK_LABEL.get(network_event.label, 120)
            length = int(system_event.metadata.get("length", 0))
            delay_seconds = base_delay + position * 6 + min(90, int(length * 0.35))
            delay_seconds += int((1.0 - compat) * 45.0)
            system_event.timestamp = network_event.timestamp + timedelta(seconds=delay_seconds)
            system_event.metadata["synthetic_alignment"] = {
                "network_parent_id": network_event.event_id,
                "network_parent_label": network_event.label,
                "compatibility_score": round(float(compat), 6),
                "assignment_score": round(float(objective), 6),
                "delay_seconds": int(delay_seconds),
            }
            aligned_events.append(system_event)

    aligned_events.sort(key=lambda event: event.timestamp)
    return aligned_events


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested if requested != "cuda" else "cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_iso_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _select_network_timestamp(meta_row: dict[str, Any]) -> datetime:
    candidates = [
        _parse_iso_timestamp(meta_row.get("start_timestamp")),
        _parse_iso_timestamp(meta_row.get("end_timestamp")),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.year >= 2018:
            return candidate
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return datetime(2018, 1, 1)


def _network_severity(label: str, anomaly_score: float) -> str:
    if label == "DDoS":
        return "Critical"
    if label in {"DoS", "Botnet", "WebAttack"}:
        return "High"
    if label == "BruteForce":
        return "High" if anomaly_score >= 0.85 else "Medium"
    if label == "OtherAttack":
        return "Medium"
    return "Low"


def _limit_candidate_indices(candidate_indices: np.ndarray, *, target_count: int) -> np.ndarray:
    if candidate_indices.size <= target_count:
        return candidate_indices
    positions = np.linspace(0, candidate_indices.size - 1, num=target_count, dtype=np.int64)
    return candidate_indices[positions]


def max_network_events_to_candidates(max_network_events: int) -> int:
    return max(500, int(max_network_events) * 40)


def max_system_events_to_candidates(max_system_events: int) -> int:
    return max(500, int(max_system_events) * 12)


def _select_evenly_spaced_events(events: list[AnomalyEvent], target_count: int) -> list[AnomalyEvent]:
    ordered = sorted(events, key=lambda event: event.timestamp)
    if len(ordered) <= target_count:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, num=target_count, dtype=np.int64)
    return [ordered[int(position)] for position in positions]
