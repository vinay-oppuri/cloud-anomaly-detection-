from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.experts.network_expert.model import CNNLSTMClassifier
from src.training.data import SequenceDataset
from src.training.metrics import compute_classification_report
from src.training.runner import build_dataloader, evaluate_model

DEFAULT_PROCESSED_PATH = Path("data/processed/cicids_processed.pt")
DEFAULT_MODEL_PATH = Path("models/network_expert_best.pth")

FALLBACK_PROCESSED_PATHS: tuple[Path, ...] = (
    Path("data/processed/cicids_processed_onefile.pt"),
    Path("D:/cloud-anomaly-artifacts/cicids_processed.pt"),
    Path("D:/cloud-anomaly-artifacts/cicids_processed_onefile.pt"),
)
FALLBACK_MODEL_PATHS: tuple[Path, ...] = (
    Path("models/network_expert_best.pth"),
    Path("D:/cloud-anomaly-artifacts/models/network_expert_best.pth"),
)


@dataclass(slots=True)
class NetworkTestConfig:
    processed_data: Path
    model_path: Path
    split: str
    batch_size: int
    device: str
    normal_class_index: int


def parse_args() -> NetworkTestConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained CICIDS network expert checkpoint."
    )
    parser.add_argument("--processed-data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normal-class-index", type=int, default=0)
    ns = parser.parse_args()
    return NetworkTestConfig(
        processed_data=ns.processed_data,
        model_path=ns.model_path,
        split=ns.split,
        batch_size=ns.batch_size,
        device=ns.device,
        normal_class_index=ns.normal_class_index,
    )


def main() -> None:
    config = parse_args()
    summary = run_evaluation(config)
    print(json.dumps(summary, indent=2))


def run_evaluation(config: NetworkTestConfig) -> dict[str, Any]:
    processed_data_path = _resolve_processed_data_path(config.processed_data)
    model_path = _resolve_model_path(config.model_path)

    if not processed_data_path.exists():
        checked = [str(config.processed_data), *(str(path) for path in FALLBACK_PROCESSED_PATHS)]
        raise FileNotFoundError("Processed dataset not found. Checked: " + ", ".join(checked))
    if not model_path.exists():
        checked = [str(config.model_path), *(str(path) for path in FALLBACK_MODEL_PATHS)]
        raise FileNotFoundError("Model checkpoint not found. Checked: " + ", ".join(checked))

    if processed_data_path != config.processed_data:
        print(f"Using fallback processed dataset: {processed_data_path}")
    if model_path != config.model_path:
        print(f"Using fallback model checkpoint: {model_path}")

    device = _resolve_device(config.device)
    print(_format_device_info(requested=config.device, resolved=device))

    bundle = torch.load(processed_data_path, map_location="cpu")
    split_X, split_y = _load_split(bundle, config.split)
    if split_X.ndim != 3:
        raise ValueError(
            f"Expected network split features [N, seq_len, feature_dim], got {tuple(split_X.shape)}"
        )

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    class_names = _resolve_class_names(bundle=bundle, checkpoint=checkpoint, labels=split_y)
    num_classes = int(checkpoint_config.get("num_classes", len(class_names)))
    if len(class_names) != num_classes:
        class_names = [f"class_{idx}" for idx in range(num_classes)]

    model = CNNLSTMClassifier(
        input_dim=int(checkpoint_config.get("input_dim", split_X.shape[-1])),
        num_classes=num_classes,
        conv_channels=int(checkpoint_config.get("conv_channels", 96)),
        conv_kernel_size=int(checkpoint_config.get("conv_kernel_size", 3)),
        flow_embedding_dim=int(checkpoint_config.get("flow_embedding_dim", 128)),
        lstm_hidden_dim=int(checkpoint_config.get("lstm_hidden_dim", 128)),
        lstm_layers=int(checkpoint_config.get("lstm_layers", 2)),
        dropout=float(checkpoint_config.get("dropout", 0.3)),
        bidirectional=bool(checkpoint_config.get("bidirectional", False)),
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    loader = build_dataloader(
        SequenceDataset(split_X, split_y),
        batch_size=config.batch_size,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()
    loss, logits, labels = evaluate_model(
        model,
        loader,
        criterion,
        device=device,
        input_dtype="float",
        progress_desc=f"{config.split}",
    )
    if logits.ndim == 2 and logits.shape[1] != len(class_names):
        class_names = [f"class_{idx}" for idx in range(int(logits.shape[1]))]

    report = compute_classification_report(
        loss=loss,
        labels=labels,
        logits=logits,
        class_names=class_names,
        normal_class_index=config.normal_class_index,
    )
    support_warnings = _support_warnings(report.to_dict().get("per_class", []))
    return {
        "task": "test_cicids_network_expert",
        "split": config.split,
        "processed_data": str(processed_data_path),
        "model_path": str(model_path),
        "num_samples": int(labels.shape[0]),
        "metrics": report.to_dict(),
        "warnings": support_warnings,
        "config": {**asdict(config), "processed_data": str(config.processed_data), "model_path": str(config.model_path)},
    }


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


def _resolve_class_names(bundle: dict[str, Any], checkpoint: Any, labels: torch.Tensor) -> list[str]:
    if isinstance(checkpoint, dict):
        checkpoint_names = checkpoint.get("class_names")
        if isinstance(checkpoint_names, (list, tuple)) and checkpoint_names:
            return [str(item) for item in checkpoint_names]

    bundle_names = bundle.get("class_names")
    if isinstance(bundle_names, (list, tuple)) and bundle_names:
        return [str(item) for item in bundle_names]

    num_classes = int(labels.max().item()) + 1 if labels.numel() > 0 else 2
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


def _resolve_model_path(requested: Path) -> Path:
    if requested.exists():
        return requested
    if requested != DEFAULT_MODEL_PATH:
        return requested
    for candidate in FALLBACK_MODEL_PATHS:
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


def _support_warnings(per_class_metrics: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for item in per_class_metrics:
        class_name = str(item.get("class_name", "unknown"))
        support = int(item.get("support", 0))
        if support < 20:
            warnings.append(
                f"Very low support for class '{class_name}' in evaluation split (support={support}). "
                "Metrics for this class are unstable."
            )
    return warnings


if __name__ == "__main__":
    main()
