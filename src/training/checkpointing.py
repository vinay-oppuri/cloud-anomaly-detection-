from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class CheckpointPaths:
    """Output artifact paths for a training run."""

    best_model_path: Path
    last_model_path: Path
    metrics_path: Path


def resolve_checkpoint_paths(output_dir: str | Path, prefix: str) -> CheckpointPaths:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    return CheckpointPaths(
        best_model_path=base / f"{prefix}_best.pth",
        last_model_path=base / f"{prefix}_last.pth",
        metrics_path=base / f"{prefix}_metrics.json",
    )


def save_model_checkpoint(
    path: Path,
    *,
    model_state_dict: dict[str, Any],
    class_names: list[str],
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "state_dict": model_state_dict,
        "class_names": class_names,
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def save_metrics_report(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

