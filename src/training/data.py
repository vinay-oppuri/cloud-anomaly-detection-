from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class LoadedDataset:
    """In-memory dataset tensors ready for DataLoader."""

    features: torch.Tensor
    labels: torch.Tensor


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset wrapper for sequence tensors and class labels."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Feature count ({features.shape[0]}) does not match label count ({labels.shape[0]})."
            )
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def load_processed_dataset(
    path: str | Path,
    x_key: str = "X",
    y_key: str = "y",
    feature_dtype: torch.dtype = torch.float32,
) -> LoadedDataset:
    """
    Loads processed training data from `.npz` or `.pt`.

    Expected formats:
    - npz: keys `X` and `y` (or custom keys via args)
    - pt: dict containing feature/label keys
    """

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".npz":
        payload = _load_npz(dataset_path, x_key=x_key, y_key=y_key)
    elif suffix == ".pt":
        payload = _load_pt(dataset_path, x_key=x_key, y_key=y_key)
    else:
        raise ValueError(f"Unsupported dataset format '{suffix}'. Use .npz or .pt")

    features = torch.as_tensor(payload.features, dtype=feature_dtype)
    labels = torch.as_tensor(payload.labels, dtype=torch.long)

    if labels.ndim != 1:
        labels = labels.reshape(-1)

    return LoadedDataset(features=features, labels=labels)


def _load_npz(path: Path, x_key: str, y_key: str) -> LoadedDataset:
    with np.load(path) as data:
        if x_key not in data or y_key not in data:
            keys = ", ".join(data.files)
            raise KeyError(f"Expected keys '{x_key}' and '{y_key}' in {path}. Found: {keys}")
        features = np.asarray(data[x_key])
        labels = np.asarray(data[y_key])
    return LoadedDataset(features=torch.from_numpy(features), labels=torch.from_numpy(labels))


def _load_pt(path: Path, x_key: str, y_key: str) -> LoadedDataset:
    raw: Any = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(raw)}")

    feature_key = _resolve_key(raw, preferred=x_key, alternatives=("features", "inputs"))
    label_key = _resolve_key(raw, preferred=y_key, alternatives=("labels", "targets"))

    if feature_key is None or label_key is None:
        raise KeyError(
            f"Could not find feature/label keys in {path}. "
            f"Expected '{x_key}'/'{y_key}' or common aliases."
        )

    features = torch.as_tensor(raw[feature_key])
    labels = torch.as_tensor(raw[label_key])
    return LoadedDataset(features=features, labels=labels)


def _resolve_key(container: dict[str, Any], preferred: str, alternatives: tuple[str, ...]) -> str | None:
    if preferred in container:
        return preferred
    for key in alternatives:
        if key in container:
            return key
    return None

