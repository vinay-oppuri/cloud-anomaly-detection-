from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.training.data import load_processed_dataset


def test_load_processed_dataset_from_npz(tmp_path: Path) -> None:
    path = tmp_path / "train_network.npz"
    X = np.arange(10 * 8 * 16, dtype=np.float32).reshape(10, 8, 16) / 1000.0
    y = (np.arange(10, dtype=np.int64) % 3).astype(np.int64)
    np.savez(path, X=X, y=y)

    loaded = load_processed_dataset(path, feature_dtype=torch.float32)
    assert loaded.features.shape == (10, 8, 16)
    assert loaded.labels.shape == (10,)


def test_load_processed_dataset_from_pt(tmp_path: Path) -> None:
    path = tmp_path / "train_system.pt"
    payload = {
        "features": torch.randint(0, 1000, size=(10, 32), dtype=torch.long),
        "labels": torch.randint(0, 4, size=(10,), dtype=torch.long),
    }
    torch.save(payload, path)

    loaded = load_processed_dataset(path, feature_dtype=torch.long)
    assert loaded.features.shape == (10, 32)
    assert loaded.labels.shape == (10,)
