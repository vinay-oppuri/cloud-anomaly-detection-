from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class RunningStats:
    """Running mean/variance state for online normalization."""

    count: int
    mean: torch.Tensor
    m2: torch.Tensor


class RunningZScoreNormalizer:
    """
    Online z-score normalizer for streaming tensor features.

    Expects the last dimension to be feature dimension.
    """

    def __init__(self, feature_dim: int, eps: float = 1e-6) -> None:
        self.feature_dim = feature_dim
        self.eps = eps
        self._stats = RunningStats(
            count=0,
            mean=torch.zeros(feature_dim, dtype=torch.float32),
            m2=torch.zeros(feature_dim, dtype=torch.float32),
        )

    def update(self, data: torch.Tensor) -> None:
        flattened = self._flatten_features(data)
        for row in flattened:
            self._stats.count += 1
            delta = row - self._stats.mean
            self._stats.mean = self._stats.mean + delta / self._stats.count
            delta2 = row - self._stats.mean
            self._stats.m2 = self._stats.m2 + (delta * delta2)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        flattened = self._flatten_features(data)
        variance = self._variance()
        std = torch.sqrt(variance + self.eps)
        normalized = (flattened - self._stats.mean) / std
        return normalized.reshape(data.shape)

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        self.update(data)
        return self.transform(data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "eps": self.eps,
            "count": self._stats.count,
            "mean": self._stats.mean.clone(),
            "m2": self._stats.m2.clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.feature_dim = int(state["feature_dim"])
        self.eps = float(state["eps"])
        self._stats = RunningStats(
            count=int(state["count"]),
            mean=state["mean"].clone().float(),
            m2=state["m2"].clone().float(),
        )

    def _variance(self) -> torch.Tensor:
        if self._stats.count < 2:
            return torch.ones(self.feature_dim, dtype=torch.float32)
        return self._stats.m2 / (self._stats.count - 1)

    def _flatten_features(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected last dimension={self.feature_dim}, got {data.shape[-1]}."
            )
        return data.reshape(-1, self.feature_dim).float()

