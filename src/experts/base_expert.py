from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class ExpertPrediction:
    """Standard prediction payload shared by all expert models."""

    expert_name: str
    anomaly_score: float
    predicted_class: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseExpert(ABC):
    """Contract for all anomaly detection experts."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def predict(self, data: torch.Tensor) -> ExpertPrediction:
        """
        Predict anomaly information for a single input sample or mini-batch.

        Args:
            data: Tensor input formatted for the concrete expert model.

        Returns:
            ExpertPrediction with normalized anomaly score and class details.
        """

