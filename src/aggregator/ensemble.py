from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from src.experts.base_expert import BaseExpert, ExpertPrediction


@dataclass(slots=True)
class IncidentDecision:
    """Final incident decision produced by the expert council."""

    anomaly_detected: bool
    threshold: float
    dominant_expert: str | None
    dominant_anomaly_type: str | None
    max_anomaly_score: float
    severity_level: str
    triggered_experts: list[str]
    predictions: list[ExpertPrediction]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_detected": self.anomaly_detected,
            "threshold": self.threshold,
            "dominant_expert": self.dominant_expert,
            "dominant_anomaly_type": self.dominant_anomaly_type,
            "max_anomaly_score": self.max_anomaly_score,
            "severity_level": self.severity_level,
            "triggered_experts": self.triggered_experts,
            "predictions": [
                {
                    "expert_name": item.expert_name,
                    "anomaly_score": item.anomaly_score,
                    "predicted_class": item.predicted_class,
                    "confidence": item.confidence,
                    "metadata": item.metadata,
                }
                for item in self.predictions
            ],
        }


class ExpertEnsemble:
    """
    Aggregates anomaly predictions from all active experts.

    Incident is flagged when at least one expert anomaly score is above threshold.
    """

    def __init__(
        self,
        experts: Sequence[BaseExpert],
        threshold: float,
    ) -> None:
        if not experts:
            raise ValueError("At least one expert must be provided.")
        self.experts: tuple[BaseExpert, ...] = tuple(experts)
        self.threshold = float(max(0.0, min(threshold, 1.0)))

    def evaluate(
        self,
        expert_inputs: Mapping[str, torch.Tensor],
    ) -> IncidentDecision:
        missing = [expert.name for expert in self.experts if expert.name not in expert_inputs]
        if missing:
            missing_names = ", ".join(missing)
            raise ValueError(f"Missing input tensors for expert(s): {missing_names}")

        predictions: list[ExpertPrediction] = []

        for expert in self.experts:
            input_tensor = expert_inputs[expert.name]
            predictions.append(expert.predict(input_tensor))

        triggered_predictions = [
            item for item in predictions if item.anomaly_score >= self.threshold
        ]

        dominant = max(predictions, key=lambda item: item.anomaly_score, default=None)
        max_score = dominant.anomaly_score if dominant is not None else 0.0

        return IncidentDecision(
            anomaly_detected=bool(triggered_predictions),
            threshold=self.threshold,
            dominant_expert=dominant.expert_name if dominant is not None else None,
            dominant_anomaly_type=dominant.predicted_class if dominant is not None else None,
            max_anomaly_score=max_score,
            severity_level=self._severity_from_score(max_score),
            triggered_experts=[item.expert_name for item in triggered_predictions],
            predictions=predictions,
        )

    def _severity_from_score(self, score: float) -> str:
        if score >= 0.9:
            return "Critical"
        if score >= 0.75:
            return "High"
        if score >= 0.5:
            return "Med"
        return "Low"
