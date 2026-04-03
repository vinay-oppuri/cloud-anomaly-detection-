from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from src.experts.base_expert import BaseExpert, ExpertPrediction
from src.interpreter.advisor import AdvisorResponse, AnomalyAdvisor


@dataclass(slots=True)
class AggregationResult:
    """Final decision from the ensemble and optional LLM advisory."""

    triggered: bool
    threshold: float
    predictions: list[ExpertPrediction]
    advisor_response: AdvisorResponse | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "threshold": self.threshold,
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
            "advisor_response": (
                self.advisor_response.to_dict() if self.advisor_response is not None else None
            ),
        }


class ExpertEnsemble:
    """
    Aggregates expert scores and conditionally invokes the LLM interpreter.

    The interpreter is triggered when at least one expert anomaly score is above
    the configured threshold.
    """

    def __init__(
        self,
        experts: Sequence[BaseExpert],
        threshold: float,
        advisor: AnomalyAdvisor | None = None,
    ) -> None:
        if not experts:
            raise ValueError("At least one expert must be provided.")
        self.experts: tuple[BaseExpert, ...] = tuple(experts)
        self.threshold = float(max(0.0, min(threshold, 1.0)))
        self.advisor = advisor

    async def evaluate(
        self,
        expert_inputs: Mapping[str, torch.Tensor],
        raw_log_snippet: str,
    ) -> AggregationResult:
        predictions: list[ExpertPrediction] = []

        for expert in self.experts:
            input_tensor = expert_inputs.get(expert.name)
            if input_tensor is None:
                continue
            predictions.append(expert.predict(input_tensor))

        triggered_predictions = [
            item for item in predictions if item.anomaly_score >= self.threshold
        ]

        advisor_response: AdvisorResponse | None = None
        if triggered_predictions and self.advisor is not None:
            highest = max(triggered_predictions, key=lambda item: item.anomaly_score)
            advisor_response = await self.advisor.advise(
                predicted_anomaly=highest.predicted_class,
                confidence=highest.confidence,
                raw_log_snippet=raw_log_snippet,
                expert_scores={item.expert_name: item.anomaly_score for item in predictions},
            )

        return AggregationResult(
            triggered=bool(triggered_predictions),
            threshold=self.threshold,
            predictions=predictions,
            advisor_response=advisor_response,
        )

