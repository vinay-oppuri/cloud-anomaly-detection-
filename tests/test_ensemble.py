from __future__ import annotations

import torch

from src.aggregator.ensemble import ExpertEnsemble
from src.experts.base_expert import BaseExpert, ExpertPrediction


class ConstantExpert(BaseExpert):
    def __init__(self, name: str, score: float, predicted_class: str) -> None:
        super().__init__(name=name)
        self._score = score
        self._predicted_class = predicted_class

    def predict(self, data: torch.Tensor) -> ExpertPrediction:
        return ExpertPrediction(
            expert_name=self.name,
            anomaly_score=self._score,
            predicted_class=self._predicted_class,
            confidence=0.9,
            metadata={},
        )


def test_ensemble_triggers_when_any_expert_crosses_threshold() -> None:
    network = ConstantExpert(name="network_expert", score=0.8, predicted_class="DDoS")
    system = ConstantExpert(name="system_expert", score=0.2, predicted_class="Normal")
    ensemble = ExpertEnsemble(experts=[network, system], threshold=0.65)

    result = ensemble.evaluate(
        expert_inputs={
            "network_expert": torch.zeros((8, 16), dtype=torch.float32),
            "system_expert": torch.zeros((32,), dtype=torch.long),
        }
    )

    assert result.anomaly_detected is True
    assert result.dominant_expert == "network_expert"
    assert result.dominant_anomaly_type == "DDoS"
    assert "network_expert" in result.triggered_experts

