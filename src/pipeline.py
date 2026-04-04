from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.aggregator.ensemble import ExpertEnsemble, IncidentDecision
from src.collectors.system_collector import SystemLogCollector
from src.collectors.vpc_collector import VPCFlowCollector
from src.experts.network_model import NetworkExpert
from src.experts.system_model import SystemExpert
from src.processing.encoders import NetworkFeatureEncoder, SystemLogEncoder
from src.processing.normalizers import RunningZScoreNormalizer


@dataclass(slots=True)
class PipelineConfig:
    """Runtime configuration for dual-expert anomaly detection pipeline."""

    threshold: float = 0.65
    network_window_size: int = 48
    network_feature_dim: int = 16
    system_sequence_length: int = 48
    system_vocab_size: int = 5000
    network_model_path: str = "models/network_expert_best.pth"
    system_model_path: str = "models/system_expert_best.pth"


class AnomalyDetectionPipeline:
    """Production-oriented orchestrator for network and system anomaly detection."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

        self.network_expert = NetworkExpert(
            input_dim=self.config.network_feature_dim,
            model_path=self.config.network_model_path,
        )
        self.system_expert = SystemExpert(
            vocab_size=self.config.system_vocab_size,
            model_path=self.config.system_model_path,
        )
        self.ensemble = ExpertEnsemble(
            experts=[self.network_expert, self.system_expert],
            threshold=self.config.threshold,
        )

        # Keep encoders aligned with actual loaded model dimensions from checkpoints.
        resolved_network_feature_dim = int(self.network_expert.model.conv1.in_channels)
        resolved_system_vocab_size = int(self.system_expert.model.embedding.num_embeddings)

        self.vpc_collector = VPCFlowCollector()
        self.system_collector = SystemLogCollector()
        self.network_encoder = NetworkFeatureEncoder(
            window_size=self.config.network_window_size,
            feature_dim=resolved_network_feature_dim,
        )
        self.system_encoder = SystemLogEncoder(
            vocab_size=resolved_system_vocab_size,
            sequence_length=self.config.system_sequence_length,
        )
        self.network_normalizer = RunningZScoreNormalizer(
            feature_dim=resolved_network_feature_dim
        )

    def process_event(self, event_name: str, vpc_flow_line: str, system_log_line: str) -> dict[str, Any]:
        """
        Processes one workload event and returns standardized incident output.

        Raises:
            ValueError: if collectors cannot parse one or both logs.
        """

        vpc_record = self.vpc_collector.parse_line(vpc_flow_line)
        system_record = self.system_collector.parse_line(system_log_line)
        if vpc_record is None:
            raise ValueError(f"Unable to parse VPC flow line for event '{event_name}'.")
        if system_record is None:
            raise ValueError(f"Unable to parse system log line for event '{event_name}'.")

        self.network_encoder.append(vpc_record)
        network_features = self.network_encoder.encode_current_window()
        network_features = self.network_normalizer.fit_transform(network_features)
        system_tokens = self.system_encoder.encode_record(system_record)

        decision: IncidentDecision = self.ensemble.evaluate(
            expert_inputs={
                self.network_expert.name: network_features,
                self.system_expert.name: system_tokens,
            }
        )
        return {"event_name": event_name, **decision.to_dict()}
