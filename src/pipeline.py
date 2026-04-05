from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.aggregator.ensemble import ExpertEnsemble, IncidentDecision
from src.collectors.system_collector import SystemLogCollector
from src.collectors.vpc_collector import VPCFlowCollector
from src.experts.network_expert.model import NetworkExpert
from src.experts.system_expert.model import SystemExpertTransformer
from src.interpreter.advisor import IncidentAdvisor
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
    use_gemini: bool = True
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str | None = None


class AnomalyDetectionPipeline:
    """Orchestrator for parsing logs, scoring anomalies, and generating guidance."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

        self.network_expert = NetworkExpert(
            input_dim=self.config.network_feature_dim,
            model_path=self.config.network_model_path,
        )
        self.system_expert = SystemExpertTransformer(
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
        self.advisor = IncidentAdvisor(
            use_gemini=self.config.use_gemini,
            gemini_model=self.config.gemini_model,
            api_key=self.config.gemini_api_key,
        )

    def process_event(self, event_name: str, vpc_flow_line: str, system_log_line: str) -> dict[str, Any]:
        """
        Parses and scores one workload event.

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
        payload = {"event_name": event_name, **decision.to_dict()}
        payload["anomaly_type"] = self._resolve_anomaly_type(payload)
        return payload

    def analyze_event(self, event_name: str, vpc_flow_line: str, system_log_line: str) -> dict[str, Any]:
        """
        End-to-end output for API use:
        anomaly detection -> anomaly type -> reason -> action.
        """
        incident = self.process_event(
            event_name=event_name,
            vpc_flow_line=vpc_flow_line,
            system_log_line=system_log_line,
        )
        advice = self.advisor.advise(incident)
        return {
            "event_name": incident["event_name"],
            "anomaly_detected": bool(incident["anomaly_detected"]),
            "anomaly_type": str(incident["anomaly_type"]),
            "reason": advice.reason,
            "action": advice.action,
            "metadata": {
                "severity_level": incident["severity_level"],
                "max_anomaly_score": incident["max_anomaly_score"],
                "triggered_experts": incident["triggered_experts"],
                "advice_source": advice.source,
            },
        }

    def _resolve_anomaly_type(self, incident: dict[str, Any]) -> str:
        generic_labels = {"", "normal", "anomaly", "attack", "unknown", "class_1"}
        if not bool(incident.get("anomaly_detected", False)):
            return "Normal"

        dominant = str(incident.get("dominant_anomaly_type", "")).strip()
        if dominant and dominant.lower() not in generic_labels:
            return dominant

        predictions = incident.get("predictions", [])
        best_specific = ""
        best_score = -1.0
        for item in predictions:
            score = float(item.get("anomaly_score", 0.0))
            predicted_class = str(item.get("predicted_class", "")).strip()
            if score < float(incident.get("threshold", 0.0)):
                continue
            if not predicted_class:
                continue
            if predicted_class.lower() in generic_labels:
                continue
            if score > best_score:
                best_score = score
                best_specific = predicted_class

        if best_specific:
            return best_specific

        return dominant or "Unknown"
