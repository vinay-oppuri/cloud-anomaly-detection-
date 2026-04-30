from __future__ import annotations

from datetime import datetime

import numpy as np

from src.correlation.pair_features import PAIR_FEATURE_DIM, build_pair_features
from src.correlation.schema import AnomalyEvent


def _event(
    *,
    event_id: str,
    event_type: str,
    label: str,
    anomaly_score: float,
    confidence: float,
    severity: str,
    embedding: list[float],
    source_ref: str = "sample",
    metadata: dict | None = None,
    synthetic_parent_id: str | None = None,
    synthetic_chain_id: str | None = None,
) -> AnomalyEvent:
    return AnomalyEvent(
        event_id=event_id,
        timestamp=datetime(2026, 4, 1, 12, 0, 0),
        event_type=event_type,
        label=label,
        anomaly_score=anomaly_score,
        confidence=confidence,
        embedding=np.asarray(embedding, dtype=np.float32),
        source_ref=source_ref,
        severity=severity,
        metadata={} if metadata is None else metadata,
        synthetic_parent_id=synthetic_parent_id,
        synthetic_chain_id=synthetic_chain_id,
    )


def test_anomaly_event_to_dict_serializes_and_rounds() -> None:
    event = _event(
        event_id="network-001",
        event_type="network",
        label="Botnet",
        anomaly_score=0.987654321,
        confidence=0.123456789,
        severity="High",
        embedding=[0.1111119, 0.2222229, 0.3333339],
        metadata={"source_file": "capture-a"},
        synthetic_chain_id="chain-network-001",
    )

    payload = event.to_dict()

    assert payload["timestamp"] == "2026-04-01T12:00:00"
    assert payload["anomaly_score"] == 0.987654
    assert payload["confidence"] == 0.123457
    assert payload["embedding"] == [0.111112, 0.222223, 0.333334]
    assert payload["synthetic_chain_id"] == "chain-network-001"


def test_build_pair_features_includes_parent_and_chain_hints() -> None:
    network_event = _event(
        event_id="network-007",
        event_type="network",
        label="Botnet",
        anomaly_score=0.95,
        confidence=0.91,
        severity="High",
        embedding=[1.0, 0.0, 0.0],
        metadata={"source_file": "capture-a"},
        synthetic_chain_id="chain-network-007",
    )
    log_event = _event(
        event_id="log-014",
        event_type="log",
        label="Node_Failure",
        anomaly_score=0.90,
        confidence=0.88,
        severity="High",
        embedding=[1.0, 0.0, 0.0],
        synthetic_parent_id="network-007",
        synthetic_chain_id="chain-network-007",
    )

    features = build_pair_features(
        left_event=network_event,
        right_event=log_event,
        delta_seconds=30.0,
        temporal_window_seconds=120.0,
        causal_score=0.6,
    )

    assert features.shape == (PAIR_FEATURE_DIM,)
    assert features.dtype == np.float32
    assert features[0] == np.float32(0.75)
    assert features[1] == np.float32(1.0)
    assert features[2] == np.float32(0.6)
    assert features[9] == np.float32(2.0)


def test_build_pair_features_marks_same_network_stream() -> None:
    left_event = _event(
        event_id="network-100",
        event_type="network",
        label="DDoS",
        anomaly_score=0.80,
        confidence=0.70,
        severity="Medium",
        embedding=[0.0, 1.0, 0.0],
        metadata={"source_file": "capture-b"},
    )
    right_event = _event(
        event_id="network-101",
        event_type="network",
        label="DDoS",
        anomaly_score=0.82,
        confidence=0.71,
        severity="Medium",
        embedding=[0.0, 1.0, 0.0],
        metadata={"source_file": "capture-b"},
    )

    features = build_pair_features(
        left_event=left_event,
        right_event=right_event,
        delta_seconds=10.0,
        temporal_window_seconds=120.0,
        causal_score=0.2,
    )

    assert features[8] == np.float32(1.0)
    assert features[9] == np.float32(0.0)
