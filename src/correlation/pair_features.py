from __future__ import annotations

import numpy as np

from src.correlation.compatibility import cosine_similarity, severity_score
from src.correlation.schema import AnomalyEvent


PAIR_FEATURE_DIM = 10


def build_pair_features(
    *,
    left_event: AnomalyEvent,
    right_event: AnomalyEvent,
    delta_seconds: float,
    temporal_window_seconds: float,
    causal_score: float,
) -> np.ndarray:
    temporal_score = max(0.0, 1.0 - (delta_seconds / max(1.0, temporal_window_seconds)))
    semantic_score = cosine_similarity(left_event.embedding, right_event.embedding)
    anomaly_gap = abs(float(left_event.anomaly_score) - float(right_event.anomaly_score))
    severity_alignment = 1.0 - abs(severity_score(left_event.severity) - severity_score(right_event.severity))
    same_label = 1.0 if left_event.label == right_event.label else 0.0
    same_chain_hint = (
        1.0
        if left_event.synthetic_chain_id is not None
        and left_event.synthetic_chain_id == right_event.synthetic_chain_id
        else 0.0
    )
    parent_rule_hint = 1.0 if _is_parent_rule_match(left_event, right_event) else 0.0
    same_stream_hint = 1.0 if _is_same_stream_hint(left_event, right_event) else 0.0

    return np.asarray(
        [
            temporal_score,
            semantic_score,
            float(causal_score),
            anomaly_gap,
            1.0 if left_event.event_type == "network" else 0.0,
            1.0 if right_event.event_type == "network" else 0.0,
            same_label,
            severity_alignment,
            same_stream_hint,
            parent_rule_hint + same_chain_hint,
        ],
        dtype=np.float32,
    )


def _is_parent_rule_match(left_event: AnomalyEvent, right_event: AnomalyEvent) -> bool:
    if left_event.event_type == "network" and right_event.event_type == "log":
        return right_event.synthetic_parent_id == left_event.event_id
    if left_event.event_type == "log" and right_event.event_type == "network":
        return left_event.synthetic_parent_id == right_event.event_id
    return False


def _is_same_stream_hint(left_event: AnomalyEvent, right_event: AnomalyEvent) -> bool:
    if left_event.event_type == "log" and right_event.event_type == "log":
        return (
            left_event.synthetic_chain_id is not None
            and left_event.synthetic_chain_id == right_event.synthetic_chain_id
        )
    if left_event.event_type == "network" and right_event.event_type == "network":
        same_source = (
            left_event.metadata.get("source_file") is not None
            and left_event.metadata.get("source_file") == right_event.metadata.get("source_file")
        )
        return same_source and left_event.label == right_event.label
    return False
