from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(slots=True)
class AnomalyEvent:
    event_id: str
    timestamp: datetime
    event_type: str
    label: str
    anomaly_score: float
    confidence: float
    embedding: np.ndarray
    source_ref: str
    severity: str
    metadata: dict[str, Any] = field(default_factory=dict)
    synthetic_parent_id: str | None = None
    synthetic_chain_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "type": self.event_type,
            "label": self.label,
            "anomaly_score": round(float(self.anomaly_score), 6),
            "confidence": round(float(self.confidence), 6),
            "embedding": self.embedding.astype(float).round(6).tolist(),
            "source_ref": self.source_ref,
            "severity": self.severity,
            "metadata": self.metadata,
            "synthetic_parent_id": self.synthetic_parent_id,
            "synthetic_chain_id": self.synthetic_chain_id,
        }


@dataclass(slots=True)
class CorrelationEdge:
    source_id: str
    target_id: str
    relation: str
    score: float
    temporal_score: float
    semantic_score: float
    causal_score: float
    attention_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation,
            "score": round(float(self.score), 6),
            "temporal_score": round(float(self.temporal_score), 6),
            "semantic_score": round(float(self.semantic_score), 6),
            "causal_score": round(float(self.causal_score), 6),
            "attention_score": round(float(self.attention_score), 6),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CorrelationCluster:
    cluster_id: str
    node_ids: list[str]
    root_cause_id: str
    summary: dict[str, Any]
    attack_chain: list[dict[str, Any]]
    mitigation: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "node_ids": self.node_ids,
            "root_cause_id": self.root_cause_id,
            "summary": self.summary,
            "attack_chain": self.attack_chain,
            "mitigation": self.mitigation,
        }
