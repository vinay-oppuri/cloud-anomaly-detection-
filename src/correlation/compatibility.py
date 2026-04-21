from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

import numpy as np


NETWORK_TO_SYSTEM_COMPATIBILITY: dict[str, dict[str, float]] = {
    "DDoS": {
        "Network_Connection_Error": 1.00,
        "Pipeline_Failure": 0.92,
        "Cascading_Failure": 0.85,
        "Node_Failure": 0.55,
        "Data_Corruption": 0.24,
    },
    "DoS": {
        "Pipeline_Failure": 1.00,
        "Network_Connection_Error": 0.90,
        "Cascading_Failure": 0.72,
        "PacketResponder_Crash": 0.52,
        "Data_Corruption": 0.18,
    },
    "BruteForce": {
        "PacketResponder_Crash": 0.82,
        "Storage_Write_Failure": 0.55,
        "Replication_Failure": 0.35,
        "Network_Connection_Error": 0.32,
        "Cascading_Failure": 0.28,
        "Data_Corruption": 0.20,
    },
    "WebAttack": {
        "Data_Corruption": 0.95,
        "Storage_Write_Failure": 0.74,
        "Replication_Failure": 0.60,
        "Cascading_Failure": 0.24,
    },
    "Botnet": {
        "Cascading_Failure": 0.82,
        "Pipeline_Failure": 0.70,
        "Unknown_System_Anomaly": 0.55,
        "Data_Corruption": 0.44,
        "Node_Failure": 0.48,
    },
    "OtherAttack": {
        "Unknown_System_Anomaly": 0.75,
        "Replication_Failure": 0.42,
        "Data_Corruption": 0.38,
        "Cascading_Failure": 0.34,
        "Pipeline_Failure": 0.26,
        "Network_Connection_Error": 0.22,
    },
    "Benign": {
        "Normal": 1.00,
    },
}


SEVERITY_TO_SCORE = {
    "Low": 0.25,
    "Medium": 0.50,
    "High": 0.75,
    "Critical": 1.00,
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def compatibility_score(network_label: str, system_label: str) -> float:
    direct = NETWORK_TO_SYSTEM_COMPATIBILITY.get(network_label, {}).get(system_label)
    if direct is not None:
        return float(direct)
    reverse = NETWORK_TO_SYSTEM_COMPATIBILITY.get(system_label, {}).get(network_label)
    if reverse is not None:
        return float(reverse)
    if network_label == system_label:
        return 0.85
    return 0.08


def severity_score(severity: str) -> float:
    return float(SEVERITY_TO_SCORE.get(str(severity), 0.50))


def deterministic_label_embedding(text: str, dim: int = 32) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    tokens = _TOKEN_PATTERN.findall(str(text).lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, 16, 2):
            index = int.from_bytes(digest[offset : offset + 2], "big") % dim
            sign = 1.0 if digest[offset] % 2 == 0 else -1.0
            vector[index] += sign

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def compress_embedding(values: np.ndarray, target_dim: int = 64) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return np.zeros(target_dim, dtype=np.float32)
    if array.size == target_dim:
        return array.copy()
    if array.size < target_dim:
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[: array.size] = array
        return padded

    indices = np.linspace(0, array.size - 1, num=target_dim, dtype=np.float32)
    left = np.floor(indices).astype(np.int64)
    right = np.ceil(indices).astype(np.int64)
    alpha = indices - left
    compressed = (1.0 - alpha) * array[left] + alpha * array[right]
    return compressed.astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vec = np.asarray(left, dtype=np.float32).reshape(-1)
    right_vec = np.asarray(right, dtype=np.float32).reshape(-1)
    left_norm = float(np.linalg.norm(left_vec))
    right_norm = float(np.linalg.norm(right_vec))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))


def build_common_embedding(
    *,
    raw_embedding: np.ndarray,
    label: str,
    event_type: str,
    anomaly_score: float,
    severity: str,
    target_dim: int = 104,
) -> np.ndarray:
    compressed = compress_embedding(raw_embedding, target_dim=64)
    label_embedding = deterministic_label_embedding(label, dim=32)
    type_features = np.asarray(
        [
            1.0 if event_type == "network" else 0.0,
            1.0 if event_type == "log" else 0.0,
        ],
        dtype=np.float32,
    )
    scalar_features = np.asarray(
        [float(anomaly_score), severity_score(severity), math.tanh(float(anomaly_score) * 2.0)],
        dtype=np.float32,
    )
    vector = np.concatenate([compressed, label_embedding, type_features, scalar_features], axis=0)
    if vector.size < target_dim:
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[: vector.size] = vector
        vector = padded
    elif vector.size > target_dim:
        vector = vector[:target_dim]

    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    return vector.astype(np.float32)


def merge_embeddings(embeddings: Iterable[np.ndarray]) -> np.ndarray:
    vectors = [np.asarray(item, dtype=np.float32).reshape(-1) for item in embeddings]
    if not vectors:
        raise ValueError("At least one embedding is required to merge.")
    merged = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = float(np.linalg.norm(merged))
    if norm > 0.0:
        merged = merged / norm
    return merged.astype(np.float32)
