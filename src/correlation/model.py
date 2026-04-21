from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class AttentionCorrelationModel(nn.Module):
    """
    Attention-style pair scorer for cross-layer anomaly correlation.

    The module encodes each event vector, computes a directional attention score,
    then combines it with pair features in an MLP to predict whether two events
    belong to the same causal chain.
    """

    def __init__(
        self,
        input_dim: int = 104,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pair_feature_dim: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.pair_feature_dim = int(pair_feature_dim)
        self.event_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.pair_encoder = nn.Sequential(
            nn.Linear(self.pair_feature_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, event_vectors: torch.Tensor) -> torch.Tensor:
        if event_vectors.ndim != 2:
            raise ValueError("Expected event vectors with shape [n_events, input_dim].")
        return self.event_encoder(event_vectors)

    def score_pairs(
        self,
        left_vectors: torch.Tensor,
        right_vectors: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> torch.Tensor:
        if left_vectors.shape != right_vectors.shape:
            raise ValueError("Pair inputs must have the same shape.")
        left_hidden = self.encode(left_vectors)
        right_hidden = self.encode(right_vectors)
        query = self.query(left_hidden)
        key = self.key(right_hidden)
        value = self.value(right_hidden)
        attention = torch.sum(query * key, dim=-1, keepdim=True) / math.sqrt(self.hidden_dim)
        attended = torch.tanh(value * attention)
        pair_hidden = self.pair_encoder(pair_features)
        pair_input = torch.cat(
            [
                left_hidden,
                right_hidden,
                torch.abs(left_hidden - right_hidden),
                left_hidden * right_hidden,
                pair_hidden,
            ],
            dim=-1,
        )
        logits = self.pair_mlp(pair_input) + attention + 0.25 * torch.sum(attended, dim=-1, keepdim=True) / self.hidden_dim
        return logits.squeeze(-1)

    @torch.inference_mode()
    def predict_score(
        self,
        left_vector: np.ndarray,
        right_vector: np.ndarray,
        pair_features: np.ndarray,
        device: torch.device,
    ) -> float:
        left_tensor = torch.as_tensor(left_vector, dtype=torch.float32, device=device).unsqueeze(0)
        right_tensor = torch.as_tensor(right_vector, dtype=torch.float32, device=device).unsqueeze(0)
        pair_tensor = torch.as_tensor(pair_features, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self.score_pairs(left_tensor, right_tensor, pair_tensor)
        return float(torch.sigmoid(logits)[0].item())


@dataclass(slots=True)
class PairSample:
    left_id: str
    right_id: str
    pair_features: np.ndarray
    label: int
