from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from src.experts.base_expert import BaseExpert, ExpertPrediction


class CNNLSTMClassifier(nn.Module):
    """CNN-LSTM backbone for network-flow sequence classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_channels: int = 64,
        lstm_hidden_dim: int = 96,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Network model expects input with shape [batch, seq_len, features].")

        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        sequence_repr = x[:, -1, :]
        logits = self.classifier(sequence_repr)
        return logits


class NetworkExpert(BaseExpert):
    """Expert specialized for VPC/network-flow anomaly detection."""

    def __init__(
        self,
        input_dim: int = 16,
        class_names: Sequence[str] | None = None,
        model_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(name="network_expert")
        self.class_names: tuple[str, ...] = tuple(
            class_names
            or ("Normal", "DDoS", "Brute Force", "Port Scan", "Data Exfiltration")
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CNNLSTMClassifier(
            input_dim=input_dim,
            num_classes=len(self.class_names),
        ).to(self.device)

        if model_path is not None:
            self._load_weights(Path(model_path))

        self.model.eval()

    def predict(self, data: torch.Tensor) -> ExpertPrediction:
        input_batch = self._prepare_input(data).to(self.device)

        with torch.inference_mode():
            logits = self.model(input_batch)
            probabilities = torch.softmax(logits, dim=-1)[0]

        class_index = int(torch.argmax(probabilities).item())
        predicted_class = self.class_names[class_index]
        confidence = float(probabilities[class_index].item())
        anomaly_score = self._compute_anomaly_score(probabilities)

        class_probs = {
            class_name: float(probabilities[idx].item())
            for idx, class_name in enumerate(self.class_names)
        }

        return ExpertPrediction(
            expert_name=self.name,
            anomaly_score=anomaly_score,
            predicted_class=predicted_class,
            confidence=confidence,
            metadata={"class_probabilities": class_probs},
        )

    def _prepare_input(self, data: torch.Tensor) -> torch.Tensor:
        if data.ndim == 2:
            return data.unsqueeze(0).float()
        if data.ndim == 3:
            return data.float()
        raise ValueError("Network input must be [seq_len, features] or [batch, seq_len, features].")

    def _compute_anomaly_score(self, probabilities: torch.Tensor) -> float:
        if "Normal" in self.class_names:
            normal_index = self.class_names.index("Normal")
            normal_probability = float(probabilities[normal_index].item())
            score = 1.0 - normal_probability
        else:
            score = float(torch.max(probabilities).item())
        return float(max(0.0, min(score, 1.0)))

    def _load_weights(self, model_path: Path) -> None:
        if not model_path.exists():
            return

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)

