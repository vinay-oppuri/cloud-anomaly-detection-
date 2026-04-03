from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from src.experts.base_expert import BaseExpert, ExpertPrediction


class BiLSTMLogClassifier(nn.Module):
    """Bi-LSTM backbone for system-log sequence classification."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("System model expects input with shape [batch, seq_len].")

        embedded = self.embedding(token_ids.long())
        _, (hidden, _) = self.lstm(embedded)

        # The last forward and backward hidden states represent both sequence directions.
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        pooled = torch.cat((forward_hidden, backward_hidden), dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class SystemExpert(BaseExpert):
    """Expert specialized for host/system log anomaly detection."""

    def __init__(
        self,
        vocab_size: int = 5000,
        class_names: Sequence[str] | None = None,
        model_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(name="system_expert")
        self.class_names: tuple[str, ...] = tuple(
            class_names
            or (
                "Normal",
                "Credential Abuse",
                "Privilege Escalation",
                "Malicious Script Execution",
                "Persistence Attempt",
            )
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = BiLSTMLogClassifier(
            vocab_size=vocab_size,
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
        if data.ndim == 1:
            return data.unsqueeze(0).long()
        if data.ndim == 2:
            return data.long()
        raise ValueError("System input must be [seq_len] or [batch, seq_len].")

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

