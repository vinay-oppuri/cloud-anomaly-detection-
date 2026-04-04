from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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
        default_class_names = (
            "Normal",
            "Credential Abuse",
            "Privilege Escalation",
            "Malicious Script Execution",
            "Persistence Attempt",
        )

        checkpoint_config: dict[str, Any] = {}
        checkpoint_class_names: tuple[str, ...] = ()
        model_path_obj = Path(model_path) if model_path is not None else None
        if model_path_obj is not None:
            checkpoint_config, checkpoint_class_names = self._peek_checkpoint(model_path_obj)

        resolved_class_names = self._resolve_class_names(
            explicit_class_names=class_names,
            checkpoint_class_names=checkpoint_class_names,
            default_class_names=default_class_names,
            checkpoint_num_classes=checkpoint_config.get("num_classes"),
        )
        resolved_num_classes = len(resolved_class_names)
        self.class_names: tuple[str, ...] = resolved_class_names

        resolved_vocab_size = int(checkpoint_config.get("vocab_size", vocab_size))
        resolved_embedding_dim = int(checkpoint_config.get("embedding_dim", 64))
        resolved_hidden_dim = int(checkpoint_config.get("hidden_dim", 128))
        resolved_lstm_layers = int(checkpoint_config.get("lstm_layers", 1))
        resolved_dropout = float(checkpoint_config.get("dropout", 0.2))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = BiLSTMLogClassifier(
            vocab_size=resolved_vocab_size,
            num_classes=resolved_num_classes,
            embedding_dim=resolved_embedding_dim,
            hidden_dim=resolved_hidden_dim,
            lstm_layers=resolved_lstm_layers,
            dropout=resolved_dropout,
        ).to(self.device)

        if model_path_obj is not None:
            self._load_weights(model_path_obj)

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

    def _peek_checkpoint(self, model_path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
        if not model_path.exists():
            return {}, ()

        checkpoint = torch.load(model_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            return {}, ()

        config = checkpoint.get("config")
        class_names_raw = checkpoint.get("class_names")
        parsed_config = config if isinstance(config, dict) else {}
        parsed_class_names = (
            tuple(str(item) for item in class_names_raw)
            if isinstance(class_names_raw, (list, tuple))
            else ()
        )
        return parsed_config, parsed_class_names

    def _resolve_class_names(
        self,
        *,
        explicit_class_names: Sequence[str] | None,
        checkpoint_class_names: tuple[str, ...],
        default_class_names: tuple[str, ...],
        checkpoint_num_classes: Any,
    ) -> tuple[str, ...]:
        if explicit_class_names is not None:
            candidate = tuple(explicit_class_names)
        elif checkpoint_class_names:
            candidate = checkpoint_class_names
        else:
            candidate = default_class_names

        inferred_num_classes = (
            int(checkpoint_num_classes)
            if isinstance(checkpoint_num_classes, int) and checkpoint_num_classes > 0
            else len(candidate)
        )
        if len(candidate) == inferred_num_classes:
            return candidate

        if explicit_class_names is not None or checkpoint_class_names:
            raise ValueError(
                "Class name count does not match checkpoint class count. "
                "Pass matching class names or use a compatible checkpoint."
            )

        generated = [f"class_{idx}" for idx in range(inferred_num_classes)]
        return tuple(generated)
