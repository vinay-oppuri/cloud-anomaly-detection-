from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from src.experts.base_expert import BaseExpert, ExpertPrediction


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for log-template sequences."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if max_len <= 0:
            raise ValueError("max_len must be positive.")

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected [batch, seq_len, hidden_dim] tensor for positional encoding.")
        return x + self.pe[:, : x.shape[1], :]


class AttentionPooling(nn.Module):
    """Learned pooling over encoder outputs for a stable sequence embedding."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected [batch, seq_len, hidden_dim] tensor for attention pooling.")
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(padding_mask, float("-inf"))
        all_padded = padding_mask.all(dim=1)
        if all_padded.any():
            logits = logits.clone()
            logits[all_padded] = 0.0
        weights = torch.softmax(logits, dim=1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class TransformerLogClassifier(nn.Module):
    """Transformer encoder over parsed log-template ids."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 160,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 384,
        dropout: float = 0.15,
        max_len: int = 512,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2.")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.padding_idx = int(padding_idx)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.padding_idx)
        self.position = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.embedding_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.pool = AttentionPooling(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.scale = math.sqrt(d_model)

    def forward_features(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("System transformer expects [batch, seq_len] token ids.")
        token_ids = token_ids.long()
        padding_mask = token_ids.eq(self.padding_idx)
        embedded = self.embedding(token_ids) * self.scale
        embedded = self.position(embedded)
        embedded = self.embedding_dropout(embedded)
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
        return self.pool(encoded, padding_mask=padding_mask)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(token_ids))


class SystemExpertTransformer(BaseExpert):
    """Inference wrapper around the HDFS transformer system expert."""

    def __init__(
        self,
        vocab_size: int = 4096,
        class_names: Sequence[str] | None = None,
        model_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(name="system_expert")
        default_class_names = ("Normal", "Anomaly")

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

        self.class_names: tuple[str, ...] = resolved_class_names
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = TransformerLogClassifier(
            vocab_size=int(checkpoint_config.get("vocab_size", vocab_size)),
            num_classes=len(self.class_names),
            d_model=int(checkpoint_config.get("d_model", 160)),
            nhead=int(checkpoint_config.get("nhead", 8)),
            num_layers=int(checkpoint_config.get("num_layers", 3)),
            dim_feedforward=int(checkpoint_config.get("dim_feedforward", 384)),
            dropout=float(checkpoint_config.get("dropout", 0.15)),
            max_len=int(checkpoint_config.get("max_len", 512)),
            padding_idx=int(checkpoint_config.get("padding_idx", 0)),
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
        raise ValueError("System transformer input must be [seq_len] or [batch, seq_len].")

    def _compute_anomaly_score(self, probabilities: torch.Tensor) -> float:
        if "Normal" in self.class_names:
            normal_index = self.class_names.index("Normal")
            score = 1.0 - float(probabilities[normal_index].item())
        else:
            score = float(torch.max(probabilities).item())
        return float(max(0.0, min(score, 1.0)))

    def _load_weights(self, model_path: Path) -> None:
        if not model_path.exists():
            return
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=False)

    def _peek_checkpoint(self, model_path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
        if not model_path.exists():
            return {}, ()

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
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
        return tuple(f"class_{idx}" for idx in range(inferred_num_classes))


__all__ = ["TransformerLogClassifier", "SystemExpertTransformer"]
