from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from src.experts.base_expert import BaseExpert, ExpertPrediction
from src.experts.network_expert.constants import ATTACK_FAMILY_CLASSES


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected [batch, seq_len, embedding_dim] tensor for positional encoding.")
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class CNNTransformerClassifier(nn.Module):
    """CNN projection over per-flow features followed by a temporal transformer encoder."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_channels: int = 128,
        conv_kernel_size: int = 3,
        flow_embedding_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        if flow_embedding_dim % transformer_heads != 0:
            raise ValueError("flow_embedding_dim must be divisible by transformer_heads.")

        padding = conv_kernel_size // 2
        self.input_dim = input_dim
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.flow_projection = nn.Sequential(
            nn.Linear(conv_channels, flow_embedding_dim),
            nn.LayerNorm(flow_embedding_dim),
            nn.GELU(),
        )
        self.positional_encoding = PositionalEncoding(flow_embedding_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=flow_embedding_dim,
            nhead=transformer_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.pool = AttentionPooling(flow_embedding_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(flow_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(flow_embedding_dim, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Network model expects input with shape [batch, seq_len, features].")

        batch_size, seq_len, feature_dim = x.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features per flow, got {feature_dim}. "
                "Run CICIDS preprocessing with matching feature schema."
            )

        flow_vectors = x.reshape(batch_size * seq_len, 1, feature_dim)
        flow_features = self.feature_encoder(flow_vectors).squeeze(-1)
        flow_embeddings = self.flow_projection(flow_features).reshape(batch_size, seq_len, -1)
        flow_embeddings = self.positional_encoding(flow_embeddings)
        temporal_outputs = self.temporal_encoder(flow_embeddings)
        return self.pool(temporal_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class NetworkExpert(BaseExpert):
    """Expert specialized for CICIDS network-flow anomaly detection."""

    def __init__(
        self,
        input_dim: int = 80,
        class_names: Sequence[str] | None = None,
        model_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(name="network_expert")
        default_class_names = ATTACK_FAMILY_CLASSES
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

        resolved_input_dim = int(checkpoint_config.get("input_dim", input_dim))
        resolved_conv_channels = int(checkpoint_config.get("conv_channels", 128))
        resolved_conv_kernel_size = int(checkpoint_config.get("conv_kernel_size", 3))
        resolved_flow_embedding_dim = int(checkpoint_config.get("flow_embedding_dim", 128))
        resolved_transformer_heads = int(checkpoint_config.get("transformer_heads", 4))
        resolved_transformer_layers = int(checkpoint_config.get("transformer_layers", 3))
        resolved_dim_feedforward = int(checkpoint_config.get("dim_feedforward", 256))
        resolved_dropout = float(checkpoint_config.get("dropout", 0.2))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CNNTransformerClassifier(
            input_dim=resolved_input_dim,
            num_classes=resolved_num_classes,
            conv_channels=resolved_conv_channels,
            conv_kernel_size=resolved_conv_kernel_size,
            flow_embedding_dim=resolved_flow_embedding_dim,
            transformer_heads=resolved_transformer_heads,
            transformer_layers=resolved_transformer_layers,
            dim_feedforward=resolved_dim_feedforward,
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
        if data.ndim == 2:
            return data.unsqueeze(0).float()
        if data.ndim == 3:
            return data.float()
        raise ValueError("Network input must be [seq_len, features] or [batch, seq_len, features].")

    def _compute_anomaly_score(self, probabilities: torch.Tensor) -> float:
        if "Benign" in self.class_names:
            benign_index = self.class_names.index("Benign")
            return float(max(0.0, min(1.0, 1.0 - float(probabilities[benign_index].item()))))
        return float(max(0.0, min(1.0, float(torch.max(probabilities).item()))))

    def _load_weights(self, model_path: Path) -> None:
        if not model_path.exists():
            return

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

        try:
            self.model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            warnings.warn(
                f"Skipping incompatible network checkpoint '{model_path}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

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


__all__ = ["CNNTransformerClassifier", "NetworkExpert"]
