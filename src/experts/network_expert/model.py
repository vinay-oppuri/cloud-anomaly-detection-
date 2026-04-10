from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from src.experts.base_expert import BaseExpert, ExpertPrediction
from src.experts.network_expert.constants import CANONICAL_CICIDS_15_CLASSES


class CNNLSTMClassifier(nn.Module):
    """Per-flow feature CNN + temporal LSTM classifier for CICIDS windows."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_channels: int = 96,
        conv_kernel_size: int = 3,
        flow_embedding_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be >= 2.")
        if conv_channels <= 0:
            raise ValueError("conv_channels must be positive.")
        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be positive.")
        if flow_embedding_dim <= 0:
            raise ValueError("flow_embedding_dim must be positive.")
        if lstm_hidden_dim <= 0:
            raise ValueError("lstm_hidden_dim must be positive.")
        if lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive.")

        padding = conv_kernel_size // 2
        self.input_dim = input_dim
        self.feature_cnn = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.flow_projection = nn.Linear(conv_channels, flow_embedding_dim)
        self.temporal_lstm = nn.LSTM(
            input_size=flow_embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Network model expects input with shape [batch, seq_len, features].")

        batch_size, seq_len, feature_dim = x.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features per flow, got {feature_dim}. "
                "Run CICIDS preprocessing with matching target_feature_count."
            )

        # Apply 1D CNN on each flow vector to learn cross-feature interactions.
        flow_vectors = x.reshape(batch_size * seq_len, 1, feature_dim)
        flow_features = self.feature_cnn(flow_vectors).squeeze(-1)
        flow_embeddings = self.flow_projection(flow_features).reshape(batch_size, seq_len, -1)

        # Model temporal evolution across consecutive flow embeddings.
        temporal_outputs, _ = self.temporal_lstm(flow_embeddings)
        sequence_repr = temporal_outputs[:, -1, :]
        logits = self.classifier(sequence_repr)
        return logits


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
        default_class_names = CANONICAL_CICIDS_15_CLASSES
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
        resolved_conv_channels = int(checkpoint_config.get("conv_channels", 96))
        resolved_conv_kernel_size = int(checkpoint_config.get("conv_kernel_size", 3))
        resolved_flow_embedding_dim = int(checkpoint_config.get("flow_embedding_dim", 128))
        resolved_lstm_hidden_dim = int(checkpoint_config.get("lstm_hidden_dim", 128))
        resolved_lstm_layers = int(checkpoint_config.get("lstm_layers", 2))
        resolved_dropout = float(checkpoint_config.get("dropout", 0.3))
        resolved_bidirectional = _to_bool(checkpoint_config.get("bidirectional", False))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CNNLSTMClassifier(
            input_dim=resolved_input_dim,
            num_classes=resolved_num_classes,
            conv_channels=resolved_conv_channels,
            conv_kernel_size=resolved_conv_kernel_size,
            flow_embedding_dim=resolved_flow_embedding_dim,
            lstm_hidden_dim=resolved_lstm_hidden_dim,
            lstm_layers=resolved_lstm_layers,
            dropout=resolved_dropout,
            bidirectional=resolved_bidirectional,
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
            normal_index = self.class_names.index("Benign")
            normal_probability = float(probabilities[normal_index].item())
            score = 1.0 - normal_probability
        elif "Normal" in self.class_names:
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


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


__all__ = ["CNNLSTMClassifier", "NetworkExpert"]

