from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.experts.network_expert.model import CNNLSTMClassifier
from src.training.cicids_preprocess import build_sequences, drop_low_support_labels


def test_drop_low_support_labels_removes_explicit_and_rare_classes() -> None:
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5, 6],
            "Label": ["Benign", "Benign", "Benign", "Botnet", "Botnet", "DoS_Slowloris"],
        }
    )
    cleaned, removed = drop_low_support_labels(
        df,
        min_rows=2,
        explicit_drop={"Botnet"},
    )
    assert set(cleaned["Label"].unique().tolist()) == {"Benign"}
    assert removed["Botnet"] == 2
    assert removed["DoS_Slowloris"] == 1


def test_build_sequences_uses_last_label_in_window() -> None:
    x = np.arange(24, dtype=np.float32).reshape(6, 4)
    y = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    x_seq, y_seq = build_sequences(x, y, seq_len=4)
    assert x_seq.shape == (2, 4, 4)
    assert y_seq.tolist() == [1, 1]


def test_cnn_lstm_forward_shape() -> None:
    model = CNNLSTMClassifier(
        input_dim=80,
        num_classes=15,
        conv_channels=32,
        flow_embedding_dim=48,
        lstm_hidden_dim=64,
        lstm_layers=1,
        dropout=0.1,
    )
    x = torch.randn(4, 32, 80)
    logits = model(x)
    assert logits.shape == (4, 15)


def test_cnn_lstm_rejects_feature_mismatch() -> None:
    model = CNNLSTMClassifier(input_dim=80, num_classes=15)
    with pytest.raises(ValueError):
        model(torch.randn(2, 8, 79))
