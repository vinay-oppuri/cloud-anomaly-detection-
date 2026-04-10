from __future__ import annotations

import pytest
import torch

from src.experts.network_expert.constants import CANONICAL_CICIDS_15_CLASSES
from src.experts.network_expert.model import CNNLSTMClassifier
from src.experts.network_expert.preprocessor import (
    _canonicalize_label,
    _fit_feature_names,
    _fit_feature_vector,
    _ordered_class_names,
)


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        ("Benign", "Benign"),
        ("DoS attacks-Hulk", "DoS-Hulk"),
        ("DDOS attack-HOIC", "DDoS-HOIC"),
        ("DDoS attacks-LOIC-UDP", "DDoS-LOIC-UDP"),
        ("FTP-BruteForce", "Brute Force-FTP"),
        ("Brute Force -XSS", "Web Attack-XSS"),
        ("SQL Injection", "Web Attack-SQL Injection"),
    ],
)
def test_canonicalize_label_maps_common_cicids_variants(raw_label: str, expected: str) -> None:
    assert _canonicalize_label(raw_label) == expected


def test_canonicalize_label_drops_header_noise_rows() -> None:
    assert _canonicalize_label("Label") == ""


def test_canonicalize_label_drops_unknown_when_force_15_enabled() -> None:
    assert _canonicalize_label("SomeFutureAttack") == ""


def test_canonicalize_label_keeps_unknown_when_force_15_disabled() -> None:
    assert _canonicalize_label("SomeFutureAttack", force_15_class_schema=False) == "SomeFutureAttack"


def test_force_15_class_schema_is_stable() -> None:
    classes = _ordered_class_names(
        labels=["Benign", "DoS-Hulk", "DDoS-HOIC"],
        force_15_class_schema=True,
    )
    assert classes[: len(CANONICAL_CICIDS_15_CLASSES)] == list(CANONICAL_CICIDS_15_CLASSES)


def test_feature_padding_and_truncation_helpers() -> None:
    vector = _fit_feature_vector([1.0, 2.0], target_count=4)
    assert vector == [1.0, 2.0, 0.0, 0.0]

    names = _fit_feature_names(["a", "b", "c"], target_count=2)
    assert names == ["a", "b"]


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
