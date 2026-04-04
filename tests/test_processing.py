from __future__ import annotations

import torch

from src.collectors.system_collector import SystemLogRecord
from src.collectors.vpc_collector import VPCFlowRecord
from src.processing.encoders import NetworkFeatureEncoder, SystemLogEncoder
from src.processing.normalizers import RunningZScoreNormalizer


def test_network_feature_encoder_output_shape() -> None:
    encoder = NetworkFeatureEncoder(window_size=8, feature_dim=16)
    record = VPCFlowRecord(
        src_ip="10.0.1.10",
        dst_ip="10.0.2.8",
        src_port=51512,
        dst_port=443,
        protocol=6,
        packets=12,
        bytes_transferred=824,
        action="ACCEPT",
        log_status="OK",
        raw="raw",
    )
    encoder.append(record)
    features = encoder.encode_current_window()

    assert features.shape == (8, 16)
    assert features.dtype == torch.float32


def test_system_log_encoder_output_shape() -> None:
    encoder = SystemLogEncoder(vocab_size=5000, sequence_length=32)
    record = SystemLogRecord(
        host="i-01",
        service="nginx",
        severity="INFO",
        user="svc",
        source_ip="10.0.0.5",
        message="service healthy",
        raw="raw",
    )
    tokens = encoder.encode_record(record)

    assert tokens.shape == (32,)
    assert tokens.dtype == torch.long


def test_running_normalizer_preserves_shape() -> None:
    normalizer = RunningZScoreNormalizer(feature_dim=4)
    sample = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]], dtype=torch.float32)
    output = normalizer.fit_transform(sample)

    assert output.shape == sample.shape
    assert torch.isfinite(output).all()

