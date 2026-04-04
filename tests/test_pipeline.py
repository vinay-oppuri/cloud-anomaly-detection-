from __future__ import annotations

from src.pipeline import AnomalyDetectionPipeline, PipelineConfig


def test_pipeline_process_event_returns_standard_payload() -> None:
    pipeline = AnomalyDetectionPipeline(
        config=PipelineConfig(
            threshold=0.65,
            network_window_size=8,
            network_feature_dim=16,
            system_sequence_length=32,
            system_vocab_size=2000,
        )
    )

    result = pipeline.process_event(
        event_name="smoke",
        vpc_flow_line=(
            "2 111122223333 eni-00abc123 10.0.1.10 10.0.2.8 51512 443 6 12 824 "
            "1700000000 1700000060 ACCEPT OK"
        ),
        system_log_line=(
            "host=i-01 app=nginx severity=info msg='service healthy' user=svc-analytics"
        ),
    )

    assert result["event_name"] == "smoke"
    assert "anomaly_detected" in result
    assert "predictions" in result

