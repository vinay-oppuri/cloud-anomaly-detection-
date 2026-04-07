from __future__ import annotations

from anomaly_classifier import classify_anomaly, format_alert, get_anomaly_statistics


def test_cascading_failure_has_priority() -> None:
    result = classify_anomaly(["E5", "E6", "E7", "E22"], 0.98)
    assert result["anomaly_type"] == "Cascading_Failure"
    assert result["severity"] == "Critical"


def test_node_failure_detected_with_dead_node_signal() -> None:
    result = classify_anomaly(["E24", "E9", "E14"], 0.90)
    assert result["anomaly_type"] == "Node_Failure"
    assert result["severity"] == "Critical"


def test_unknown_anomaly_fallback() -> None:
    result = classify_anomaly(["E1", "E2", "E3", "E4"], 0.12)
    assert result["anomaly_type"] == "Unknown_System_Anomaly"
    assert result["severity"] == "Medium"


def test_format_alert_for_data_corruption_action() -> None:
    classified = classify_anomaly(["E22"], 0.96)
    alert = format_alert(classified, "blk-1")
    assert alert["anomaly_type"] == "Data_Corruption"
    assert alert["recommended_action"] == "Trigger immediate block re-verification and re-replication"


def test_statistics_aggregates_counts_and_timeline() -> None:
    payload = [
        {
            "block_id": "b1",
            "score": 0.96,
            "event_names": ["E22"],
            "timestamp": "2026-04-07T08:00:00Z",
        },
        {
            "block_id": "b2",
            "score": 0.99,
            "event_names": ["E24"],
            "timestamp": "2026-04-07T08:05:00Z",
        },
    ]
    stats = get_anomaly_statistics(payload)
    assert stats["counts_per_anomaly_type"]["Data_Corruption"] == 1
    assert stats["counts_per_anomaly_type"]["Node_Failure"] == 1
    assert len(stats["timeline"]) == 2
