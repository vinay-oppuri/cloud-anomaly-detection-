from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import system_api


class _DummyService:
    def analyze_event_sequence(self, event_sequence: str, *, event_name: str = "uploaded-log") -> dict[str, object]:
        return {
            "task": "dummy",
            "mode": "analyze",
            "event_name": event_name,
            "anomaly_detected": True,
            "anomaly_type": "Anomaly",
            "reason": f"seq={event_sequence}",
            "action": "dummy-action",
            "metadata": {"advice_source": "dummy"},
        }

    def analyze_log_lines(self, lines: list[str], *, event_name: str = "uploaded-log") -> dict[str, object]:
        return {
            "task": "dummy",
            "mode": "analyze",
            "event_name": event_name,
            "anomaly_detected": False,
            "anomaly_type": "Normal",
            "reason": f"lines={len(lines)}",
            "action": "dummy-action",
            "metadata": {"advice_source": "dummy"},
        }


def test_health_endpoint() -> None:
    client = TestClient(system_api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_json_event_sequence(monkeypatch) -> None:
    monkeypatch.setattr(system_api, "get_system_service", lambda: _DummyService())
    client = TestClient(system_api.app)

    response = client.post(
        "/v1/system/analyze",
        json={"event_name": "evt-1", "event_sequence": "E5 E26 E11"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["event_name"] == "evt-1"
    assert payload["anomaly_detected"] is True
    assert payload["metadata"]["advice_source"] == "dummy"


def test_analyze_file_upload(monkeypatch) -> None:
    monkeypatch.setattr(system_api, "get_system_service", lambda: _DummyService())
    client = TestClient(system_api.app)

    response = client.post(
        "/v1/system/analyze-file",
        data={"event_name": "upload-1"},
        files={"log_file": ("logs.txt", b"line1\nline2\n", "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["event_name"] == "upload-1"
    assert payload["anomaly_detected"] is False
