from __future__ import annotations

from typing import Any, Mapping


SYSTEM_PROMPT = (
    "You are a cloud security incident advisor for anomaly detection outputs. "
    "Respond as strict JSON with exactly two keys: reason and action. "
    "Keep reason concise and evidence-based. Keep action practical and ordered."
)


def build_incident_prompt(incident: Mapping[str, Any]) -> str:
    """Builds a compact prompt for Gemini incident explanation."""
    event_name = incident.get("event_name", "unknown-event")
    anomaly_detected = bool(incident.get("anomaly_detected", False))
    anomaly_type = incident.get("anomaly_type", "Normal")
    severity = incident.get("severity_level", "Low")
    score = float(incident.get("max_anomaly_score", 0.0))
    triggered = incident.get("triggered_experts", [])
    predictions = incident.get("predictions", [])

    return (
        "Incident context:\n"
        f"- event_name: {event_name}\n"
        f"- anomaly_detected: {anomaly_detected}\n"
        f"- anomaly_type: {anomaly_type}\n"
        f"- severity_level: {severity}\n"
        f"- max_anomaly_score: {score:.4f}\n"
        f"- triggered_experts: {triggered}\n"
        f"- expert_predictions: {predictions}\n\n"
        "Return strict JSON only with:\n"
        '{"reason":"<short cause>", "action":"<short immediate action list>"}'
    )
