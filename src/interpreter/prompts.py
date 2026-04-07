from __future__ import annotations

from typing import Any, Mapping


SYSTEM_PROMPT = (
    "You are a cloud security incident advisor for anomaly detection outputs. "
    "The anomaly_type is already classified by deterministic rules and must not be changed. "
    "Explain reason and action based on that classified anomaly context. "
    "Respond as strict JSON with exactly two keys: reason and action."
)


def build_incident_prompt(incident: Mapping[str, Any]) -> str:
    """Builds a compact prompt for Gemini incident explanation."""
    event_name = incident.get("event_name", "unknown-event")
    anomaly_detected = bool(incident.get("anomaly_detected", False))
    anomaly_type = incident.get("anomaly_type", "Normal")
    severity = incident.get("severity_level", "Low")
    score = float(incident.get("max_anomaly_score", 0.0))
    triggered = incident.get("triggered_experts", [])
    classification_source = incident.get("classification_source", "unknown")
    classification_confidence = float(incident.get("classification_confidence", 0.0))
    matched_rules = incident.get("classification_matched_rules", [])
    classification_description = incident.get("classification_description", "")
    event_names = incident.get("event_names", [])
    predictions = incident.get("predictions", [])

    return (
        "Incident context:\n"
        f"- event_name: {event_name}\n"
        f"- anomaly_detected: {anomaly_detected}\n"
        f"- anomaly_type: {anomaly_type}\n"
        f"- severity_level: {severity}\n"
        f"- max_anomaly_score: {score:.4f}\n"
        f"- triggered_experts: {triggered}\n"
        f"- classification_source: {classification_source}\n"
        f"- classification_confidence: {classification_confidence:.4f}\n"
        f"- classification_matched_rules: {matched_rules}\n"
        f"- classification_description: {classification_description}\n"
        f"- event_names: {event_names}\n"
        f"- expert_predictions: {predictions}\n\n"
        "Instructions:\n"
        "- Treat anomaly_type as final rule-based class.\n"
        "- reason: explain why this class fits these events and score.\n"
        "- action: give clear immediate mitigation steps specific to this class.\n"
        "Return strict JSON only with:\n"
        '{"reason":"<short cause>", "action":"<short immediate action list>"}'
    )
