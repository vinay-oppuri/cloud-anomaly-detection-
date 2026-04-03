from __future__ import annotations

from typing import Mapping


def build_advisor_prompt(
    predicted_anomaly: str,
    confidence: float,
    raw_log_snippet: str,
    expert_scores: Mapping[str, float],
) -> str:
    """Builds a strict JSON prompt for Gemini."""

    scores_block = "\n".join(
        f"- {expert_name}: {score:.4f}" for expert_name, score in sorted(expert_scores.items())
    )
    if not scores_block:
        scores_block = "- No expert scores were provided."

    return f"""
You are a cloud security incident response assistant.
You are given machine-learning predictions and raw cloud log evidence.

Model signal:
- Predicted anomaly type: {predicted_anomaly}
- Confidence: {confidence:.4f}
- Expert anomaly scores:
{scores_block}

Raw log snippet:
{raw_log_snippet}

Return ONLY a valid JSON object with this exact schema:
{{
  "anomaly_type": "string",
  "root_cause_reason": "string",
  "remediation_action": "string",
  "severity_level": "Low|Med|High|Critical"
}}

Rules:
1) Confirm or correct anomaly_type using the log evidence.
2) root_cause_reason must be plain English and concise.
3) remediation_action must include actionable steps and at least one concrete CLI command.
4) severity_level must be one of: Low, Med, High, Critical.
5) Do not include markdown, commentary, or any extra fields.
""".strip()

