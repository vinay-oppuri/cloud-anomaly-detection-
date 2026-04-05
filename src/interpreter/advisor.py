from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, TypedDict

from src.interpreter.prompts import SYSTEM_PROMPT, build_incident_prompt

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - optional dependency fallback.
    genai = None
    genai_types = None


class GeminiAdvicePayload(TypedDict):
    reason: str
    action: str


@dataclass(slots=True)
class IncidentAdvice:
    reason: str
    action: str
    source: str


class IncidentAdvisor:
    """
    Generates reason/action for an incident using Gemini.

    Falls back to deterministic heuristic advice when API key/dependency is missing
    or when Gemini call fails at runtime.
    """

    def __init__(
        self,
        *,
        use_gemini: bool = True,
        gemini_model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        self.use_gemini = use_gemini
        self.gemini_model = gemini_model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._client = self._build_client()

    def advise(self, incident: Mapping[str, Any]) -> IncidentAdvice:
        if self._client is not None:
            gemini_advice = self._advise_with_gemini(incident)
            if gemini_advice is not None:
                return gemini_advice
        return self._heuristic_advice(incident)

    def _build_client(self) -> Any | None:
        if not self.use_gemini:
            return None
        if genai is None:
            return None
        if not self.api_key:
            return None
        return genai.Client(api_key=self.api_key)

    def _advise_with_gemini(self, incident: Mapping[str, Any]) -> IncidentAdvice | None:
        if self._client is None or genai_types is None:
            return None

        prompt = build_incident_prompt(incident)
        try:
            response = self._client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=GeminiAdvicePayload,
                    temperature=0.1,
                ),
            )
        except Exception:
            return None

        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            return None

        payload = _safe_json_parse(text)
        reason = str(payload.get("reason", "")).strip()
        action = str(payload.get("action", "")).strip()
        if not reason or not action:
            return None

        return IncidentAdvice(reason=reason, action=action, source="gemini")

    def _heuristic_advice(self, incident: Mapping[str, Any]) -> IncidentAdvice:
        anomaly_detected = bool(incident.get("anomaly_detected", False))
        anomaly_type = str(incident.get("anomaly_type", "Normal"))
        severity = str(incident.get("severity_level", "Low"))
        triggered = incident.get("triggered_experts", [])
        triggered_text = ", ".join(str(item) for item in triggered) if triggered else "none"

        if not anomaly_detected:
            reason = "Current signal pattern is consistent with normal cloud workload behavior."
            action = (
                "Continue monitoring; keep current alert thresholds and verify periodic model drift checks."
            )
            return IncidentAdvice(reason=reason, action=action, source="heuristic")

        reason = (
            f"{severity} anomaly detected with type '{anomaly_type}' from expert signals ({triggered_text})."
        )
        action = (
            "Isolate affected workload, block suspicious traffic/credentials, and inspect related host and "
            "network logs for root cause."
        )
        return IncidentAdvice(reason=reason, action=action, source="heuristic")


def _safe_json_parse(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}
