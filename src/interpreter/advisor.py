from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from src.interpreter.prompts import build_advisor_prompt

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency in local dev.
    genai = None  # type: ignore[assignment]


@dataclass(slots=True)
class AdvisorResponse:
    """Structured LLM response consumed by the orchestration layer."""

    anomaly_type: str
    root_cause_reason: str
    remediation_action: str
    severity_level: str

    def to_dict(self) -> dict[str, str]:
        return {
            "anomaly_type": self.anomaly_type,
            "root_cause_reason": self.root_cause_reason,
            "remediation_action": self.remediation_action,
            "severity_level": self.severity_level,
        }


class AnomalyAdvisor:
    """Asynchronous Gemini-based interpreter for anomaly decisions."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        timeout_seconds: float = 12.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self._client: Any | None = None

        if self.api_key and genai is not None:
            self._client = genai.Client(api_key=self.api_key)

    async def advise(
        self,
        predicted_anomaly: str,
        confidence: float,
        raw_log_snippet: str,
        expert_scores: Mapping[str, float],
    ) -> AdvisorResponse:
        """
        Resolve final anomaly advice using Gemini, with deterministic fallback.

        This method is fully async and does not block the detection pipeline.
        """

        prompt = build_advisor_prompt(
            predicted_anomaly=predicted_anomaly,
            confidence=confidence,
            raw_log_snippet=raw_log_snippet,
            expert_scores=expert_scores,
        )

        if self._client is None:
            return self._fallback_response(predicted_anomaly, confidence)

        try:
            response_text = await asyncio.wait_for(
                self._generate_content(prompt),
                timeout=self.timeout_seconds,
            )
            payload = self._parse_json_payload(response_text)
            return self._payload_to_response(payload, predicted_anomaly, confidence)
        except Exception:
            return self._fallback_response(predicted_anomaly, confidence)

    async def _generate_content(self, prompt: str) -> str:
        assert self._client is not None
        client = self._client

        aio_client = getattr(client, "aio", None)
        if aio_client is not None and getattr(aio_client, "models", None) is not None:
            response = await aio_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
        else:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=prompt,
            )

        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        extracted = self._extract_text(response)
        if extracted:
            return extracted
        raise RuntimeError("Gemini returned an empty response.")

    def _extract_text(self, response: Any) -> str:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""

        text_parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None)
            if parts is None:
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    text_parts.append(part_text.strip())

        return "\n".join(text_parts)

    def _parse_json_payload(self, response_text: str) -> dict[str, Any]:
        normalized = response_text.strip()
        normalized = re.sub(r"^```(?:json)?", "", normalized, flags=re.IGNORECASE).strip()
        normalized = re.sub(r"```$", "", normalized).strip()

        try:
            payload = json.loads(normalized)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload

        raise ValueError("Could not parse JSON payload from Gemini response.")

    def _payload_to_response(
        self,
        payload: Mapping[str, Any],
        predicted_anomaly: str,
        confidence: float,
    ) -> AdvisorResponse:
        anomaly_type = str(payload.get("anomaly_type", predicted_anomaly)).strip()
        root_cause_reason = str(
            payload.get(
                "root_cause_reason",
                "The model confidence suggests abnormal behavior, but the LLM response was incomplete.",
            )
        ).strip()
        remediation_action = str(
            payload.get(
                "remediation_action",
                self._default_remediation(),
            )
        ).strip()
        severity_level = self._normalize_severity(
            str(payload.get("severity_level", "")).strip(),
            confidence,
        )

        return AdvisorResponse(
            anomaly_type=anomaly_type or predicted_anomaly,
            root_cause_reason=root_cause_reason,
            remediation_action=remediation_action,
            severity_level=severity_level,
        )

    def _fallback_response(self, predicted_anomaly: str, confidence: float) -> AdvisorResponse:
        severity_level = self._normalize_severity("", confidence)
        return AdvisorResponse(
            anomaly_type=predicted_anomaly,
            root_cause_reason=(
                "Fallback response: expert models detected suspicious activity and "
                "Gemini was unavailable or timed out."
            ),
            remediation_action=self._default_remediation(),
            severity_level=severity_level,
        )

    def _default_remediation(self) -> str:
        return (
            "1) Investigate recent events: aws logs tail /aws/cloudtrail --since 30m\n"
            "2) Isolate suspect source IP: aws ec2 revoke-security-group-ingress "
            "--group-id <sg-id> --protocol tcp --port 0-65535 --cidr <ip>/32\n"
            "3) Disable compromised credentials: aws iam update-access-key "
            "--user-name <user> --access-key-id <key-id> --status Inactive"
        )

    def _normalize_severity(self, severity_level: str, confidence: float) -> str:
        normalized_map = {
            "low": "Low",
            "med": "Med",
            "medium": "Med",
            "high": "High",
            "critical": "Critical",
        }

        key = severity_level.strip().lower()
        if key in normalized_map:
            return normalized_map[key]

        if confidence >= 0.9:
            return "Critical"
        if confidence >= 0.75:
            return "High"
        if confidence >= 0.5:
            return "Med"
        return "Low"

