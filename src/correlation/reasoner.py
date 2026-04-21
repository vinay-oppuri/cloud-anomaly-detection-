from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.correlation.response import suggest_mitigations

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None


ENV_KEY_CANDIDATES: tuple[str, ...] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

CHAIN_SYSTEM_PROMPT = (
    "You are a security reasoning agent for cross-layer anomaly correlation. "
    "Infer likely relationships between correlated network and system anomalies, "
    "identify the root cause, reconstruct the attack chain, and propose mitigations. "
    "Respond as strict JSON only."
)


@dataclass(slots=True)
class ChainReasoningResult:
    title: str
    root_cause: str
    explanation: str
    attack_chain: list[dict[str, Any]]
    mitigation: list[str]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "root_cause": self.root_cause,
            "explanation": self.explanation,
            "attack_chain": self.attack_chain,
            "mitigation": self.mitigation,
            "source": self.source,
        }


class AttackChainReasoner:
    def __init__(
        self,
        *,
        use_gemini: bool = True,
        gemini_model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        self.use_gemini = use_gemini
        self.gemini_model = gemini_model
        self.api_key = api_key or _resolve_api_key()
        self._client = self._build_client()

    def reason(self, cluster_payload: Mapping[str, Any]) -> ChainReasoningResult:
        if self._client is not None:
            llm_result = self._reason_with_gemini(cluster_payload)
            if llm_result is not None:
                return llm_result
        return self._heuristic_reason(cluster_payload)

    def _build_client(self) -> Any | None:
        if not self.use_gemini or genai is None or not self.api_key:
            return None
        return genai.Client(api_key=self.api_key)

    def _reason_with_gemini(self, cluster_payload: Mapping[str, Any]) -> ChainReasoningResult | None:
        if self._client is None or genai_types is None:
            return None

        prompt = build_chain_prompt(cluster_payload)
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "root_cause": {"type": "string"},
                "explanation": {"type": "string"},
                "attack_chain": {"type": "array"},
                "mitigation": {"type": "array"},
            },
            "required": ["title", "root_cause", "explanation", "attack_chain", "mitigation"],
        }
        try:
            response = self._client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=CHAIN_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1,
                ),
            )
        except Exception:
            return None

        text = str(getattr(response, "text", "") or "").strip()
        payload = _safe_json_parse(text)
        if not payload:
            return None

        try:
            return ChainReasoningResult(
                title=str(payload["title"]),
                root_cause=str(payload["root_cause"]),
                explanation=str(payload["explanation"]),
                attack_chain=list(payload["attack_chain"]),
                mitigation=[str(item) for item in payload["mitigation"]],
                source="gemini",
            )
        except Exception:
            return None

    def _heuristic_reason(self, cluster_payload: Mapping[str, Any]) -> ChainReasoningResult:
        attack_chain = list(cluster_payload.get("attack_chain", []))
        labels = [str(item.get("label", "Unknown")) for item in attack_chain]
        network_steps = [item for item in attack_chain if item.get("type") == "network"]
        log_steps = [item for item in attack_chain if item.get("type") == "log"]
        root_step = attack_chain[0] if attack_chain else {"label": "Unknown", "type": "unknown"}
        title = f"{root_step.get('label', 'Unknown')} cross-layer anomaly chain"
        root_cause = str(root_step.get("label", "Unknown"))

        if network_steps and log_steps:
            explanation = (
                f"{network_steps[0].get('label', 'Network anomaly')} likely preceded "
                f"{', '.join(step.get('label', 'system impact') for step in log_steps[:3])}, "
                "forming a cross-layer attack path from network abuse into HDFS-side failures."
            )
        elif attack_chain:
            explanation = (
                "Correlated anomalies form a single high-confidence cluster, but only one layer "
                "contributed clear causal evidence."
            )
        else:
            explanation = "No sufficient correlated anomaly chain was available for explanation."

        mitigation = suggest_mitigations(labels)
        return ChainReasoningResult(
            title=title,
            root_cause=root_cause,
            explanation=explanation,
            attack_chain=attack_chain,
            mitigation=mitigation,
            source="heuristic",
        )


def build_chain_prompt(cluster_payload: Mapping[str, Any]) -> str:
    return (
        "Cross-layer anomaly cluster:\n"
        f"{json.dumps(cluster_payload, indent=2)}\n\n"
        "Tasks:\n"
        "- infer the likely root cause\n"
        "- reconstruct the attack chain in chronological order\n"
        "- explain the cross-layer relationship in plain English\n"
        "- provide concrete mitigations\n"
        "Return strict JSON with keys: title, root_cause, explanation, attack_chain, mitigation."
    )


def _resolve_api_key() -> str | None:
    for key in ENV_KEY_CANDIDATES:
        value = os.getenv(key)
        if value:
            return value
    dotenv = Path(".env")
    if dotenv.exists():
        for raw_line in dotenv.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", maxsplit=1)
            if name.strip() in ENV_KEY_CANDIDATES:
                return value.strip().strip('"').strip("'")
    return None


def _safe_json_parse(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            return {}
    return {}
