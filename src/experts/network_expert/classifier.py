from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypedDict

import numpy as np


class NetworkClassificationResult(TypedDict):
    anomaly_type: str
    severity: str
    confidence: float
    matched_rules: list[str]
    description: str
    feature_summary: dict[str, float]


@dataclass(slots=True)
class _RuleEvaluation:
    matched: bool
    confidence: float


def summarize_network_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "row_count": 0.0,
            "mean_flow_pkts_per_sec": 0.0,
            "p95_flow_pkts_per_sec": 0.0,
            "mean_flow_bytes_per_sec": 0.0,
            "p95_flow_bytes_per_sec": 0.0,
            "short_flow_ratio": 0.0,
            "long_duration_ratio": 0.0,
            "one_sided_ratio": 0.0,
            "low_payload_ratio": 0.0,
            "tiny_response_ratio": 0.0,
            "fwd_dominant_ratio": 0.0,
            "rst_or_ack_ratio": 0.0,
            "syn_or_rst_ratio": 0.0,
            "ack_or_psh_ratio": 0.0,
            "high_pps_ratio": 0.0,
            "outbound_to_inbound_bytes_ratio": 0.0,
        }

    flow_pkts = _array(rows, "Flow Pkts/s")
    flow_bytes = _array(rows, "Flow Byts/s")
    duration = _array(rows, "Flow Duration")
    fwd_pkts = _array(rows, "Tot Fwd Pkts")
    bwd_pkts = _array(rows, "Tot Bwd Pkts")
    fwd_bytes = _array(rows, "TotLen Fwd Pkts")
    bwd_bytes = _array(rows, "TotLen Bwd Pkts")
    syn = _array(rows, "SYN Flag Cnt")
    ack = _array(rows, "ACK Flag Cnt")
    rst = _array(rows, "RST Flag Cnt")
    psh = _array(rows, "PSH Flag Cnt")

    total_bytes = fwd_bytes + np.clip(bwd_bytes, a_min=0.0, a_max=None)
    safe_bwd_total = max(float(np.clip(bwd_bytes, a_min=0.0, a_max=None).sum()), 1.0)

    summary = {
        "row_count": float(len(rows)),
        "mean_flow_pkts_per_sec": _mean(flow_pkts),
        "p95_flow_pkts_per_sec": _percentile(flow_pkts, 95),
        "mean_flow_bytes_per_sec": _mean(flow_bytes),
        "p95_flow_bytes_per_sec": _percentile(flow_bytes, 95),
        "short_flow_ratio": _ratio(duration < 100_000.0),
        "long_duration_ratio": _ratio(duration > 10_000_000.0),
        "one_sided_ratio": _ratio((bwd_pkts <= 0.0) | (bwd_bytes <= 1.0)),
        "low_payload_ratio": _ratio(total_bytes <= 100.0),
        "tiny_response_ratio": _ratio(bwd_bytes <= 1.0),
        "fwd_dominant_ratio": _ratio(fwd_pkts > bwd_pkts),
        "rst_or_ack_ratio": _ratio((rst > 0.0) | (ack > 0.0)),
        "syn_or_rst_ratio": _ratio((syn > 0.0) | (rst > 0.0)),
        "ack_or_psh_ratio": _ratio((ack > 0.0) | (psh > 0.0)),
        "high_pps_ratio": _ratio(flow_pkts > 1_000.0),
        "outbound_to_inbound_bytes_ratio": round(float(fwd_bytes.sum()) / safe_bwd_total, 4),
    }
    return {key: round(float(value), 4) for key, value in summary.items()}


def classify_network_anomaly(
    *,
    rows: list[dict[str, Any]],
    anomaly_score: float,
) -> NetworkClassificationResult:
    summary = summarize_network_rows(rows)
    score = _safe_score(anomaly_score)

    priority_rules = [
        ("DDoS_Flood", _rule_ddos_flood),
        ("Data_Exfiltration", _rule_data_exfiltration),
        ("Recon_Scan", _rule_recon_scan),
        ("Brute_Force_Abuse", _rule_bruteforce_abuse),
        ("Botnet_C2_Beaconing", _rule_botnet_c2),
    ]

    evaluations: dict[str, _RuleEvaluation] = {}
    for rule_name, rule_fn in priority_rules:
        matched, confidence = rule_fn(summary, score)
        evaluations[rule_name] = _RuleEvaluation(matched=matched, confidence=confidence)

    matched_rules = [name for name, result in evaluations.items() if result.matched]
    selected_type = next((name for name, _ in priority_rules if evaluations[name].matched), "Unknown_Network_Anomaly")

    if selected_type == "Unknown_Network_Anomaly":
        evaluations[selected_type] = _RuleEvaluation(matched=True, confidence=_unknown_confidence(score))
        matched_rules = [selected_type]

    severity = _resolve_severity(selected_type, summary, score)
    return NetworkClassificationResult(
        anomaly_type=selected_type,
        severity=severity,
        confidence=round(float(evaluations[selected_type].confidence), 3),
        matched_rules=matched_rules,
        description=_description_for(selected_type),
        feature_summary=summary,
    )


def _rule_ddos_flood(summary: Mapping[str, float], score: float) -> tuple[bool, float]:
    matched = (
        summary["p95_flow_pkts_per_sec"] >= 5_000.0
        and summary["high_pps_ratio"] >= 0.25
        and summary["one_sided_ratio"] >= 0.25
        and summary["short_flow_ratio"] >= 0.40
    )
    confidence = 0.0 if not matched else min(1.0, 0.62 + 0.20 * score + 0.10 * summary["high_pps_ratio"])
    return matched, round(confidence, 3)


def _rule_data_exfiltration(summary: Mapping[str, float], score: float) -> tuple[bool, float]:
    matched = (
        summary["outbound_to_inbound_bytes_ratio"] >= 3.0
        and summary["fwd_dominant_ratio"] >= 0.60
        and summary["mean_flow_bytes_per_sec"] >= 10_000.0
    )
    confidence = 0.0 if not matched else min(1.0, 0.58 + 0.22 * score + 0.05 * summary["fwd_dominant_ratio"])
    return matched, round(confidence, 3)


def _rule_recon_scan(summary: Mapping[str, float], score: float) -> tuple[bool, float]:
    matched = (
        summary["one_sided_ratio"] >= 0.30
        and summary["low_payload_ratio"] >= 0.30
        and summary["rst_or_ack_ratio"] >= 0.60
    )
    confidence = 0.0 if not matched else min(
        1.0,
        0.57 + 0.20 * score + 0.10 * summary["one_sided_ratio"] + 0.08 * summary["low_payload_ratio"],
    )
    return matched, round(confidence, 3)


def _rule_bruteforce_abuse(summary: Mapping[str, float], score: float) -> tuple[bool, float]:
    matched = (
        summary["short_flow_ratio"] >= 0.55
        and summary["syn_or_rst_ratio"] >= 0.45
        and summary["low_payload_ratio"] >= 0.25
    )
    confidence = 0.0 if not matched else min(1.0, 0.55 + 0.22 * score + 0.08 * summary["syn_or_rst_ratio"])
    return matched, round(confidence, 3)


def _rule_botnet_c2(summary: Mapping[str, float], score: float) -> tuple[bool, float]:
    matched = (
        summary["long_duration_ratio"] >= 0.30
        and summary["low_payload_ratio"] >= 0.25
        and summary["high_pps_ratio"] <= 0.15
        and summary["ack_or_psh_ratio"] >= 0.60
    )
    confidence = 0.0 if not matched else min(1.0, 0.54 + 0.20 * score + 0.08 * summary["long_duration_ratio"])
    return matched, round(confidence, 3)


def _unknown_confidence(score: float) -> float:
    return round(min(1.0, 0.35 + 0.35 * score), 3)


def _resolve_severity(anomaly_type: str, summary: Mapping[str, float], score: float) -> str:
    if anomaly_type == "DDoS_Flood":
        return "Critical"
    if anomaly_type == "Data_Exfiltration":
        return "Critical" if summary["outbound_to_inbound_bytes_ratio"] >= 5.0 else "High"
    if anomaly_type in {"Recon_Scan", "Brute_Force_Abuse"}:
        return "High" if score >= 0.45 else "Medium"
    if anomaly_type == "Botnet_C2_Beaconing":
        return "High" if summary["long_duration_ratio"] >= 0.45 else "Medium"
    return "Medium"


def _description_for(anomaly_type: str) -> str:
    descriptions = {
        "DDoS_Flood": "High-rate one-sided traffic patterns suggest a denial-of-service or flooding attempt.",
        "Data_Exfiltration": "Outbound-heavy anomalous flows suggest sensitive data may be leaving the environment.",
        "Recon_Scan": "Repeated low-payload probing flows suggest reconnaissance, scanning, or pre-attack discovery.",
        "Brute_Force_Abuse": "Short repetitive connection attempts suggest credential abuse or brute-force behavior.",
        "Botnet_C2_Beaconing": "Low-volume but suspicious long-lived control traffic suggests botnet command-and-control beaconing.",
        "Unknown_Network_Anomaly": "Traffic is anomalous but does not match a stronger network rule signature.",
    }
    return descriptions[anomaly_type]


def _array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [float(_to_float(row.get(key, 0.0))) for row in rows]
    return np.asarray(values, dtype=np.float32)


def _to_float(value: Any) -> float:
    try:
        number = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return float(number)


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size > 0 else 0.0


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size > 0 else 0.0


def _ratio(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size > 0 else 0.0


def _safe_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))
