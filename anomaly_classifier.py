"""Rule-based multi-class anomaly classifier for HDFS event sequences."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import uuid
from typing import Any, Sequence, TypedDict


ERROR_EVENTS_FOR_CASCADE = ("E5", "E6", "E7", "E15", "E20", "E21", "E22", "E24")


class ClassificationResult(TypedDict):
    """JSON-serializable result for one classified anomaly sequence."""

    anomaly_type: str
    severity: str
    confidence: float
    matched_rules: list[str]
    description: str


class TimelineEntry(TypedDict):
    """One timeline entry built from a model prediction record."""

    timestamp: str
    block_id: str
    anomaly_type: str
    severity: str
    score: float


class StatisticsResult(TypedDict):
    """Aggregate statistics over multiple model prediction records."""

    counts_per_anomaly_type: dict[str, int]
    average_score_per_type: dict[str, float]
    most_common_severity: str
    timeline: list[TimelineEntry]


class PredictResult(TypedDict, total=False):
    """Expected input schema from predict.py outputs (or compatible dicts)."""

    block_id: str
    score: float
    anomaly_score: float
    event_sequence: list[str] | str
    event_names: list[str]
    timestamp: str
    anomaly_type: str
    severity: str


@dataclass(slots=True)
class _RuleEvaluation:
    """Internal holder for one rule's match decision and confidence."""

    matched: bool
    confidence: float


def _normalize_events(event_names: Sequence[str]) -> list[str]:
    """Normalize event labels into canonical uppercase IDs like E22."""

    normalized: list[str] = []
    for item in event_names:
        event = str(item).strip().upper()
        if event:
            normalized.append(event)
    return normalized


def _safe_score(anomaly_score: float) -> float:
    """Clamp model score to [0.0, 1.0] to keep all outputs stable."""

    return max(0.0, min(1.0, float(anomaly_score)))


def _rule_cascading_failure(event_counts: Counter[str], anomaly_score: float) -> tuple[bool, float]:
    """Detects multi-symptom outages that usually indicate system-wide impact."""
    # Real-world meaning: many independent error signatures happening together is a cascade.
    matched_events = sum(1 for event in ERROR_EVENTS_FOR_CASCADE if event_counts[event] > 0)
    # Threshold 3: at least three different critical error families suggests propagation, not noise.
    matched = matched_events >= 3
    confidence = 0.0 if not matched else min(1.0, 0.60 + 0.10 * matched_events + 0.20 * _safe_score(anomaly_score))
    return matched, round(confidence, 3)


def _rule_node_failure(event_counts: Counter[str], anomaly_score: float) -> tuple[bool, float]:
    """Detects DataNode-level failure from death signal or repeated receive interruptions."""
    # Real-world meaning: dead node signal (E24) or repeated interrupt+receive faults indicates node instability.
    repeated_interrupts = event_counts["E5"] >= 3 and event_counts["E7"] > 0  # Threshold 3: repeated failures over transient blips.
    matched = event_counts["E24"] > 0 or repeated_interrupts
    confidence = 0.0 if not matched else min(1.0, 0.72 + 0.18 * _safe_score(anomaly_score))
    return matched, round(confidence, 3)


def _rule_data_corruption(event_counts: Counter[str], anomaly_score: float) -> tuple[bool, float]:
    """Detects corruption indicators from checksum/packet/bad-block errors."""
    # Real-world meaning: checksum mismatch, corrupt packet, or bad block report are direct corruption evidence.
    matched = event_counts["E22"] > 0 or event_counts["E16"] > 0 or event_counts["E28"] > 0
    signal_count = event_counts["E22"] + event_counts["E16"] + event_counts["E28"]
    confidence = 0.0 if not matched else min(1.0, 0.58 + 0.10 * signal_count + 0.20 * _safe_score(anomaly_score))
    return matched, round(confidence, 3)


def _rule_network_connection_error(event_counts: Counter[str], anomaly_score: float) -> tuple[bool, float]:
    """Detects network/write-path instability across transfer and receive stages."""
    # Real-world meaning: write IO errors plus receive exceptions and interruptions strongly indicate network path issues.
    matched = event_counts["E6"] > 0 and event_counts["E7"] > 0 and event_counts["E5"] > 0
    repeated_errors = event_counts["E6"] + event_counts["E7"]  # Threshold 4 used below to flag persistent link failure.
    confidence = 0.0 if not matched else min(1.0, 0.56 + 0.07 * repeated_errors + 0.18 * _safe_score(anomaly_score))
    return matched, round(confidence, 3)


def _rule_replication_failure(event_counts: Counter[str], anomaly_score: float) -> tuple[bool, float]:
    """Detects failed recovery/replication actions after block issues."""
    # Real-world meaning: timeout or transfer-sending failures show replication pipeline cannot heal data placement.
    matched = event_counts["E15"] > 0 or event_counts["E21"] > 0 or event_counts["E20"] > 0
    signal_count = event_counts["E15"] + event_counts["E21"] + event_counts["E20"]
    confidence = 0.0 if not matched else min(1.0, 0.55 + 0.10 * signal_count + 0.20 * _safe_score(anomaly_score))
    return matched, round(confidence, 3)


def _rule_storage_write_failure(event_counts: Counter[str], anomaly_score: float) -> tuple[bool, float]:
    """Detects local storage write/delete failures that disrupt block persistence."""
    # Real-world meaning: write errors with abandoned creation or delete errors indicate storage layer problems.
    matched = (event_counts["E6"] > 0 and event_counts["E18"] > 0) or event_counts["E19"] > 0
    confidence = 0.0 if not matched else min(1.0, 0.52 + 0.15 * (event_counts["E19"] > 0) + 0.18 * _safe_score(anomaly_score))
    return matched, round(confidence, 3)


def _rule_packetresponder_crash(event_counts: Counter[str], anomaly_score: float) -> tuple[bool, float]:
    """Detects pure PacketResponder interruption without lower-level IO/receive evidence."""
    # Real-world meaning: repeated PacketResponder interruptions alone often point to responder thread crash/restart.
    has_pure_interruptions = event_counts["E5"] >= 2 and event_counts["E6"] == 0 and event_counts["E7"] == 0
    # Threshold 2: one interrupt can be noise, two+ indicates recurring responder instability.
    confidence = 0.0 if not has_pure_interruptions else min(1.0, 0.50 + 0.08 * event_counts["E5"] + 0.18 * _safe_score(anomaly_score))
    return has_pure_interruptions, round(confidence, 3)


def _rule_pipeline_failure(event_counts: Counter[str], anomaly_score: float) -> tuple[bool, float]:
    """Detects pipeline state problems during delivery/update stages."""
    # Real-world meaning: explicit pipeline failure or status updates alongside transport errors indicates pipeline breakage.
    matched = event_counts["E23"] > 0 or (event_counts["E27"] > 0 and (event_counts["E6"] > 0 or event_counts["E7"] > 0))
    confidence = 0.0 if not matched else min(1.0, 0.50 + 0.10 * (event_counts["E23"] > 0) + 0.18 * _safe_score(anomaly_score))
    return matched, round(confidence, 3)


def _rule_unknown_system_anomaly(_: Counter[str], anomaly_score: float) -> tuple[bool, float]:
    """Fallback rule when no known signature family is matched."""
    # Real-world meaning: model flagged anomaly but signature does not map cleanly to known failure taxonomy.
    confidence = 0.35 + 0.30 * _safe_score(anomaly_score)  # Lower base confidence due to missing explicit rule signal.
    return True, round(min(1.0, confidence), 3)


def _resolve_severity(anomaly_type: str, event_counts: Counter[str], anomaly_score: float) -> str:
    """Resolve severity levels using class-specific thresholds from domain rules."""

    score = _safe_score(anomaly_score)
    if anomaly_type == "Cascading_Failure":
        return "Critical"
    if anomaly_type == "Node_Failure":
        return "Critical" if event_counts["E24"] > 0 else "High"
    if anomaly_type == "Data_Corruption":
        return "High" if score > 0.95 else "Medium"  # Threshold 0.95: near-certain model signal means urgent corruption risk.
    if anomaly_type == "Network_Connection_Error":
        return "High" if (event_counts["E6"] + event_counts["E7"]) >= 4 else "Medium"  # Threshold 4: repeated transport errors imply persistent connectivity fault.
    if anomaly_type == "Replication_Failure":
        return "High"
    if anomaly_type == "Storage_Write_Failure":
        return "High" if event_counts["E9"] > 0 else "Medium"  # E9 implies recovery was already attempted, so impact is higher.
    if anomaly_type in {"PacketResponder_Crash", "Pipeline_Failure"}:
        return "Medium"
    return "Medium"


def _description_for(anomaly_type: str) -> str:
    """Return one-sentence plain-English description for each anomaly class."""

    descriptions = {
        "Cascading_Failure": "Multiple independent failure signals indicate a cascading cluster-wide incident.",
        "Node_Failure": "A DataNode appears unhealthy or unreachable, causing block operations to fail.",
        "Data_Corruption": "Corruption indicators suggest block contents or packets may be damaged.",
        "Network_Connection_Error": "Transfer and receive errors point to unstable DataNode network connectivity.",
        "Replication_Failure": "Replication mechanisms are failing to restore healthy block redundancy.",
        "Storage_Write_Failure": "Block persistence operations are failing at the storage write/delete layer.",
        "PacketResponder_Crash": "PacketResponder interruptions suggest responder thread/process instability.",
        "Pipeline_Failure": "The HDFS write pipeline shows delivery/state failures during transfer.",
        "Unknown_System_Anomaly": "No known error signature matched, but behavior remains anomalous.",
    }
    return descriptions[anomaly_type]


def classify_anomaly(event_names: list[str], anomaly_score: float) -> ClassificationResult:
    """Classify an anomalous sequence into a specific system anomaly category."""

    normalized_events = _normalize_events(event_names)
    counts = Counter(normalized_events)

    priority_rules: list[tuple[str, Any]] = [
        ("Cascading_Failure", _rule_cascading_failure),
        ("Node_Failure", _rule_node_failure),
        ("Data_Corruption", _rule_data_corruption),
        ("Network_Connection_Error", _rule_network_connection_error),
        ("Replication_Failure", _rule_replication_failure),
        ("Storage_Write_Failure", _rule_storage_write_failure),
        ("PacketResponder_Crash", _rule_packetresponder_crash),
        ("Pipeline_Failure", _rule_pipeline_failure),
    ]

    evaluations: dict[str, _RuleEvaluation] = {}
    for rule_name, rule_fn in priority_rules:
        matched, confidence = rule_fn(counts, anomaly_score)
        evaluations[rule_name] = _RuleEvaluation(matched=matched, confidence=confidence)

    matched_rules = [name for name, result in evaluations.items() if result.matched]
    selected_type = next((name for name in [r[0] for r in priority_rules] if evaluations[name].matched), "Unknown_System_Anomaly")

    if selected_type == "Unknown_System_Anomaly":
        unknown_match, unknown_conf = _rule_unknown_system_anomaly(counts, anomaly_score)
        evaluations[selected_type] = _RuleEvaluation(matched=unknown_match, confidence=unknown_conf)
        matched_rules = ["Unknown_System_Anomaly"]

    severity = _resolve_severity(selected_type, counts, anomaly_score)
    confidence = evaluations[selected_type].confidence

    return ClassificationResult(
        anomaly_type=selected_type,
        severity=severity,
        confidence=round(confidence, 3),
        matched_rules=matched_rules,
        description=_description_for(selected_type),
    )


def _extract_events(raw_result: PredictResult) -> list[str]:
    """Extract event names from either `event_names` or `event_sequence` fields."""

    if isinstance(raw_result.get("event_names"), list):
        return [str(item) for item in raw_result["event_names"]]
    event_sequence = raw_result.get("event_sequence", [])
    if isinstance(event_sequence, str):
        return [token for token in event_sequence.replace(",", " ").split() if token]
    if isinstance(event_sequence, list):
        return [str(item) for item in event_sequence]
    return []


def _extract_score(raw_result: PredictResult) -> float:
    """Extract anomaly score from supported prediction keys."""

    if "score" in raw_result:
        return _safe_score(float(raw_result["score"]))
    if "anomaly_score" in raw_result:
        return _safe_score(float(raw_result["anomaly_score"]))
    return 0.0


def get_anomaly_statistics(results_list: list[dict[str, Any]]) -> StatisticsResult:
    """Compute anomaly-type counts, average scores, severity mode, and optional timeline."""

    counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    score_totals: dict[str, float] = defaultdict(float)
    timeline: list[TimelineEntry] = []

    for item in results_list:
        result: PredictResult = dict(item)
        events = _extract_events(result)
        score = _extract_score(result)
        classification = classify_anomaly(events, score)
        anomaly_type = str(result.get("anomaly_type", classification["anomaly_type"]))
        severity = str(result.get("severity", classification["severity"]))

        counts[anomaly_type] += 1
        severity_counts[severity] += 1
        score_totals[anomaly_type] += score

        if "timestamp" in result:
            timeline.append(
                TimelineEntry(
                    timestamp=str(result["timestamp"]),
                    block_id=str(result.get("block_id", "unknown")),
                    anomaly_type=anomaly_type,
                    severity=severity,
                    score=round(score, 4),
                )
            )

    averages = {kind: round(score_totals[kind] / counts[kind], 4) for kind in counts}
    most_common_severity = severity_counts.most_common(1)[0][0] if severity_counts else "Unknown"
    timeline.sort(key=lambda entry: entry["timestamp"])

    return StatisticsResult(
        counts_per_anomaly_type=dict(counts),
        average_score_per_type=averages,
        most_common_severity=most_common_severity,
        timeline=timeline,
    )


def _recommended_action(anomaly_type: str) -> str:
    """Map anomaly type to one concise operational recommendation."""

    if anomaly_type == "Data_Corruption":
        return "Trigger immediate block re-verification and re-replication"
    if anomaly_type == "Node_Failure":
        return "Isolate affected DataNode and redistribute its blocks"
    if anomaly_type.startswith("Network_"):
        return "Check network connectivity between DataNodes"
    if anomaly_type.startswith("Replication_"):
        return "Force re-replication of affected blocks"
    if anomaly_type.startswith("Cascading_"):
        return "Alert on-call team immediately, check cluster health dashboard"
    return "Monitor for recurrence, log for review"


def format_alert(classification_result: ClassificationResult, block_id: str) -> dict[str, Any]:
    """Build a clean JSON-serializable alert payload for downstream systems."""

    return {
        "alert_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "block_id": str(block_id),
        "anomaly_type": classification_result["anomaly_type"],
        "severity": classification_result["severity"],
        "confidence": classification_result["confidence"],
        "description": classification_result["description"],
        "matched_rules": list(classification_result["matched_rules"]),
        "recommended_action": _recommended_action(classification_result["anomaly_type"]),
    }


def _run_demo() -> list[dict[str, Any]]:
    """Run standalone examples on five fixed sequences and return formatted alerts."""

    examples: dict[str, tuple[list[str], float, str]] = {
        "Normal": (["E1", "E2", "E3", "E4", "E1", "E2", "E4"], 0.08, "blk-normal-001"),
        "Corruption": (["E5", "E22", "E6", "E7", "E9", "E11", "E14"], 0.97, "blk-corrupt-001"),
        "NodeDown": (["E5", "E5", "E5", "E7", "E24", "E9", "E14"], 0.99, "blk-node-001"),
        "Network": (["E3", "E5", "E6", "E7", "E6", "E7", "E6", "E7"], 0.92, "blk-net-001"),
        "Cascading": (["E5", "E6", "E7", "E22", "E24", "E15", "E21", "E9"], 0.995, "blk-cascade-001"),
    }

    outputs: list[dict[str, Any]] = []
    for test_name, (events, score, block_id) in examples.items():
        classification = classify_anomaly(events, score)
        alert = format_alert(classification, block_id)
        alert["test_case"] = test_name
        alert["input_events"] = events
        alert["model_anomaly_score"] = score
        outputs.append(alert)
    return outputs


if __name__ == "__main__":
    for item in _run_demo():
        print(json.dumps(item, indent=2))
