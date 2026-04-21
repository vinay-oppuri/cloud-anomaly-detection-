from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.correlation.compatibility import build_common_embedding
from src.correlation.events import _network_severity, aggregate_network_events
from src.correlation.pipeline import (
    _load_correlation_model,
    _resolve_device,
    build_anomaly_graph,
    derive_clusters,
)
from src.correlation.schema import AnomalyEvent, CorrelationCluster, CorrelationEdge
from src.experts.network_expert.classifier import classify_network_anomaly
from src.experts.network_expert.test import BinaryAnalyzeConfig
from src.experts.network_expert import test as network_test
from src.experts.system_expert.classifier import classify_anomaly
from src.experts.system_expert.drain import extract_block_ids
from src.experts.system_expert.parser import extract_message
from src.experts.system_expert.service import (
    DEFAULT_CACHE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_PATH,
    EVENT_ID_PATTERN,
    HDFS_TEMPLATE_MAP,
    SystemAnomalyService,
    SystemServiceConfig,
    _encode_event_tokens,
)


DEFAULT_NETWORK_MODEL_PATH = Path("models/network_expert.pth")
DEFAULT_NETWORK_PREPROCESSED_DIR = Path("data/processed")
DEFAULT_NETWORK_METRICS_PATH = Path("models/network_meta.json")
DEFAULT_SYSTEM_METRICS_PATH = Path("models/system_expert_metrics.json")
DEFAULT_CORRELATION_MODEL_PATH = Path("models/correlation_attention.pth")

_BRACKETED_TIMESTAMP_RE = re.compile(r"\[(?P<ts>[^\]]+)\]")
_ISO_TIMESTAMP_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[T ][0-9:\.]+(?:Z|[+-]\d{2}:\d{2})?)"
)
_COMPACT_TIMESTAMP_RE = re.compile(r"(?P<ts>\d{6}\s+\d{6})")
_TIMESTAMP_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S,%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%y%m%d %H%M%S",
)


@dataclass(slots=True)
class RealWorldAnalyzerConfig:
    device: str = "cuda"
    network_model_path: Path = DEFAULT_NETWORK_MODEL_PATH
    network_preprocessed_dir: Path = DEFAULT_NETWORK_PREPROCESSED_DIR
    network_metrics_path: Path = DEFAULT_NETWORK_METRICS_PATH
    system_processed_path: Path = DEFAULT_PROCESSED_PATH
    system_cache_path: Path = DEFAULT_CACHE_PATH
    system_model_path: Path = DEFAULT_MODEL_PATH
    system_metrics_path: Path = DEFAULT_SYSTEM_METRICS_PATH
    correlation_model_path: Path | None = DEFAULT_CORRELATION_MODEL_PATH
    batch_size: int = 512
    window_step: int = 1
    max_report_items: int = 10
    temporal_window_minutes: int = 20
    edge_threshold: float = 0.62
    use_correlation_model: bool = True
    use_gemini: bool = False
    gemini_model: str = "gemini-2.5-flash"
    session_gap_seconds: int = 90


@dataclass(slots=True)
class _NetworkRuntime:
    model: torch.nn.Module
    class_names: list[str]
    feature_cols: list[str]
    scaler: Any
    seq_len: int
    threshold: float
    threshold_source: str
    device: torch.device


@dataclass(slots=True)
class _SystemGroup:
    group_id: str
    group_type: str
    event_tokens: list[str] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    line_count: int = 0
    explicit_event_count: int = 0
    inferred_event_count: int = 0

    @property
    def start_timestamp(self) -> datetime | None:
        return min(self.timestamps) if self.timestamps else None

    @property
    def end_timestamp(self) -> datetime | None:
        return max(self.timestamps) if self.timestamps else None


class CrossLayerRealWorldAnalyzer:
    def __init__(self, config: RealWorldAnalyzerConfig | None = None) -> None:
        self.config = config or RealWorldAnalyzerConfig()
        self.device = _resolve_device(self.config.device)
        self._network_runtime: _NetworkRuntime | None = None
        self._system_service: SystemAnomalyService | None = None
        self._system_threshold: float | None = None
        self._correlation_model = None

    def analyze_texts(
        self,
        *,
        network_log_text: str,
        system_log_text: str,
        incident_name: str,
        use_correlation_model: bool | None = None,
        use_gemini: bool | None = None,
        gemini_model: str | None = None,
        temporal_window_minutes: int | None = None,
        edge_threshold: float | None = None,
        align_on_relative_start: bool = False,
    ) -> dict[str, Any]:
        network_events, network_result = self._analyze_network_text(
            raw_text=network_log_text,
            incident_name=incident_name,
        )
        system_events, system_result = self._analyze_system_text(
            raw_text=system_log_text,
            incident_name=incident_name,
        )

        alignment_note: dict[str, Any] | None = None
        if align_on_relative_start and network_events and system_events:
            alignment_note = _align_relative_start(system_events, network_events)

        runtime_use_correlation = self.config.use_correlation_model if use_correlation_model is None else bool(use_correlation_model)
        runtime_use_gemini = self.config.use_gemini if use_gemini is None else bool(use_gemini)
        runtime_gemini_model = gemini_model or self.config.gemini_model
        runtime_temporal_window = int(temporal_window_minutes or self.config.temporal_window_minutes)
        runtime_edge_threshold = float(edge_threshold or self.config.edge_threshold)

        all_events = sorted([*network_events, *system_events], key=lambda event: event.timestamp)
        correlation_model = self._get_correlation_model() if runtime_use_correlation else None
        graph, edges = build_anomaly_graph(
            events=all_events,
            correlation_model=correlation_model,
            device=self.device,
            temporal_window_minutes=runtime_temporal_window,
            edge_threshold=runtime_edge_threshold,
        )
        used_correlation_model = bool(correlation_model is not None)
        if used_correlation_model and network_events and system_events and _count_cross_layer_edges(edges) == 0:
            fallback_graph, fallback_edges = build_anomaly_graph(
                events=all_events,
                correlation_model=None,
                device=self.device,
                temporal_window_minutes=runtime_temporal_window,
                edge_threshold=runtime_edge_threshold,
            )
            if _count_cross_layer_edges(fallback_edges) > 0:
                graph, edges = fallback_graph, fallback_edges
                used_correlation_model = False
        clusters = derive_clusters(
            graph,
            edges,
            use_gemini=runtime_use_gemini,
            gemini_model=runtime_gemini_model,
        )

        return {
            "task": "cross_layer_realworld_analyze",
            "incident_name": incident_name,
            "config": {
                "device": str(self.device),
                "requested_correlation_model": bool(correlation_model is not None),
                "used_correlation_model": used_correlation_model,
                "use_gemini": runtime_use_gemini,
                "gemini_model": runtime_gemini_model,
                "temporal_window_minutes": runtime_temporal_window,
                "edge_threshold": runtime_edge_threshold,
                "align_on_relative_start": bool(align_on_relative_start),
            },
            "network_detection": network_result,
            "system_detection": system_result,
            "correlation": {
                "summary": _summarize_realworld_correlation(
                    events=all_events,
                    edges=edges,
                    clusters=clusters,
                ),
                "alignment_note": alignment_note,
                "events": [_serialize_event(event) for event in all_events],
                "edges": [edge.to_dict() for edge in edges],
                "clusters": [cluster.to_dict() for cluster in clusters],
                "case_studies": [cluster.to_dict() for cluster in clusters[:2]],
            },
        }

    def analyze_files(
        self,
        *,
        network_log_path: Path,
        system_log_path: Path,
        incident_name: str,
        use_correlation_model: bool | None = None,
        use_gemini: bool | None = None,
        gemini_model: str | None = None,
        temporal_window_minutes: int | None = None,
        edge_threshold: float | None = None,
        align_on_relative_start: bool = False,
    ) -> dict[str, Any]:
        if not network_log_path.exists():
            raise FileNotFoundError(f"Network raw log file not found: {network_log_path}")
        if not system_log_path.exists():
            raise FileNotFoundError(f"System raw log file not found: {system_log_path}")
        return self.analyze_texts(
            network_log_text=network_log_path.read_text(encoding="utf-8", errors="ignore"),
            system_log_text=system_log_path.read_text(encoding="utf-8", errors="ignore"),
            incident_name=incident_name,
            use_correlation_model=use_correlation_model,
            use_gemini=use_gemini,
            gemini_model=gemini_model,
            temporal_window_minutes=temporal_window_minutes,
            edge_threshold=edge_threshold,
            align_on_relative_start=align_on_relative_start,
        )

    def _analyze_network_text(
        self,
        *,
        raw_text: str,
        incident_name: str,
    ) -> tuple[list[AnomalyEvent], dict[str, Any]]:
        runtime = self._get_network_runtime()
        analysis_config = BinaryAnalyzeConfig(
            model_path=self.config.network_model_path,
            preprocessed_dir=self.config.network_preprocessed_dir,
            metrics_path=self.config.network_metrics_path,
            dataset_split=None,
            input_file=None,
            log_file=None,
            log_text=raw_text,
            input_format="auto",
            interactive=False,
            threshold=None,
            device=self.config.device,
            batch_size=self.config.batch_size,
            window_step=self.config.window_step,
            max_report_items=self.config.max_report_items,
        )
        rows, row_timestamps = _parse_network_rows_with_timestamps(
            raw_text=raw_text,
            feature_cols=runtime.feature_cols,
        )
        if not rows:
            raise ValueError(
                "No usable network rows were parsed from the raw input. "
                "Provide key=value style flow lines with numeric CICIDS-like features."
            )

        scaled_rows = network_test._preprocess_input_rows(
            rows=rows,
            feature_cols=runtime.feature_cols,
            scaler=runtime.scaler,
        )
        sequences, sequence_starts = network_test._build_sequences(
            scaled_rows=scaled_rows,
            seq_len=runtime.seq_len,
            step=self.config.window_step,
        )

        benign_index = runtime.class_names.index("Benign") if "Benign" in runtime.class_names else 0
        all_scores: list[float] = []
        raw_event_rows: list[dict[str, Any]] = []
        window_reports: list[dict[str, Any]] = []

        with torch.inference_mode():
            for batch_start in range(0, int(sequences.shape[0]), int(self.config.batch_size)):
                batch_end = min(batch_start + int(self.config.batch_size), int(sequences.shape[0]))
                batch = torch.from_numpy(sequences[batch_start:batch_end]).to(
                    device=runtime.device,
                    dtype=torch.float32,
                )
                logits = runtime.model(batch)
                embeddings = runtime.model.forward_features(batch)
                probabilities = torch.softmax(logits, dim=-1)
                anomaly_scores = 1.0 - probabilities[:, benign_index]
                predicted_indices = torch.argmax(probabilities, dim=-1)

                for local_index in range(batch_end - batch_start):
                    score = float(anomaly_scores[local_index].item())
                    all_scores.append(score)

                    row_index = batch_start + local_index
                    start_row = int(sequence_starts[row_index])
                    end_row = min(start_row + runtime.seq_len - 1, len(rows) - 1)
                    window_rows = rows[start_row : end_row + 1]
                    probs = probabilities[local_index].detach().cpu().numpy()

                    pred_idx = int(predicted_indices[local_index].item())
                    pred_label = str(runtime.class_names[pred_idx])
                    if pred_label == "Benign" and score >= runtime.threshold and len(runtime.class_names) > 1:
                        probs_without_benign = probs.copy()
                        probs_without_benign[benign_index] = -1.0
                        pred_idx = int(np.argmax(probs_without_benign))
                        pred_label = str(runtime.class_names[pred_idx])

                    confidence = float(max(0.0, probs[pred_idx]))
                    timestamp = row_timestamps[end_row]

                    if score < runtime.threshold:
                        continue

                    severity = _network_severity(pred_label, score)
                    embedding = build_common_embedding(
                        raw_embedding=embeddings[local_index].detach().cpu().numpy(),
                        label=pred_label,
                        event_type="network",
                        anomaly_score=score,
                        severity=severity,
                    )

                    raw_event_rows.append(
                        {
                            "row_index": row_index,
                            "timestamp": timestamp,
                            "label": pred_label,
                            "anomaly_score": score,
                            "confidence": confidence,
                            "embedding": embedding,
                            "severity": severity,
                            "source_ref": f"{incident_name}:network:{start_row}-{end_row}",
                            "metadata": {
                                "source_file": incident_name,
                                "start_index": start_row,
                                "end_index": end_row,
                                "window_row_count": len(window_rows),
                            },
                        }
                    )
                    window_reports.append(
                        {
                            "start_row": start_row,
                            "end_row": end_row,
                            "timestamp": timestamp.isoformat(),
                            "label": pred_label,
                            "anomaly_score": score,
                            "confidence": confidence,
                            "rows": window_rows,
                        }
                    )

        events = aggregate_network_events(raw_event_rows)
        label_counts = Counter(event.label for event in events)
        max_score = max(all_scores, default=0.0)
        final_window_score = all_scores[-1] if all_scores else 0.0
        final_window_label = "Anomaly" if final_window_score >= runtime.threshold else "Benign"
        anomaly_window_ratio = float(len(window_reports) / max(1, int(sequences.shape[0])))
        top_window = max(window_reports, key=lambda item: item["anomaly_score"], default=None)
        heuristic_summary: dict[str, Any] | None = None
        if top_window is not None:
            heuristic = classify_network_anomaly(
                rows=list(top_window["rows"]),
                anomaly_score=float(top_window["anomaly_score"]),
            )
            heuristic_summary = {
                "anomaly_type": heuristic["anomaly_type"],
                "severity": heuristic["severity"],
                "confidence": heuristic["confidence"],
                "matched_rules": heuristic["matched_rules"],
                "description": heuristic["description"],
            }
        session_decision_label, session_decision_reason = network_test._resolve_session_decision(
            config=analysis_config,
            final_window_label=final_window_label,
            anomaly_window_ratio=anomaly_window_ratio,
            max_anomaly_score=float(max_score),
            input_rows=rows,
        )
        if session_decision_label == "Anomaly" and not events:
            heuristic = classify_network_anomaly(
                rows=list(rows),
                anomaly_score=max(float(max_score), runtime.threshold),
            )
            heuristic_label = _map_network_rule_label_to_family(str(heuristic["anomaly_type"]))
            fallback_score = max(
                float(max_score),
                float(runtime.threshold),
                float(heuristic["confidence"]),
            )
            fallback_embedding = build_common_embedding(
                raw_embedding=np.asarray(list(heuristic["feature_summary"].values()), dtype=np.float32),
                label=heuristic_label,
                event_type="network",
                anomaly_score=fallback_score,
                severity=_network_severity(heuristic_label, fallback_score),
            )
            fallback_timestamp = row_timestamps[-1]
            events = [
                AnomalyEvent(
                    event_id="network-00001",
                    timestamp=fallback_timestamp,
                    event_type="network",
                    label=heuristic_label,
                    anomaly_score=fallback_score,
                    confidence=float(heuristic["confidence"]),
                    embedding=fallback_embedding,
                    source_ref=f"{incident_name}:network:raw-session",
                    severity=_network_severity(heuristic_label, fallback_score),
                    metadata={
                        "source_file": incident_name,
                        "classification_source": "raw_log_session_rule",
                        "matched_rules": list(heuristic["matched_rules"]),
                        "feature_summary": heuristic["feature_summary"],
                    },
                )
            ]
            label_counts = Counter(event.label for event in events)
            heuristic_summary = {
                "anomaly_type": heuristic["anomaly_type"],
                "severity": heuristic["severity"],
                "confidence": heuristic["confidence"],
                "matched_rules": heuristic["matched_rules"],
                "description": heuristic["description"],
            }

        return events, {
            "anomaly_detected": bool(events),
            "decision_label": session_decision_label,
            "decision_reason": session_decision_reason,
            "num_raw_rows": len(rows),
            "num_sequences": int(sequences.shape[0]),
            "threshold": round(float(runtime.threshold), 6),
            "threshold_source": runtime.threshold_source,
            "max_anomaly_score": round(float(max_score), 6),
            "event_count": len(events),
            "labels": dict(label_counts),
            "heuristic_summary": heuristic_summary,
            "top_events": [_serialize_event(event) for event in _top_events(events, limit=self.config.max_report_items)],
        }

    def _analyze_system_text(
        self,
        *,
        raw_text: str,
        incident_name: str,
    ) -> tuple[list[AnomalyEvent], dict[str, Any]]:
        service = self._get_system_service()
        threshold = self._get_system_threshold()
        coarse_result = service.analyze_log_lines(
            raw_text.splitlines(),
            event_name=f"{incident_name}:coarse",
        )
        groups, extraction = _parse_system_groups(
            raw_text=raw_text,
            session_gap_seconds=self.config.session_gap_seconds,
        )
        if not groups:
            raise ValueError(
                "No HDFS-style event tokens could be extracted from the raw system log. "
                "Provide HDFS logs with block ids or known templates."
            )
        has_block_groups = any(group.group_type == "block" for group in groups)

        encoded_batches: list[torch.Tensor] = []
        group_payloads: list[dict[str, Any]] = []
        for group in groups:
            encoded, unknown_tokens = _encode_event_tokens(
                event_tokens=list(group.event_tokens),
                vocab=service.vocab,
                sequence_length=service.sequence_length,
            )
            encoded_batches.append(encoded)
            group_payloads.append(
                {
                    "group": group,
                    "unknown_tokens": unknown_tokens,
                }
            )

        token_tensor = torch.stack(encoded_batches, dim=0).to(service.device)
        class_names = list(service.class_names)
        normal_index = class_names.index(service.normal_class_name) if service.normal_class_name in class_names else 0

        events: list[AnomalyEvent] = []
        score_values: list[float] = []
        with torch.inference_mode():
            for batch_start in range(0, int(token_tensor.shape[0]), int(self.config.batch_size)):
                batch_end = min(batch_start + int(self.config.batch_size), int(token_tensor.shape[0]))
                batch = token_tensor[batch_start:batch_end]
                logits = service.expert.model(batch)
                embeddings = service.expert.model.forward_features(batch)
                probabilities = torch.softmax(logits, dim=-1)
                anomaly_scores = 1.0 - probabilities[:, normal_index]
                predicted_indices = torch.argmax(probabilities, dim=-1)

                for local_index in range(batch_end - batch_start):
                    payload = group_payloads[batch_start + local_index]
                    group: _SystemGroup = payload["group"]
                    score = float(anomaly_scores[local_index].item())
                    score_values.append(score)
                    pred_idx = int(predicted_indices[local_index].item())
                    pred_class = str(class_names[pred_idx])
                    is_anomaly = bool(score >= threshold or pred_class != service.normal_class_name)
                    if not is_anomaly:
                        continue

                    classified = classify_anomaly(
                        event_names=[token.upper() for token in group.event_tokens],
                        anomaly_score=score,
                    )
                    if not _should_emit_system_group_event(
                        group=group,
                        score=score,
                        classified_label=str(classified["anomaly_type"]),
                        coarse_result=coarse_result,
                        has_block_groups=has_block_groups,
                    ):
                        continue
                    timestamp = group.start_timestamp or _default_base_timestamp()
                    severity = str(classified["severity"])
                    label = str(classified["anomaly_type"])
                    embedding = build_common_embedding(
                        raw_embedding=embeddings[local_index].detach().cpu().numpy(),
                        label=label,
                        event_type="log",
                        anomaly_score=score,
                        severity=severity,
                    )
                    events.append(
                        AnomalyEvent(
                            event_id=f"log-{len(events)+1:05d}",
                            timestamp=timestamp,
                            event_type="log",
                            label=label,
                            anomaly_score=score,
                            confidence=float(probabilities[local_index, pred_idx].item()),
                            embedding=embedding,
                            source_ref=f"{incident_name}:{group.group_id}",
                            severity=severity,
                            metadata={
                                "group_id": group.group_id,
                                "group_type": group.group_type,
                                "line_count": group.line_count,
                                "event_token_count": len(group.event_tokens),
                                "event_tokens_tail": [token.upper() for token in group.event_tokens[-20:]],
                                "predicted_class": pred_class,
                                "matched_rules": list(classified["matched_rules"]),
                                "description": str(classified["description"]),
                                "unknown_event_tokens": list(payload["unknown_tokens"][:20]),
                                "start_timestamp": group.start_timestamp.isoformat() if group.start_timestamp else None,
                                "end_timestamp": group.end_timestamp.isoformat() if group.end_timestamp else None,
                            },
                        )
                    )

        events.sort(key=lambda event: event.timestamp)
        label_counts = Counter(event.label for event in events)
        return events, {
            "anomaly_detected": bool(events),
            "decision_label": "Anomaly" if events else "Benign",
            "num_groups": len(groups),
            "threshold": round(float(threshold), 6),
            "max_anomaly_score": round(float(max(score_values, default=0.0)), 6),
            "event_count": len(events),
            "labels": dict(label_counts),
            "extraction": extraction,
            "coarse_file_detection": {
                "anomaly_detected": bool(coarse_result.get("anomaly_detected", False)),
                "anomaly_type": str(coarse_result.get("anomaly_type", "Unknown")),
                "predicted_label": str(coarse_result.get("predicted_label", "Unknown")),
                "anomaly_score": round(float(coarse_result.get("anomaly_score", 0.0)), 6),
            },
            "top_events": [_serialize_event(event) for event in _top_events(events, limit=self.config.max_report_items)],
        }

    def _get_network_runtime(self) -> _NetworkRuntime:
        if self._network_runtime is not None:
            return self._network_runtime

        raw_mode_config = BinaryAnalyzeConfig(
            model_path=self.config.network_model_path,
            preprocessed_dir=self.config.network_preprocessed_dir,
            metrics_path=self.config.network_metrics_path,
            dataset_split=None,
            input_file=None,
            log_file=None,
            log_text="<runtime>",
            input_format="auto",
            interactive=False,
            threshold=None,
            device=self.config.device,
            batch_size=self.config.batch_size,
            window_step=self.config.window_step,
            max_report_items=self.config.max_report_items,
        )
        (
            model,
            class_names,
            feature_cols,
            scaler,
            seq_len,
            threshold_from_metrics,
            threshold_source_from_metrics,
            device,
        ) = network_test._load_runtime_assets(raw_mode_config)
        threshold, threshold_source = network_test._resolve_runtime_threshold(
            config=raw_mode_config,
            threshold_from_metrics=float(threshold_from_metrics),
            threshold_source_from_metrics=str(threshold_source_from_metrics),
        )
        self._network_runtime = _NetworkRuntime(
            model=model,
            class_names=list(class_names),
            feature_cols=list(feature_cols),
            scaler=scaler,
            seq_len=int(seq_len),
            threshold=float(threshold),
            threshold_source=str(threshold_source),
            device=device,
        )
        return self._network_runtime

    def _get_system_service(self) -> SystemAnomalyService:
        if self._system_service is not None:
            return self._system_service
        self._system_service = SystemAnomalyService.from_config(
            SystemServiceConfig(
                processed_data=self.config.system_processed_path,
                cache_path=self.config.system_cache_path,
                model_path=self.config.system_model_path,
                device=self.config.device,
                use_gemini=False,
                show_workflow_progress=False,
            )
        )
        return self._system_service

    def _get_system_threshold(self) -> float:
        if self._system_threshold is not None:
            return self._system_threshold
        self._system_threshold = _load_system_threshold(self.config.system_metrics_path)
        return self._system_threshold

    def _get_correlation_model(self) -> Any | None:
        if self._correlation_model is not None:
            return self._correlation_model
        if self.config.correlation_model_path is None:
            return None
        self._correlation_model = _load_correlation_model(
            self.config.correlation_model_path,
            device=self.device,
        )
        return self._correlation_model


def _parse_network_rows_with_timestamps(
    *,
    raw_text: str,
    feature_cols: list[str],
) -> tuple[list[dict[str, Any]], list[datetime]]:
    rows: list[dict[str, Any]] = []
    timestamps: list[datetime | None] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed_rows = network_test._parse_raw_text_logs(stripped, feature_cols=feature_cols)
        if not parsed_rows:
            continue
        rows.append(parsed_rows[0])
        timestamps.append(_extract_timestamp(stripped))
    return rows, _ensure_monotonic_timestamps(timestamps)


def _parse_system_groups(
    *,
    raw_text: str,
    session_gap_seconds: int,
) -> tuple[list[_SystemGroup], dict[str, Any]]:
    groups: dict[str, _SystemGroup] = {}
    unmatched_lines = 0
    explicit_event_total = 0
    inferred_event_total = 0
    session_index = 0
    current_session_id: str | None = None
    last_session_timestamp: datetime | None = None

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        timestamp = _extract_timestamp(line)
        message = extract_message(line)
        event_tokens, explicit_count, inferred_count = _extract_system_event_tokens(message)
        if not event_tokens:
            unmatched_lines += 1
            continue

        explicit_event_total += explicit_count
        inferred_event_total += inferred_count
        block_ids = extract_block_ids(message)
        if block_ids:
            target_group_ids = [f"block:{block_id}" for block_id in block_ids]
            group_type = "block"
        else:
            if current_session_id is None:
                session_index += 1
                current_session_id = f"session:{session_index:05d}"
            elif (
                timestamp is not None
                and last_session_timestamp is not None
                and (timestamp - last_session_timestamp).total_seconds() > max(1, int(session_gap_seconds))
            ):
                session_index += 1
                current_session_id = f"session:{session_index:05d}"
            target_group_ids = [current_session_id]
            group_type = "session"
            if timestamp is not None:
                last_session_timestamp = timestamp

        for group_id in target_group_ids:
            group = groups.setdefault(group_id, _SystemGroup(group_id=group_id, group_type=group_type))
            group.event_tokens.extend(event_tokens)
            if timestamp is not None:
                group.timestamps.append(timestamp)
            group.line_count += 1
            group.explicit_event_count += explicit_count
            group.inferred_event_count += inferred_count

    ordered_groups = sorted(
        groups.values(),
        key=lambda item: (item.start_timestamp or _default_base_timestamp(), item.group_id),
    )
    return ordered_groups, {
        "groups_parsed": len(ordered_groups),
        "explicit_event_tokens": explicit_event_total,
        "template_inferred_tokens": inferred_event_total,
        "unmatched_lines": unmatched_lines,
    }


def _extract_system_event_tokens(message: str) -> tuple[list[str], int, int]:
    explicit = [token.lower() for token in EVENT_ID_PATTERN.findall(message)]
    if explicit:
        return explicit, len(explicit), 0
    for pattern, event_id in HDFS_TEMPLATE_MAP:
        if pattern.search(message):
            return [str(event_id).lower()], 0, 1
    return [], 0, 0


def _extract_timestamp(text: str) -> datetime | None:
    candidates: list[str] = []
    bracket_match = _BRACKETED_TIMESTAMP_RE.search(text)
    if bracket_match is not None:
        candidates.append(bracket_match.group("ts").strip())

    iso_match = _ISO_TIMESTAMP_RE.search(text)
    if iso_match is not None:
        candidates.append(iso_match.group("ts").strip())

    compact_match = _COMPACT_TIMESTAMP_RE.search(text)
    if compact_match is not None:
        candidates.append(compact_match.group("ts").strip())

    for candidate in candidates:
        parsed = _parse_timestamp_candidate(candidate)
        if parsed is not None:
            return parsed
    return None


def _parse_timestamp_candidate(value: str) -> datetime | None:
    candidate = str(value).strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            return _to_naive_utc(parsed)
        except ValueError:
            pass
    for fmt in _TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(candidate, fmt)
            return _to_naive_utc(parsed)
        except ValueError:
            continue
    try:
        return _to_naive_utc(datetime.fromisoformat(candidate))
    except ValueError:
        return None


def _to_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _default_base_timestamp() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def _ensure_monotonic_timestamps(values: Sequence[datetime | None]) -> list[datetime]:
    if not values:
        return []
    resolved: list[datetime] = []
    current = next((item for item in values if item is not None), None) or _default_base_timestamp()
    for raw in values:
        candidate = raw if raw is not None else (current + timedelta(seconds=1) if resolved else current)
        if resolved and candidate <= resolved[-1]:
            candidate = resolved[-1] + timedelta(seconds=1)
        resolved.append(candidate)
        current = candidate
    return resolved


def _load_system_threshold(path: Path) -> float:
    if not path.exists():
        return 0.5
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0.5
    validation = payload.get("validation")
    if isinstance(validation, dict):
        raw_best = validation.get("best_threshold")
        if raw_best is not None:
            try:
                return float(raw_best)
            except (TypeError, ValueError):
                pass
    for key in ("best_threshold", "threshold"):
        raw_value = payload.get(key)
        if raw_value is None:
            continue
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            continue
    return 0.5


def _serialize_event(event: AnomalyEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "timestamp": event.timestamp.isoformat(),
        "type": event.event_type,
        "label": event.label,
        "anomaly_score": round(float(event.anomaly_score), 6),
        "confidence": round(float(event.confidence), 6),
        "source_ref": event.source_ref,
        "severity": event.severity,
        "metadata": event.metadata,
    }


def _top_events(events: list[AnomalyEvent], *, limit: int) -> list[AnomalyEvent]:
    return sorted(events, key=lambda event: event.anomaly_score, reverse=True)[: max(1, int(limit))]


def _map_network_rule_label_to_family(rule_label: str) -> str:
    mapping = {
        "DDoS_Flood": "DDoS",
        "Brute_Force_Abuse": "BruteForce",
        "Botnet_C2_Beaconing": "Botnet",
        "Data_Exfiltration": "OtherAttack",
        "Recon_Scan": "OtherAttack",
        "Unknown_Network_Anomaly": "OtherAttack",
    }
    return mapping.get(str(rule_label), "OtherAttack")


def _count_cross_layer_edges(edges: list[CorrelationEdge]) -> int:
    return sum(
        1
        for edge in edges
        if edge.metadata.get("left_type") == "network" and edge.metadata.get("right_type") == "log"
    )


def _should_emit_system_group_event(
    *,
    group: _SystemGroup,
    score: float,
    classified_label: str,
    coarse_result: dict[str, Any],
    has_block_groups: bool,
) -> bool:
    coarse_anomalous = bool(coarse_result.get("anomaly_detected", False))

    if group.group_type == "block":
        if not coarse_anomalous and score < 0.90:
            return False
        return True

    # Session groups are noisy in HDFS because normal lines like completeFile
    # and HA state transitions often have no block id. Keep them only when the
    # whole-file detector already sees an anomaly and the local evidence is strong.
    if has_block_groups:
        if not coarse_anomalous:
            return False
        if len(group.event_tokens) < 8 and score < 0.90:
            return False
        if classified_label in {"Unknown_System_Anomaly", "Pipeline_Failure"} and score < 0.95:
            return False
        return True

    if not coarse_anomalous and score < 0.85:
        return False
    return True


def _align_relative_start(
    system_events: list[AnomalyEvent],
    network_events: list[AnomalyEvent],
) -> dict[str, Any]:
    system_start = min(event.timestamp for event in system_events)
    network_start = min(event.timestamp for event in network_events)
    shift = network_start - system_start
    for event in system_events:
        event.timestamp = event.timestamp + shift
        event.metadata["relative_clock_shift_seconds"] = round(float(shift.total_seconds()), 3)
    return {
        "mode": "relative_start_alignment",
        "shift_seconds": round(float(shift.total_seconds()), 3),
        "network_start": network_start.isoformat(),
        "system_start_before_shift": system_start.isoformat(),
        "system_start_after_shift": min(event.timestamp for event in system_events).isoformat(),
    }


def _summarize_realworld_correlation(
    *,
    events: list[AnomalyEvent],
    edges: list[CorrelationEdge],
    clusters: list[CorrelationCluster],
) -> dict[str, Any]:
    cross_layer_edges = [
        edge for edge in edges
        if edge.metadata.get("left_type") == "network" and edge.metadata.get("right_type") == "log"
    ]
    cluster_sizes = [len(cluster.node_ids) for cluster in clusters]
    if events:
        start_time = min(event.timestamp for event in events)
        end_time = max(event.timestamp for event in events)
        duration_seconds = max(0.0, (end_time - start_time).total_seconds())
    else:
        start_time = None
        end_time = None
        duration_seconds = 0.0

    if cross_layer_edges:
        status = "cross_layer_correlated"
    elif any(event.event_type == "network" for event in events) and any(event.event_type == "log" for event in events):
        status = "detections_without_cross_layer_link"
    else:
        status = "single_layer_or_no_anomaly"

    return {
        "status": status,
        "num_events": len(events),
        "num_network_events": sum(1 for event in events if event.event_type == "network"),
        "num_log_events": sum(1 for event in events if event.event_type == "log"),
        "num_edges": len(edges),
        "num_cross_layer_edges": len(cross_layer_edges),
        "num_clusters": len(clusters),
        "cluster_sizes": cluster_sizes[:10],
        "strongest_edge_score": round(float(max((edge.score for edge in edges), default=0.0)), 6),
        "average_cross_layer_edge_score": round(
            float(np.mean([edge.score for edge in cross_layer_edges])) if cross_layer_edges else 0.0,
            6,
        ),
        "time_range": {
            "start": start_time.isoformat() if start_time else None,
            "end": end_time.isoformat() if end_time else None,
            "duration_seconds": round(float(duration_seconds), 3),
        },
    }
