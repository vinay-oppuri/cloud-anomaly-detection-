from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from tqdm.auto import tqdm

from anomaly_classifier import classify_anomaly
from src.experts.system_expert.model import SystemExpertTransformer
from src.interpreter.advisor import IncidentAdvisor
from src.training.data import SequenceDataset
from src.training.metrics import compute_classification_report
from src.training.runner import build_dataloader, evaluate_model

DEFAULT_PROCESSED_PATH = Path("data/processed/hdfs_processed.pt")
DEFAULT_CACHE_PATH = Path("data/processed/hdfs_cache.json")
DEFAULT_MODEL_PATH = Path("models/system_expert_best.pth")

FALLBACK_PROCESSED_PATHS: tuple[Path, ...] = (
    Path("D:/cloud-anomaly-artifacts/hdfs_processed.pt"),
)
FALLBACK_MODEL_PATHS: tuple[Path, ...] = (
    Path("models/system_expert_best.pth"),
    Path("D:/cloud-anomaly-artifacts/models/system_expert_best.pth"),
)
WORKFLOW_TOTAL_STEPS = 6

EVENT_ID_PATTERN = re.compile(r"\be\d+\b", flags=re.IGNORECASE)

# Lightweight template-to-event mapping for raw HDFS log lines that do not already
# include explicit event_id=E* tokens.
HDFS_TEMPLATE_MAP: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"allocate\s+blk_", flags=re.IGNORECASE), "e5"),
    (re.compile(r"receiving\s+block", flags=re.IGNORECASE), "e26"),
    (re.compile(r"packetresponder.*success", flags=re.IGNORECASE), "e11"),
    (re.compile(r"completefile", flags=re.IGNORECASE), "e9"),
    (re.compile(r"addstoredblock", flags=re.IGNORECASE), "e21"),
    (re.compile(r"state change.*active", flags=re.IGNORECASE), "e23"),
    (re.compile(r"writeblock", flags=re.IGNORECASE), "e22"),
    (re.compile(r"finalized block", flags=re.IGNORECASE), "e3"),
    (re.compile(r"slow blockreceiver", flags=re.IGNORECASE), "e1"),
    (re.compile(r"checksum mismatch", flags=re.IGNORECASE), "e17"),
    (re.compile(r"failed to write packet|broken pipe", flags=re.IGNORECASE), "e19"),
    (re.compile(r"invalidateblocks|corrupt replica", flags=re.IGNORECASE), "e24"),
    (re.compile(r"missing\s+\d+\s+replicas", flags=re.IGNORECASE), "e29"),
    (re.compile(r"replicate\s+blk_.*\s+to\s+", flags=re.IGNORECASE), "e13"),
)


@dataclass(slots=True)
class SystemServiceConfig:
    processed_data: Path = DEFAULT_PROCESSED_PATH
    cache_path: Path = DEFAULT_CACHE_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    device: str = "cuda"
    normal_class_index: int = 0
    use_gemini: bool = True
    gemini_model: str = "gemini-2.5-flash"
    show_workflow_progress: bool = True


@dataclass(slots=True)
class EventExtractionResult:
    event_tokens: list[str]
    extracted_from_event_id: int
    inferred_from_templates: int
    unmatched_lines: int


class SystemAnomalyService:
    """
    Single reusable service for system-log anomaly detection.

    This class supports:
    - dataset split evaluation (`evaluate_split`) for offline testing
    - uploaded/raw log analysis (`analyze_*`) for runtime/API use
    """

    def __init__(
        self,
        *,
        config: SystemServiceConfig,
        bundle: dict[str, Any],
        vocab: dict[str, int],
        model_path: Path,
        processed_data_path: Path,
    ) -> None:
        self.config = config
        self.bundle = bundle
        self.vocab = vocab
        self.model_path = model_path
        self.processed_data_path = processed_data_path

        self.sequence_length = int(bundle.get("sequence_length", 128))
        self.class_names = _resolve_class_names(bundle)
        self.normal_class_name = _resolve_normal_class_name(
            class_names=self.class_names,
            normal_class_index=config.normal_class_index,
        )
        self.vocab_size = int(bundle.get("vocab_size", max(vocab.values(), default=1) + 1))
        self.device = _resolve_device(config.device)

        self.expert = SystemExpertTransformer(
            vocab_size=self.vocab_size,
            class_names=self.class_names,
            model_path=self.model_path,
            device=self.device,
        )
        self.advisor = IncidentAdvisor(
            use_gemini=config.use_gemini,
            gemini_model=config.gemini_model,
        )

    @classmethod
    def from_config(cls, config: SystemServiceConfig | None = None) -> SystemAnomalyService:
        resolved_config = config or SystemServiceConfig()
        processed_data_path = _resolve_processed_data_path(resolved_config.processed_data)
        model_path = _resolve_model_path(resolved_config.model_path)

        if not processed_data_path.exists():
            checked = [str(resolved_config.processed_data), *(str(p) for p in FALLBACK_PROCESSED_PATHS)]
            raise FileNotFoundError("Processed dataset not found. Checked: " + ", ".join(checked))
        if not model_path.exists():
            checked = [str(resolved_config.model_path), *(str(p) for p in FALLBACK_MODEL_PATHS)]
            raise FileNotFoundError("Model checkpoint not found. Checked: " + ", ".join(checked))
        if not resolved_config.cache_path.exists():
            raise FileNotFoundError(
                f"HDFS vocab cache not found: {resolved_config.cache_path}. Run `uv run prepare_hdfs`."
            )

        bundle = torch.load(processed_data_path, map_location="cpu")
        cache = json.loads(resolved_config.cache_path.read_text(encoding="utf-8"))
        vocab = _extract_vocab(cache)

        return cls(
            config=resolved_config,
            bundle=bundle,
            vocab=vocab,
            model_path=model_path,
            processed_data_path=processed_data_path,
        )

    def evaluate_split(
        self,
        *,
        split: str = "test",
        batch_size: int = 512,
    ) -> dict[str, Any]:
        split_X, split_y = _load_split(self.bundle, split)
        if split_X.ndim != 2:
            raise ValueError(f"Expected split tokens [N, seq_len], got {tuple(split_X.shape)}")

        loader = build_dataloader(
            SequenceDataset(split_X, split_y),
            batch_size=batch_size,
            shuffle=False,
        )
        criterion = nn.CrossEntropyLoss()
        loss, logits, labels = evaluate_model(
            self.expert.model,
            loader,
            criterion,
            device=self.device,
            input_dtype="long",
            progress_desc=split,
        )
        class_names = self.class_names
        if logits.ndim == 2 and logits.shape[1] != len(class_names):
            class_names = [f"class_{idx}" for idx in range(int(logits.shape[1]))]

        report = compute_classification_report(
            loss=loss,
            labels=labels,
            logits=logits,
            class_names=class_names,
            normal_class_index=self.config.normal_class_index,
        )
        return {
            "task": "test_hdfs_system_expert",
            "mode": "evaluate",
            "split": split,
            "processed_data": str(self.processed_data_path),
            "model_path": str(self.model_path),
            "device": str(self.device),
            "num_samples": int(labels.shape[0]),
            "metrics": report.to_dict(),
            "config": {
                "batch_size": batch_size,
                "normal_class_index": self.config.normal_class_index,
                "gemini_enabled": bool(self.config.use_gemini),
                "gemini_model": self.config.gemini_model,
            },
        }

    def analyze_event_sequence(self, event_sequence: str, *, event_name: str = "uploaded-log") -> dict[str, Any]:
        show_progress = bool(self.config.show_workflow_progress)
        with _workflow_step_progress(
            enabled=show_progress,
            step_number=1,
            step_title="Validate API input",
            total=1,
            unit="step",
        ) as bar:
            if bar is not None:
                bar.update(1)

        if show_progress:
            print("\n[System API Workflow] Step 2/6: Preprocess and extract events")
        extraction = extract_event_tokens_from_lines(
            [event_sequence],
            show_progress=show_progress,
            progress_desc="Step 2/6: Preprocess and extract events",
        )
        if not extraction.event_tokens:
            raise ValueError(
                "No event tokens found in --event-sequence. "
                "Use ids like E5 E26 E11 ..."
            )
        return self.analyze_event_tokens(
            extraction.event_tokens,
            event_name=event_name,
            extraction=extraction,
            source="event_sequence",
            source_value=event_sequence,
        )

    def analyze_log_lines(self, lines: Sequence[str], *, event_name: str = "uploaded-log") -> dict[str, Any]:
        show_progress = bool(self.config.show_workflow_progress)
        with _workflow_step_progress(
            enabled=show_progress,
            step_number=1,
            step_title="Validate API input",
            total=1,
            unit="step",
        ) as bar:
            if bar is not None:
                bar.update(1)

        if show_progress:
            print("\n[System API Workflow] Step 2/6: Preprocess and extract events")
        extraction = extract_event_tokens_from_lines(
            lines,
            show_progress=show_progress,
            progress_desc="Step 2/6: Preprocess and extract events",
        )
        if not extraction.event_tokens:
            raise ValueError(
                "No event tokens could be extracted from log lines. "
                "Provide logs with event ids (E*) or known HDFS templates."
            )
        return self.analyze_event_tokens(
            extraction.event_tokens,
            event_name=event_name,
            extraction=extraction,
            source="log_lines",
            source_value=f"{len(lines)} line(s)",
        )

    def analyze_log_file(self, path: Path, *, event_name: str = "uploaded-log") -> dict[str, Any]:
        show_progress = bool(self.config.show_workflow_progress)
        with _workflow_step_progress(
            enabled=show_progress,
            step_number=1,
            step_title="Load and validate uploaded log file",
            total=1,
            unit="step",
        ) as bar:
            if bar is not None:
                bar.update(1)

        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if show_progress:
            print("\n[System API Workflow] Step 2/6: Preprocess and extract events")
        extraction = extract_event_tokens_from_lines(
            lines,
            show_progress=show_progress,
            progress_desc="Step 2/6: Preprocess and extract events",
        )
        if not extraction.event_tokens:
            raise ValueError(
                f"No event tokens could be extracted from log file {path}. "
                "Provide logs with event ids (E*) or known HDFS templates."
            )
        return self.analyze_event_tokens(
            extraction.event_tokens,
            event_name=event_name,
            extraction=extraction,
            source="log_file",
            source_value=str(path),
        )

    def analyze_event_tokens(
        self,
        event_tokens: Sequence[str],
        *,
        event_name: str,
        extraction: EventExtractionResult | None = None,
        source: str = "event_tokens",
        source_value: str | None = None,
    ) -> dict[str, Any]:
        show_progress = bool(self.config.show_workflow_progress)
        normalized_event_names = [str(token).upper() for token in event_tokens]

        with _workflow_step_progress(
            enabled=show_progress,
            step_number=3,
            step_title="Encode event sequence",
            total=1,
            unit="step",
        ) as bar:
            encoded, unknown_tokens = _encode_event_tokens(
                event_tokens=list(event_tokens),
                vocab=self.vocab,
                sequence_length=self.sequence_length,
            )
            if bar is not None:
                bar.update(1)

        with _workflow_step_progress(
            enabled=show_progress,
            step_number=4,
            step_title="Run transformer inference",
            total=1,
            unit="step",
        ) as bar:
            prediction = self.expert.predict(encoded)
            if bar is not None:
                bar.update(1)

        anomaly_detected = prediction.predicted_class != self.normal_class_name
        model_score = float(prediction.anomaly_score)

        with _workflow_step_progress(
            enabled=show_progress,
            step_number=5,
            step_title="Classify anomaly type (rule-based)",
            total=1,
            unit="step",
        ) as bar:
            classified = classify_anomaly(
                event_names=normalized_event_names,
                anomaly_score=model_score,
            )
            if bar is not None:
                bar.update(1)

        if anomaly_detected:
            anomaly_type = str(classified["anomaly_type"])
            severity = str(classified["severity"])
            classifier_confidence = float(classified["confidence"])
            matched_rules = list(classified["matched_rules"])
            anomaly_description = str(classified["description"])
        else:
            anomaly_type = "Normal"
            severity = "Low"
            classifier_confidence = 1.0
            matched_rules = []
            anomaly_description = "No anomaly pattern detected."

        with _workflow_step_progress(
            enabled=show_progress,
            step_number=6,
            step_title="Generate LLM reason/action and build response",
            total=1,
            unit="step",
        ) as bar:
            incident = {
                "event_name": event_name,
                "anomaly_detected": anomaly_detected,
                "anomaly_type": anomaly_type,
                "severity_level": severity,
                "max_anomaly_score": model_score,
                "triggered_experts": ["system_expert"] if anomaly_detected else [],
                "classification_source": "rule_based",
                "classification_confidence": classifier_confidence,
                "classification_matched_rules": matched_rules,
                "classification_description": anomaly_description,
                "event_names": normalized_event_names,
                "predictions": [
                    {
                        "expert_name": prediction.expert_name,
                        "anomaly_score": model_score,
                        "predicted_class": prediction.predicted_class,
                        "confidence": float(prediction.confidence),
                        "metadata": prediction.metadata,
                    }
                ],
            }
            advice = self.advisor.advise(incident)

            token_tail = [str(token).lower() for token in list(event_tokens)][-min(20, len(event_tokens)) :]
            extraction_stats = extraction or EventExtractionResult(
                event_tokens=list(token_tail),
                extracted_from_event_id=0,
                inferred_from_templates=0,
                unmatched_lines=0,
            )
            response = {
                "task": "test_hdfs_system_expert",
                "mode": "analyze",
                "event_name": event_name,
                "input": {
                    "source": source,
                    "source_value": source_value,
                    "event_token_count": len(event_tokens),
                    "event_tokens_tail": token_tail,
                },
                "anomaly_detected": bool(anomaly_detected),
                "anomaly_type": str(anomaly_type),
                "predicted_label": str(prediction.predicted_class),
                "anomaly_score": model_score,
                "confidence": float(prediction.confidence),
                "anomaly_description": anomaly_description,
                "reason": advice.reason,
                "action": advice.action,
                "metadata": {
                    "severity_level": severity,
                    "triggered_experts": incident["triggered_experts"],
                    "advice_source": advice.source,
                    "classification_source": "rule_based",
                    "classification_confidence": classifier_confidence,
                    "classification_matched_rules": matched_rules,
                    "known_event_count": int(len(event_tokens) - len(unknown_tokens)),
                    "unknown_event_count": int(len(unknown_tokens)),
                    "unknown_event_tokens": unknown_tokens[:20],
                    "sequence_length": self.sequence_length,
                    "device": str(self.device),
                    "model_path": str(self.model_path),
                    "event_tokens_from_event_id": extraction_stats.extracted_from_event_id,
                    "event_tokens_from_template_inference": extraction_stats.inferred_from_templates,
                    "unmatched_log_lines": extraction_stats.unmatched_lines,
                },
            }
            if bar is not None:
                bar.update(1)

        return response


def extract_event_tokens_from_lines(
    lines: Sequence[str],
    *,
    show_progress: bool = False,
    progress_desc: str = "Step 2/6: Preprocess and extract events",
) -> EventExtractionResult:
    tokens: list[str] = []
    from_event_id = 0
    from_templates = 0
    unmatched_lines = 0

    iterator = lines
    if show_progress:
        iterator = tqdm(
            lines,
            total=len(lines),
            desc=progress_desc,
            unit="line",
            dynamic_ncols=True,
            leave=True,
        )

    for line in iterator:
        explicit = EVENT_ID_PATTERN.findall(str(line))
        if explicit:
            normalized = [item.lower() for item in explicit]
            tokens.extend(normalized)
            from_event_id += len(normalized)
            continue

        inferred = _infer_event_from_template(str(line))
        if inferred is not None:
            tokens.append(inferred)
            from_templates += 1
        else:
            unmatched_lines += 1

    return EventExtractionResult(
        event_tokens=tokens,
        extracted_from_event_id=from_event_id,
        inferred_from_templates=from_templates,
        unmatched_lines=unmatched_lines,
    )


def _infer_event_from_template(line: str) -> str | None:
    for pattern, event_id in HDFS_TEMPLATE_MAP:
        if pattern.search(line):
            return event_id
    return None


def _load_split(bundle: dict[str, Any], split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    splits = bundle.get("splits")
    if not isinstance(splits, dict) or split_name not in splits:
        raise KeyError(f"Missing split '{split_name}' in processed bundle.")
    split = splits[split_name]
    if not isinstance(split, dict):
        raise TypeError(f"Split '{split_name}' payload must be a dict.")

    features = split.get("X", split.get("features"))
    labels = split.get("y", split.get("labels"))
    if features is None or labels is None:
        raise KeyError(f"Split '{split_name}' must contain X/y or features/labels.")
    return torch.as_tensor(features, dtype=torch.long), torch.as_tensor(labels, dtype=torch.long)


def _extract_vocab(cache: dict[str, Any]) -> dict[str, int]:
    raw = cache.get("vocab")
    if not isinstance(raw, dict):
        raise ValueError("Invalid cache: missing vocab mapping.")

    parsed: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        try:
            parsed[key] = int(value)
        except (TypeError, ValueError):
            continue
    if "<PAD>" not in parsed:
        parsed["<PAD>"] = 0
    if "<UNK>" not in parsed:
        parsed["<UNK>"] = 1
    return parsed


def _encode_event_tokens(
    *,
    event_tokens: list[str],
    vocab: dict[str, int],
    sequence_length: int,
) -> tuple[torch.Tensor, list[str]]:
    normalized_tokens = [token.lower() for token in event_tokens]
    unk_id = int(vocab.get("<UNK>", 1))
    unknown = [token for token in normalized_tokens if token not in vocab]

    ids = [int(vocab.get(token, unk_id)) for token in normalized_tokens]
    if len(ids) > sequence_length:
        ids = ids[-sequence_length:]
    else:
        ids = ([0] * (sequence_length - len(ids))) + ids
    return torch.tensor(ids, dtype=torch.long), unknown


def _resolve_class_names(bundle: dict[str, Any]) -> list[str]:
    names = bundle.get("class_names")
    if isinstance(names, (list, tuple)) and names:
        return [str(item) for item in names]
    return ["Normal", "Anomaly"]


def _resolve_normal_class_name(*, class_names: list[str], normal_class_index: int) -> str:
    if 0 <= normal_class_index < len(class_names):
        return str(class_names[normal_class_index])
    if "Normal" in class_names:
        return "Normal"
    return str(class_names[0] if class_names else "Normal")


def _resolve_processed_data_path(requested: Path) -> Path:
    if requested.exists():
        return requested
    if requested != DEFAULT_PROCESSED_PATH:
        return requested
    for candidate in FALLBACK_PROCESSED_PATHS:
        if candidate.exists():
            return candidate
    return requested


def _resolve_model_path(requested: Path) -> Path:
    if requested.exists():
        return requested
    if requested != DEFAULT_MODEL_PATH:
        return requested
    for candidate in FALLBACK_MODEL_PATHS:
        if candidate.exists():
            return candidate
    return requested


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


@contextmanager
def _workflow_step_progress(
    *,
    enabled: bool,
    step_number: int,
    step_title: str,
    total: int,
    unit: str,
) -> Any:
    if not enabled:
        yield None
        return

    desc = f"Step {step_number}/{WORKFLOW_TOTAL_STEPS}: {step_title}"
    print(f"\n[System API Workflow] {desc}")
    bar = tqdm(
        total=max(1, int(total)),
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
    )
    try:
        yield bar
    finally:
        if bar.n < bar.total:
            bar.update(bar.total - bar.n)
        bar.close()
