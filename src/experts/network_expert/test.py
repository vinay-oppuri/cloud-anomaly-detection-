from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.experts.network_expert.classifier import classify_network_anomaly, summarize_network_rows
from src.experts.network_expert.model import CNNTransformerClassifier
from src.interpreter.advisor import IncidentAdvisor

DEFAULT_MODEL_PATH = Path("models/network_expert.pth")
DEFAULT_PREPROCESSED_DIR = Path("data/processed")
DEFAULT_METRICS_PATH = Path("models/network_meta.json")
RAW_LOG_AUTO_THRESHOLD_CAP = 0.10
RAW_LOG_DECISION_MIN_RATIO = 0.20
RAW_LOG_DECISION_MIN_MAX_SCORE = 0.50
RAW_LOG_RULE_HIGH_PPS = 5_000.0
RAW_LOG_RULE_HIGH_BPS = 1_000_000.0
RAW_LOG_RULE_ONE_SIDED = 0.30
RAW_LOG_RULE_SHORT_FLOW = 0.40

_NUM_PATTERN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
_KEY_VALUE_RE = re.compile(
    rf"(?P<key>[A-Za-z][A-Za-z0-9_./\- ]{{1,80}}?)\s*(?:=|:)\s*(?P<value>{_NUM_PATTERN})"
)
_ALIAS_TO_FEATURE = {
    "duration": "Flow Duration",
    "flowduration": "Flow Duration",
    "fwdpkts": "Tot Fwd Pkts",
    "forwardpkts": "Tot Fwd Pkts",
    "forwardpackets": "Tot Fwd Pkts",
    "totfwdpkts": "Tot Fwd Pkts",
    "bwdpkts": "Tot Bwd Pkts",
    "backwardpkts": "Tot Bwd Pkts",
    "backwardpackets": "Tot Bwd Pkts",
    "totbwdpkts": "Tot Bwd Pkts",
    "fwdbytes": "TotLen Fwd Pkts",
    "totlenfwdpkts": "TotLen Fwd Pkts",
    "bwdbytes": "TotLen Bwd Pkts",
    "totlenbwdpkts": "TotLen Bwd Pkts",
    "bytesps": "Flow Byts/s",
    "bytespersec": "Flow Byts/s",
    "bytepersec": "Flow Byts/s",
    "flowbytesps": "Flow Byts/s",
    "flowbytss": "Flow Byts/s",
    "pktsps": "Flow Pkts/s",
    "packetspersec": "Flow Pkts/s",
    "packetpersec": "Flow Pkts/s",
    "pps": "Flow Pkts/s",
    "flowpktsps": "Flow Pkts/s",
    "flowpktss": "Flow Pkts/s",
    "syn": "SYN Flag Cnt",
    "synflagcnt": "SYN Flag Cnt",
    "ack": "ACK Flag Cnt",
    "ackflagcnt": "ACK Flag Cnt",
    "rst": "RST Flag Cnt",
    "rstflagcnt": "RST Flag Cnt",
    "fin": "FIN Flag Cnt",
    "finflagcnt": "FIN Flag Cnt",
    "psh": "PSH Flag Cnt",
    "pshflagcnt": "PSH Flag Cnt",
    "urg": "URG Flag Cnt",
    "urgflagcnt": "URG Flag Cnt",
}


@dataclass
class _ManualRobustScalerCompat:
    """Compatibility shim for joblib artifacts saved from script scope."""

    center_: np.ndarray
    scale_: np.ndarray
    feature_names_in_: np.ndarray
    n_features_in_: int

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.center_) / self.scale_


@dataclass(slots=True)
class BinaryAnalyzeConfig:
    model_path: Path
    preprocessed_dir: Path
    metrics_path: Path
    dataset_split: str | None
    input_file: Path | None
    log_file: Path | None
    log_text: str | None
    input_format: str
    interactive: bool
    threshold: float | None
    device: str
    batch_size: int
    window_step: int
    max_report_items: int


def parse_args() -> BinaryAnalyzeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run binary network anomaly detection.\n"
            "By default this evaluates the model on the CICIDS preprocessed test split.\n"
            "You can also run file/log inference with --input-file, --log-file, --log-text, or --interactive."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--dataset-split",
        type=str,
        choices=("train", "val", "test"),
        default=None,
        help="Evaluate directly on a preprocessed CICIDS split. Default: test when no input source is provided.",
    )
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Raw text log file (one raw network log per line).",
    )
    parser.add_argument(
        "--log-text",
        type=str,
        default=None,
        help="Inline raw text logs. Supports multi-line strings.",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=("auto", "csv", "json", "jsonl"),
        default="auto",
        help="Ignored in interactive mode.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt user to enter one flow record per line in terminal.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override anomaly threshold. Default: from metrics file, else 0.5.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument("--max-report-items", type=int, default=20)
    ns = parser.parse_args()
    dataset_split = ns.dataset_split
    has_runtime_input = bool(ns.input_file or ns.log_file or ns.log_text or ns.interactive)
    if dataset_split is not None and has_runtime_input:
        parser.error("--dataset-split cannot be combined with file/log/interactive input.")
    if dataset_split is None and not has_runtime_input:
        dataset_split = "test"
    return BinaryAnalyzeConfig(
        model_path=ns.model_path,
        preprocessed_dir=ns.preprocessed_dir,
        metrics_path=ns.metrics_path,
        dataset_split=dataset_split,
        input_file=ns.input_file,
        log_file=ns.log_file,
        log_text=ns.log_text,
        input_format=ns.input_format,
        interactive=bool(ns.interactive),
        threshold=ns.threshold,
        device=ns.device,
        batch_size=max(1, int(ns.batch_size)),
        window_step=max(1, int(ns.window_step)),
        max_report_items=max(1, int(ns.max_report_items)),
    )


def main() -> None:
    config = parse_args()
    report = run(config)
    print(json.dumps(report, indent=2))


def run(config: BinaryAnalyzeConfig) -> dict[str, Any]:
    (
        model,
        class_names,
        feature_cols,
        scaler,
        seq_len,
        threshold_from_metrics,
        threshold_source_from_metrics,
        device,
    ) = _load_runtime_assets(config)

    threshold, threshold_source = _resolve_runtime_threshold(
        config=config,
        threshold_from_metrics=float(threshold_from_metrics),
        threshold_source_from_metrics=threshold_source_from_metrics,
    )

    if config.dataset_split is not None:
        return _run_dataset_evaluation(
            config=config,
            model=model,
            class_names=class_names,
            seq_len=seq_len,
            threshold=threshold,
            threshold_source=threshold_source,
            device=device,
        )

    input_rows = _load_input_rows(config, feature_cols=feature_cols)
    if not input_rows:
        raise ValueError(
            "No usable rows found from input. "
            "Provide --input-file/--log-file/--log-text or use --interactive."
        )

    scaled_rows = _preprocess_input_rows(
        rows=input_rows,
        feature_cols=feature_cols,
        scaler=scaler,
    )
    sequences, sequence_start_rows = _build_sequences(
        scaled_rows=scaled_rows,
        seq_len=seq_len,
        step=config.window_step,
    )
    scores = _predict_anomaly_scores(
        model=model,
        sequences=sequences,
        class_names=class_names,
        batch_size=config.batch_size,
        device=device,
    )

    predictions = scores >= threshold
    anomaly_idx = np.flatnonzero(predictions)
    benign_count = int(predictions.shape[0] - anomaly_idx.shape[0])
    anomaly_count = int(anomaly_idx.shape[0])

    result_rows: list[dict[str, Any]] = []
    max_items = min(config.max_report_items, int(scores.shape[0]))
    if max_items > 0:
        # Report highest-risk windows first.
        top_idx = np.argsort(scores)[::-1][:max_items]
        for i in top_idx:
            result_rows.append(
                {
                    "window_index": int(i),
                    "source_row_start": int(sequence_start_rows[i]),
                    "source_row_end": int(sequence_start_rows[i] + seq_len - 1),
                    "anomaly_score": float(scores[i]),
                    "predicted_label": "Anomaly" if bool(predictions[i]) else "Benign",
                }
            )

    final_score = float(scores[-1]) if scores.size > 0 else 0.0
    final_label = "Anomaly" if final_score >= threshold else "Benign"
    anomaly_ratio = float(anomaly_count / max(1, predictions.shape[0]))
    max_score = float(scores.max(initial=0.0))
    session_label, session_reason = _resolve_session_decision(
        config=config,
        final_window_label=final_label,
        anomaly_window_ratio=anomaly_ratio,
        max_anomaly_score=max_score,
        input_rows=input_rows,
    )
    response: dict[str, Any] = {
        "task": "network_binary_realworld_test",
        "model_path": str(config.model_path),
        "preprocessed_dir": str(config.preprocessed_dir),
        "input_source": _describe_input_source(config),
        "device": str(device),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "decision_label": session_label,
        "decision_reason": session_reason,
        "num_input_rows": int(scaled_rows.shape[0]),
        "num_sequences": int(sequences.shape[0]),
        "sequence_length": int(seq_len),
        "summary": {
            "final_window_label": final_label,
            "final_window_score": final_score,
            "anomaly_windows": anomaly_count,
            "benign_windows": benign_count,
            "anomaly_window_ratio": anomaly_ratio,
            "max_anomaly_score": max_score,
            "mean_anomaly_score": float(scores.mean()) if scores.size > 0 else 0.0,
            "session_decision_label": session_label,
            "session_decision_reason": session_reason,
        },
        "top_windows_by_score": result_rows,
        "config": {
            **asdict(config),
            "model_path": str(config.model_path),
            "preprocessed_dir": str(config.preprocessed_dir),
            "metrics_path": str(config.metrics_path),
            "input_file": str(config.input_file) if config.input_file is not None else None,
            "log_file": str(config.log_file) if config.log_file is not None else None,
            "log_text": "<provided>" if config.log_text else None,
        },
    }
    _append_network_incident_analysis(
        response=response,
        config=config,
        input_rows=input_rows,
        predictions=predictions,
        sequence_start_rows=sequence_start_rows,
        seq_len=seq_len,
        anomaly_score=max_score,
        session_label=session_label,
    )
    return response


def _run_dataset_evaluation(
    *,
    config: BinaryAnalyzeConfig,
    model: torch.nn.Module,
    class_names: list[str],
    seq_len: int,
    threshold: float,
    threshold_source: str,
    device: torch.device,
) -> dict[str, Any]:
    split_name = str(config.dataset_split)
    x_split, y_split = _load_dataset_split(config.preprocessed_dir, split_name)
    scores = _predict_anomaly_scores(
        model=model,
        sequences=x_split,
        class_names=class_names,
        batch_size=config.batch_size,
        device=device,
    )

    y_raw = _to_numpy_int_labels(y_split)
    anomaly_class_index = _resolve_anomaly_class_index(class_names)
    y_true = (y_raw == anomaly_class_index).astype(np.int64, copy=False)
    y_pred = (scores >= threshold).astype(np.int64, copy=False)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Anomaly"],
        output_dict=True,
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "anomaly_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "anomaly_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "anomaly_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "benign_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "benign_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "benign_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        metrics["roc_auc"] = None

    top_rows: list[dict[str, Any]] = []
    max_items = min(config.max_report_items, int(scores.shape[0]))
    if max_items > 0:
        top_idx = np.argsort(scores)[::-1][:max_items]
        for i in top_idx:
            top_rows.append(
                {
                    "sequence_index": int(i),
                    "anomaly_score": float(scores[i]),
                    "predicted_label": "Anomaly" if bool(y_pred[i]) else "Benign",
                    "true_label": "Anomaly" if bool(y_true[i]) else "Benign",
                }
            )

    return {
        "task": "network_binary_dataset_test",
        "model_path": str(config.model_path),
        "preprocessed_dir": str(config.preprocessed_dir),
        "metrics_path": str(config.metrics_path),
        "dataset_split": split_name,
        "device": str(device),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "num_sequences": int(len(y_true)),
        "sequence_length": int(seq_len),
        "n_features": int(x_split.shape[-1]),
        "class_names": class_names,
        "metrics": metrics,
        "support": {
            "benign": int((y_true == 0).sum()),
            "anomaly": int((y_true == 1).sum()),
        },
        "score_summary": {
            "min": float(scores.min()) if scores.size > 0 else 0.0,
            "mean": float(scores.mean()) if scores.size > 0 else 0.0,
            "max": float(scores.max()) if scores.size > 0 else 0.0,
        },
        "classification_report": report,
        "top_sequences_by_score": top_rows,
        "config": {
            **asdict(config),
            "model_path": str(config.model_path),
            "preprocessed_dir": str(config.preprocessed_dir),
            "metrics_path": str(config.metrics_path),
            "input_file": None,
            "log_file": None,
            "log_text": None,
        },
    }


def _load_dataset_split(preprocessed_dir: Path, split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    x_path = preprocessed_dir / f"{split_name}_X.pt"
    y_path = preprocessed_dir / f"{split_name}_y.pt"
    if not x_path.exists():
        raise FileNotFoundError(f"CICIDS split tensor not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"CICIDS split tensor not found: {y_path}")
    x_split = torch.load(x_path, map_location="cpu")
    y_split = torch.load(y_path, map_location="cpu")
    if not isinstance(x_split, torch.Tensor):
        x_split = torch.as_tensor(x_split, dtype=torch.float32)
    if not isinstance(y_split, torch.Tensor):
        y_split = torch.as_tensor(y_split, dtype=torch.long)
    return x_split.float().cpu(), y_split.long().cpu()


def _to_numpy_int_labels(labels: torch.Tensor | np.ndarray | list[int]) -> np.ndarray:
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    return np.asarray(labels, dtype=np.int64).reshape(-1)


def _resolve_anomaly_class_index(class_names: list[str]) -> int:
    if "Anomaly" in class_names:
        return int(class_names.index("Anomaly"))
    if len(class_names) == 2 and "Benign" in class_names:
        return 1 - int(class_names.index("Benign"))
    return 1


def _append_network_incident_analysis(
    *,
    response: dict[str, Any],
    config: BinaryAnalyzeConfig,
    input_rows: list[dict[str, Any]],
    predictions: np.ndarray,
    sequence_start_rows: np.ndarray,
    seq_len: int,
    anomaly_score: float,
    session_label: str,
) -> None:
    focus_rows = _select_focus_rows(
        input_rows=input_rows,
        predictions=predictions,
        sequence_start_rows=sequence_start_rows,
        seq_len=seq_len,
    )

    if session_label != "Anomaly":
        response["anomaly_type"] = "Normal"
        response["anomaly_description"] = "No network anomaly pattern detected."
        response["reason"] = "Current flow pattern is consistent with benign network behavior."
        response["action"] = "Continue monitoring and retain the current detection threshold."
        response["metadata"] = {
            "severity_level": "Low",
            "triggered_experts": [],
            "advice_source": "heuristic",
            "classification_source": "rule_based",
            "classification_confidence": 1.0,
            "classification_matched_rules": [],
            "classification_feature_summary": {
                "row_count": int(len(focus_rows)),
            },
            "input_mode": _input_mode(config),
        }
        return

    classified = classify_network_anomaly(rows=focus_rows, anomaly_score=anomaly_score)
    advisor = IncidentAdvisor()
    incident = {
        "event_name": _describe_input_source(config),
        "anomaly_detected": True,
        "anomaly_type": classified["anomaly_type"],
        "severity_level": classified["severity"],
        "max_anomaly_score": anomaly_score,
        "triggered_experts": ["network_expert"],
        "classification_source": "rule_based",
        "classification_confidence": classified["confidence"],
        "classification_matched_rules": classified["matched_rules"],
        "classification_description": classified["description"],
        "event_names": list(classified["matched_rules"]),
        "predictions": [
            {
                "expert_name": "network_expert",
                "anomaly_score": anomaly_score,
                "predicted_class": session_label,
                "confidence": anomaly_score,
                "metadata": {
                    "feature_summary": classified["feature_summary"],
                    "input_mode": _input_mode(config),
                },
            }
        ],
    }
    advice = advisor.advise(incident)

    response["anomaly_type"] = classified["anomaly_type"]
    response["anomaly_description"] = classified["description"]
    response["reason"] = advice.reason
    response["action"] = advice.action
    response["metadata"] = {
        "severity_level": classified["severity"],
        "triggered_experts": ["network_expert"],
        "advice_source": advice.source,
        "classification_source": "rule_based",
        "classification_confidence": classified["confidence"],
        "classification_matched_rules": classified["matched_rules"],
        "classification_feature_summary": classified["feature_summary"],
        "focus_row_count": int(len(focus_rows)),
        "input_mode": _input_mode(config),
    }


def _select_focus_rows(
    *,
    input_rows: list[dict[str, Any]],
    predictions: np.ndarray,
    sequence_start_rows: np.ndarray,
    seq_len: int,
) -> list[dict[str, Any]]:
    if not input_rows:
        return []

    anomaly_windows = np.flatnonzero(predictions)
    if anomaly_windows.size == 0:
        return list(input_rows)

    focus_indexes: set[int] = set()
    for window_idx in anomaly_windows.tolist():
        start = int(sequence_start_rows[window_idx])
        end = min(start + seq_len, len(input_rows))
        focus_indexes.update(range(start, end))

    if not focus_indexes:
        return list(input_rows)
    return [input_rows[idx] for idx in sorted(focus_indexes)]


def _input_mode(config: BinaryAnalyzeConfig) -> str:
    if config.log_file is not None or config.log_text or config.interactive:
        return "raw_log"
    if config.input_file is not None:
        return "structured_file"
    if config.dataset_split is not None:
        return "dataset_split"
    return "unknown"


def _load_runtime_assets(
    config: BinaryAnalyzeConfig,
) -> tuple[
    torch.nn.Module,
    list[str],
    list[str],
    Any,
    int,
    float,
    str,
    torch.device,
]:
    model_path = config.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    preprocessed_dir = config.preprocessed_dir
    scaler_path = preprocessed_dir / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Required preprocessing artifact not found: {scaler_path}")

    feature_cols, seq_len = _load_feature_schema(preprocessed_dir)

    scaler = _load_scaler_with_compat(scaler_path)
    if not hasattr(scaler, "transform"):
        raise TypeError("Loaded scaler does not provide transform().")

    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format at {model_path}.")

    class_names = ["Benign", "Anomaly"]
    if "state_dict" in checkpoint:
        ckpt_cfg = checkpoint.get("config", {})
        class_names_raw = checkpoint.get("class_names", class_names)
        class_names = [str(item) for item in class_names_raw]
        if len(class_names) != 2:
            raise ValueError(f"Expected binary class names in checkpoint, got {class_names}.")

        input_dim = int(ckpt_cfg.get("input_dim", len(feature_cols)))
        model = CNNTransformerClassifier(
            input_dim=input_dim,
            num_classes=int(ckpt_cfg.get("num_classes", 1)),
            conv_channels=int(ckpt_cfg.get("conv_channels", 128)),
            conv_kernel_size=int(ckpt_cfg.get("conv_kernel_size", 3)),
            flow_embedding_dim=int(ckpt_cfg.get("flow_embedding_dim", 128)),
            transformer_heads=int(ckpt_cfg.get("transformer_heads", 4)),
            transformer_layers=int(ckpt_cfg.get("transformer_layers", 2)),
            dim_feedforward=int(ckpt_cfg.get("dim_feedforward", 256)),
            dropout=float(ckpt_cfg.get("dropout", 0.3)),
        )
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    elif "model_state" in checkpoint:
        # Legacy checkpoint from train.py (single-logit binary head).
        from src.experts.network_expert.train import CNN_LSTM_Binary

        legacy_cfg = checkpoint.get("cfg", {})
        input_dim = int(checkpoint.get("n_features", len(feature_cols)))
        model = CNN_LSTM_Binary(
            n_features=input_dim,
            cnn_channels=int(legacy_cfg.get("cnn_channels", 128)),
            cnn_kernel=int(legacy_cfg.get("cnn_kernel", 3)),
            lstm_hidden=int(legacy_cfg.get("lstm_hidden", 256)),
            lstm_layers=int(legacy_cfg.get("lstm_layers", 2)),
            fc_hidden=int(legacy_cfg.get("fc_hidden", 128)),
            dropout=float(legacy_cfg.get("dropout", 0.3)),
        )
        model.load_state_dict(checkpoint["model_state"], strict=False)
        seq_len = int(checkpoint.get("seq_len", seq_len))
    else:
        raise TypeError(
            f"Expected checkpoint with 'state_dict' or 'model_state' at {model_path}."
        )

    device = _resolve_device(config.device)
    model = model.to(device)
    model.eval()

    threshold_from_metrics, threshold_source_from_metrics = _load_threshold(metrics_path=config.metrics_path)
    if threshold_source_from_metrics == "default":
        ckpt_thr = _load_threshold_from_checkpoint(checkpoint)
        if ckpt_thr is not None:
            threshold_from_metrics = float(ckpt_thr)
            threshold_source_from_metrics = "checkpoint_threshold"
    return model, class_names, feature_cols, scaler, seq_len, threshold_from_metrics, threshold_source_from_metrics, device


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_feature_schema(preprocessed_dir: Path) -> tuple[list[str], int]:
    feature_cols_path = preprocessed_dir / "feature_cols.json"
    class_info_path = preprocessed_dir / "class_info.json"
    meta_path = preprocessed_dir / "meta.json"

    feature_cols: list[str] = []
    seq_len = 10

    if feature_cols_path.exists():
        payload = json.loads(feature_cols_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            feature_cols = [str(item) for item in payload]

    if class_info_path.exists():
        class_info = json.loads(class_info_path.read_text(encoding="utf-8"))
        seq_len = int(class_info.get("seq_len", seq_len))

    if not feature_cols or not class_info_path.exists():
        if not meta_path.exists():
            if not feature_cols:
                raise FileNotFoundError(
                    f"Required preprocessing artifact not found: {feature_cols_path} (or legacy {meta_path})."
                )
        else:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not feature_cols:
                raw_cols = meta.get("feat_cols", [])
                if isinstance(raw_cols, list):
                    feature_cols = [str(item) for item in raw_cols]
            seq_len = int(meta.get("seq_len", seq_len))

    if not feature_cols:
        raise ValueError("Could not resolve feature columns from preprocessing artifacts.")
    if seq_len <= 0:
        raise ValueError("Invalid seq_len in preprocessing artifacts.")
    return feature_cols, seq_len


def _load_threshold(metrics_path: Path) -> tuple[float, str]:
    if not metrics_path.exists():
        return 0.5, "default"
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = float(payload.get("best_threshold", payload.get("threshold", 0.5)))
        source = str(payload.get("threshold_source", "metrics_best_threshold"))
        raw_calibration = payload.get("raw_log_calibration")
        if source == "metrics_best_threshold" and isinstance(raw_calibration, dict):
            if bool(raw_calibration.get("status") == "applied"):
                source = "raw_log_calibration"
        return threshold, source
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0.5, "default"


def _load_threshold_from_checkpoint(checkpoint: dict[str, Any]) -> float | None:
    for key in ("threshold",):
        raw = checkpoint.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_runtime_threshold(
    *,
    config: BinaryAnalyzeConfig,
    threshold_from_metrics: float,
    threshold_source_from_metrics: str,
) -> tuple[float, str]:
    if config.threshold is not None:
        return float(config.threshold), "manual_override"

    uses_raw_log_mode = bool(config.log_file is not None or config.log_text or config.interactive)
    if uses_raw_log_mode:
        if threshold_source_from_metrics == "raw_log_calibration":
            return float(threshold_from_metrics), "metrics_raw_log_calibration"
        # Raw text parsing can dilute anomaly scores vs CICIDS validation tensors.
        # Cap the default threshold to avoid missing obvious anomalies.
        return min(float(threshold_from_metrics), RAW_LOG_AUTO_THRESHOLD_CAP), "raw_log_auto_cap"
    return float(threshold_from_metrics), threshold_source_from_metrics


def _resolve_session_decision(
    *,
    config: BinaryAnalyzeConfig,
    final_window_label: str,
    anomaly_window_ratio: float,
    max_anomaly_score: float,
    input_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    uses_raw_log_mode = bool(config.log_file is not None or config.log_text or config.interactive)
    if not uses_raw_log_mode:
        return final_window_label, "final_window_threshold"

    if max_anomaly_score >= RAW_LOG_DECISION_MIN_MAX_SCORE:
        return "Anomaly", "raw_log_max_score_rule"
    if anomaly_window_ratio >= RAW_LOG_DECISION_MIN_RATIO:
        return "Anomaly", "raw_log_ratio_rule"
    # Raw logs can have sparse feature extraction; use simple feature summary rules
    # as a safety net to catch obvious high-rate one-sided attacks.
    if input_rows:
        summary = summarize_network_rows(input_rows)
        if (
            summary.get("p95_flow_pkts_per_sec", 0.0) >= RAW_LOG_RULE_HIGH_PPS
            and summary.get("p95_flow_bytes_per_sec", 0.0) >= RAW_LOG_RULE_HIGH_BPS
            and summary.get("one_sided_ratio", 0.0) >= RAW_LOG_RULE_ONE_SIDED
            and summary.get("short_flow_ratio", 0.0) >= RAW_LOG_RULE_SHORT_FLOW
        ):
            return "Anomaly", "raw_log_feature_rule"
    return "Benign", "raw_log_session_rule"


def _load_scaler_with_compat(path: Path) -> Any:
    try:
        return joblib.load(path)
    except AttributeError as exc:
        # Some preprocess runs pickle ManualRobustScaler under script scope (__main__).
        if "ManualRobustScaler" not in str(exc):
            raise
        main_mod = sys.modules.get("__main__")
        if main_mod is not None and not hasattr(main_mod, "ManualRobustScaler"):
            setattr(main_mod, "ManualRobustScaler", _ManualRobustScalerCompat)
        return joblib.load(path)


def _describe_input_source(config: BinaryAnalyzeConfig) -> str:
    if config.dataset_split is not None:
        return f"cicids:{config.dataset_split}"
    if config.log_file is not None:
        return str(config.log_file)
    if config.log_text:
        return "inline_log_text"
    if config.input_file is not None:
        return str(config.input_file)
    return "interactive_terminal"


def _load_input_rows(config: BinaryAnalyzeConfig, *, feature_cols: list[str]) -> list[dict[str, Any]]:
    if config.log_text:
        return _parse_raw_text_logs(config.log_text, feature_cols=feature_cols)
    if config.log_file is not None:
        if not config.log_file.exists():
            raise FileNotFoundError(f"Raw log file not found: {config.log_file}")
        raw_text = config.log_file.read_text(encoding="utf-8", errors="ignore")
        return _parse_raw_text_logs(raw_text, feature_cols=feature_cols)
    if config.input_file is not None:
        return _read_rows_from_file(config.input_file, file_format=config.input_format, feature_cols=feature_cols)
    return _read_rows_interactive(feature_cols=feature_cols)


def _read_rows_from_file(path: Path, *, file_format: str, feature_cols: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    fmt = file_format
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix == ".csv":
            fmt = "csv"
        elif suffix in {".jsonl", ".ndjson"}:
            fmt = "jsonl"
        else:
            fmt = "json"

    if fmt == "csv":
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [{str(k): v for k, v in row.items()} for row in reader]
            return _coerce_rows_to_feature_schema(rows, feature_cols=feature_cols)

    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    if fmt == "jsonl":
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError("JSONL line must be an object.")
            rows.append({str(k): v for k, v in obj.items()})
        return _coerce_rows_to_feature_schema(rows, feature_cols=feature_cols)

    obj = json.loads(text)
    if isinstance(obj, list):
        rows_out: list[dict[str, Any]] = []
        for item in obj:
            if not isinstance(item, dict):
                raise TypeError("JSON array input must contain objects.")
            rows_out.append({str(k): v for k, v in item.items()})
        return _coerce_rows_to_feature_schema(rows_out, feature_cols=feature_cols)
    if isinstance(obj, dict):
        if "rows" in obj and isinstance(obj["rows"], list):
            rows_out = []
            for item in obj["rows"]:
                if not isinstance(item, dict):
                    raise TypeError("JSON 'rows' items must be objects.")
                rows_out.append({str(k): v for k, v in item.items()})
            return _coerce_rows_to_feature_schema(rows_out, feature_cols=feature_cols)
        return _coerce_rows_to_feature_schema([{str(k): v for k, v in obj.items()}], feature_cols=feature_cols)
    raise TypeError("Unsupported JSON format. Use object, array of objects, or {'rows': [...]} structure.")


def _read_rows_interactive(*, feature_cols: list[str]) -> list[dict[str, Any]]:
    print("Interactive network flow input mode.")
    print("Enter one flow per line as JSON, key=value pairs, or raw text log line.")
    print("Examples:")
    print('  {"Flow Duration": 12345, "Tot Fwd Pkts": 10, "Dst Port": 443}')
    print("  Flow Duration=12345,Tot Fwd Pkts=10,Dst Port=443")
    print("  [flow] duration=4321 fwd_packets=20 bwd_packets=8 bytes/s=9000 pps=120")
    print("Submit an empty line to run prediction.\n")

    rows: list[dict[str, Any]] = []
    while True:
        try:
            line = input("flow> ").strip()
        except EOFError:
            break
        if not line:
            break
        parsed = _parse_interactive_line(line, feature_cols=feature_cols)
        if not parsed:
            print("Could not parse line. Include at least one numeric network field.")
            continue
        rows.append(parsed)
    return rows


def _parse_interactive_line(line: str, *, feature_cols: list[str]) -> dict[str, Any] | None:
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows = _coerce_rows_to_feature_schema([{str(k): v for k, v in obj.items()}], feature_cols=feature_cols)
            return rows[0] if rows else None
    except json.JSONDecodeError:
        pass

    parts = [item.strip() for item in line.split(",") if item.strip()]
    row: dict[str, Any] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", maxsplit=1)
        row[key.strip()] = value.strip()
    if row:
        rows = _coerce_rows_to_feature_schema([row], feature_cols=feature_cols)
        if rows:
            return rows[0]
    parsed_raw = _parse_raw_text_logs(line, feature_cols=feature_cols)
    return parsed_raw[0] if parsed_raw else None


def _coerce_rows_to_feature_schema(
    rows: list[dict[str, Any]],
    *,
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    lookup = _feature_lookup(feature_cols)
    out: list[dict[str, Any]] = []
    for row in rows:
        mapped: dict[str, Any] = {}
        for key, value in row.items():
            feature_name = _map_raw_key_to_feature(str(key), lookup)
            if feature_name is None:
                continue
            mapped[feature_name] = value
        if mapped:
            out.append(mapped)
    return out


def _parse_raw_text_logs(raw_text: str, *, feature_cols: list[str]) -> list[dict[str, Any]]:
    lookup = _feature_lookup(feature_cols)
    rows: list[dict[str, Any]] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = _extract_row_from_raw_line(line, feature_lookup=lookup)
        if row:
            rows.append(row)
    return rows


def _extract_row_from_raw_line(line: str, *, feature_lookup: dict[str, str]) -> dict[str, Any] | None:
    row: dict[str, Any] = {}
    for match in _KEY_VALUE_RE.finditer(line):
        raw_key = match.group("key")
        feature_name = _map_raw_key_to_feature(raw_key, feature_lookup)
        if feature_name is None:
            continue
        row[feature_name] = match.group("value")
    _apply_raw_log_fallbacks(line, row=row, feature_lookup=feature_lookup)
    return row or None


def _apply_raw_log_fallbacks(line: str, *, row: dict[str, Any], feature_lookup: dict[str, str]) -> None:
    rules = (
        ("Flow Duration", rf"(?:flow[_\s-]*duration|duration)\D*({_NUM_PATTERN})"),
        ("Tot Fwd Pkts", rf"(?:fwd|forward)[^\d]{{0,20}}(?:pkts|packets)\D*({_NUM_PATTERN})"),
        ("Tot Bwd Pkts", rf"(?:bwd|backward)[^\d]{{0,20}}(?:pkts|packets)\D*({_NUM_PATTERN})"),
        ("Flow Byts/s", rf"(?:bytes[/_]s|bytesps|bytes[_\s-]*per[_\s-]*sec|throughput)\D*({_NUM_PATTERN})"),
        ("Flow Pkts/s", rf"(?:pkts[/_]s|packets[/_]s|pps|packet[_\s-]*rate)\D*({_NUM_PATTERN})"),
    )
    for target_feature, pattern in rules:
        target_norm = _normalize_key(target_feature)
        feature_name = feature_lookup.get(target_norm)
        if feature_name is None or feature_name in row:
            continue
        found = re.search(pattern, line, flags=re.IGNORECASE)
        if found is None:
            continue
        row[feature_name] = found.group(1)


def _feature_lookup(feature_cols: list[str]) -> dict[str, str]:
    return {_normalize_key(name): name for name in feature_cols}


def _map_raw_key_to_feature(raw_key: str, feature_lookup: dict[str, str]) -> str | None:
    norm = _normalize_key(raw_key)
    if not norm:
        return None
    if norm in feature_lookup:
        return feature_lookup[norm]
    alias_target = _ALIAS_TO_FEATURE.get(norm)
    if alias_target is None:
        return None
    return feature_lookup.get(_normalize_key(alias_target))


def _normalize_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _preprocess_input_rows(
    *,
    rows: list[dict[str, Any]],
    feature_cols: list[str],
    scaler: Any,
) -> np.ndarray:
    n_rows = len(rows)
    n_features = len(feature_cols)
    center = np.asarray(getattr(scaler, "center_", np.zeros(n_features)), dtype=np.float32)
    if center.shape[0] != n_features:
        center = np.zeros(n_features, dtype=np.float32)

    x_raw = np.empty((n_rows, n_features), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_cols):
            raw_value = row.get(name, center[j])
            x_raw[i, j] = _to_finite_float(raw_value, fallback=float(center[j]))

    x_scaled = scaler.transform(x_raw)
    x_scaled = np.asarray(x_scaled, dtype=np.float32)
    x_scaled = np.clip(x_scaled, -10.0, 10.0)
    return x_scaled


def _to_finite_float(value: Any, *, fallback: float = 0.0) -> float:
    text = str(value).strip().replace(",", "")
    if not text:
        return fallback
    lowered = text.lower()
    if lowered in {"nan", "na", "none", "null", "inf", "+inf", "-inf", "infinity", "-infinity"}:
        return fallback
    try:
        val = float(text)
    except ValueError:
        return fallback
    if not np.isfinite(val):
        return fallback
    return float(val)


def _build_sequences(
    *,
    scaled_rows: np.ndarray,
    seq_len: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_features = scaled_rows.shape
    if n_rows == 0:
        raise ValueError("No rows available after preprocessing.")

    if n_rows < seq_len:
        pad = np.zeros((seq_len - n_rows, n_features), dtype=np.float32)
        seq = np.concatenate([pad, scaled_rows], axis=0)
        return seq[np.newaxis, :, :], np.asarray([0], dtype=np.int64)

    windows: list[np.ndarray] = []
    starts: list[int] = []
    for start in range(0, n_rows - seq_len + 1, step):
        end = start + seq_len
        windows.append(scaled_rows[start:end])
        starts.append(start)
    x_seq = np.stack(windows, axis=0).astype(np.float32, copy=False)
    return x_seq, np.asarray(starts, dtype=np.int64)


@torch.inference_mode()
def _predict_anomaly_scores(
    *,
    model: torch.nn.Module,
    sequences: np.ndarray | torch.Tensor,
    class_names: list[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    anomaly_idx = class_names.index("Anomaly") if "Anomaly" in class_names else 1
    out_scores: list[np.ndarray] = []
    total = int(sequences.shape[0])

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        if isinstance(sequences, torch.Tensor):
            batch = sequences[start:end].to(device=device, dtype=torch.float32)
        else:
            batch = torch.from_numpy(sequences[start:end]).to(device=device, dtype=torch.float32)
        logits = model(batch)
        if logits.ndim == 1:
            scores = torch.sigmoid(logits)
        elif logits.ndim == 2 and logits.shape[1] == 1:
            scores = torch.sigmoid(logits[:, 0])
        elif logits.ndim == 2 and logits.shape[1] >= 2:
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, anomaly_idx]
        else:
            raise ValueError(f"Unsupported logits shape from model: {tuple(logits.shape)}")
        out_scores.append(scores.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(out_scores, axis=0) if out_scores else np.empty((0,), dtype=np.float32)


if __name__ == "__main__":
    main()
