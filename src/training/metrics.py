from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(slots=True)
class ClassificationReport:
    """Structured metrics report for training/validation/test splits."""

    loss: float
    accuracy: float
    macro_f1: float
    per_class: list[dict[str, Any]]
    confusion_matrix: list[list[int]]
    anomaly_roc_auc: float | None
    anomaly_average_precision: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "anomaly_roc_auc": self.anomaly_roc_auc,
            "anomaly_average_precision": self.anomaly_average_precision,
        }


def compute_classification_report(
    *,
    loss: float,
    labels: torch.Tensor,
    logits: torch.Tensor,
    class_names: Sequence[str],
    normal_class_index: int = 0,
) -> ClassificationReport:
    labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    logits_np = logits.detach().cpu().numpy().astype(np.float64, copy=False)
    probs = _softmax_np(logits_np)
    preds = np.argmax(probs, axis=1)

    num_classes = int(len(class_names))
    cm = _confusion_matrix(labels_np, preds, num_classes=num_classes)
    accuracy = float((preds == labels_np).mean()) if labels_np.size > 0 else 0.0

    per_class: list[dict[str, Any]] = []
    f1_values: list[float] = []
    for idx, class_name in enumerate(class_names):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        support = int(cm[idx, :].sum())
        f1_values.append(f1)
        per_class.append(
            {
                "class_index": idx,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0

    anomaly_true = (labels_np != normal_class_index).astype(np.int64)
    if 0 <= normal_class_index < probs.shape[1]:
        anomaly_score = 1.0 - probs[:, normal_class_index]
    else:
        anomaly_score = probs.max(axis=1)

    roc_auc = _binary_roc_auc(anomaly_true, anomaly_score)
    avg_precision = _average_precision(anomaly_true, anomaly_score)

    return ClassificationReport(
        loss=float(loss),
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=cm.astype(np.int64).tolist(),
        anomaly_roc_auc=roc_auc,
        anomaly_average_precision=avg_precision,
    )


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    sums = exps.sum(axis=1, keepdims=True)
    return exps / np.clip(sums, a_min=1e-12, a_max=None)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred, strict=False):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            cm[truth, pred] += 1
    return cm


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positive_mask = y_true == 1
    negative_mask = y_true == 0
    positives = int(positive_mask.sum())
    negatives = int(negative_mask.sum())
    if positives == 0 or negatives == 0:
        return None

    ranks = _average_ranks(y_score)
    positive_rank_sum = float(ranks[positive_mask].sum())
    auc = (positive_rank_sum - (positives * (positives + 1) / 2.0)) / (positives * negatives)
    return float(auc)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = int((y_true == 1).sum())
    if positives == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]

    tp = 0
    fp = 0
    precision_sum = 0.0
    for label in sorted_true:
        if int(label) == 1:
            tp += 1
            precision_sum += tp / max(1, tp + fp)
        else:
            fp += 1

    return float(precision_sum / positives)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty_like(sorted_values, dtype=np.float64)

    index = 0
    while index < sorted_values.size:
        end = index + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[index]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        ranks[index:end] = avg_rank
        index = end

    unsorted_ranks = np.empty_like(ranks)
    unsorted_ranks[order] = ranks
    return unsorted_ranks

