from __future__ import annotations

import torch

from src.training.metrics import compute_classification_report


def test_compute_classification_report_fields() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.2],
            [0.2, 2.5, 1.0],
            [0.1, 0.5, 2.2],
            [2.4, 0.3, 0.2],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    class_names = ["Normal", "AttackA", "AttackB"]

    report = compute_classification_report(
        loss=0.42,
        labels=labels,
        logits=logits,
        class_names=class_names,
        normal_class_index=0,
    )

    payload = report.to_dict()
    assert payload["loss"] == 0.42
    assert 0.0 <= payload["accuracy"] <= 1.0
    assert 0.0 <= payload["macro_f1"] <= 1.0
    assert len(payload["per_class"]) == 3
    assert len(payload["confusion_matrix"]) == 3

