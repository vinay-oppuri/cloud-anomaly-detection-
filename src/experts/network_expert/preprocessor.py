from __future__ import annotations

import argparse
import csv
import json
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

DEFAULT_INPUT_DIR = Path("data/raw/cicids2018")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "cicids_processed.pt"


@dataclass(slots=True)
class CICIDSPreprocessConfig:
    input_dir: Path = DEFAULT_INPUT_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_path: Path = DEFAULT_OUTPUT_PATH
    sequence_length: int = 64
    stride: int = 4
    train_ratio: float = 0.75
    val_ratio: float = 0.10
    test_ratio: float = 0.15
    seed: int = 42
    max_rows_per_file: int | None = None
    max_files: int | None = None


def parse_args() -> CICIDSPreprocessConfig:
    parser = argparse.ArgumentParser(
        description="Clean CICIDS2018 CSVs and export sequence windows for CNN-LSTM."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    ns = parser.parse_args()
    return CICIDSPreprocessConfig(
        input_dir=ns.input_dir,
        output_dir=ns.output_dir,
        output_path=ns.output_path,
        sequence_length=ns.sequence_length,
        stride=ns.stride,
        train_ratio=ns.train_ratio,
        val_ratio=ns.val_ratio,
        test_ratio=ns.test_ratio,
        seed=ns.seed,
        max_rows_per_file=ns.max_rows_per_file,
        max_files=ns.max_files,
    )


class CICIDSPreprocessor:
    """Transforms CICIDS2018 CSV files into train/val/test sequence tensors."""

    _label_candidates = ("Label", "label", "Attack", "attack", "Class", "class")
    _ignored_columns = {
        "flow id",
        "src ip",
        "dst ip",
        "timestamp",
        "simillarhttp",
        "fwd header length.1",
    }

    def __init__(self, config: CICIDSPreprocessConfig) -> None:
        self.config = config
        if self.config.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if self.config.stride <= 0:
            raise ValueError("stride must be positive.")
        self._validate_ratios()

    def run(self) -> dict[str, object]:
        csv_files = sorted(path for path in self.config.input_dir.glob("*.csv") if path.is_file())
        if self.config.max_files is not None:
            if self.config.max_files <= 0:
                raise ValueError("max_files must be positive when provided.")
            csv_files = csv_files[: self.config.max_files]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.config.input_dir}")

        print(f"Preparing CICIDS from {len(csv_files)} file(s) in {self.config.input_dir}")
        feature_names: list[str] | None = None
        sequences: list[np.ndarray] = []
        window_labels_raw: list[str] = []

        files_bar = tqdm(
            csv_files,
            desc="CICIDS files",
            unit="file",
            dynamic_ncols=True,
            disable=False,
            mininterval=0.2,
        )
        for csv_path in files_bar:
            files_bar.set_postfix(current=csv_path.name)
            with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                label_column = _resolve_column(reader.fieldnames, self._label_candidates)
                if label_column is None:
                    raise KeyError(
                        f"Could not find label column in {csv_path}. "
                        f"Columns: {', '.join(reader.fieldnames)}"
                    )

                if feature_names is None:
                    feature_names = self._select_feature_columns(reader.fieldnames, label_column)
                    if not feature_names:
                        raise ValueError(f"No usable feature columns found in {csv_path}")

                file_features: list[list[float]] = []
                file_labels: list[str] = []
                row_total = self._estimate_rows(csv_path)
                if self.config.max_rows_per_file is not None:
                    row_total = min(row_total, self.config.max_rows_per_file)
                row_total_value = row_total if row_total > 0 else self.config.max_rows_per_file
                row_source = (
                    itertools.islice(reader, self.config.max_rows_per_file)
                    if self.config.max_rows_per_file is not None
                    else reader
                )
                rows_bar = tqdm(
                    row_source,
                    total=row_total_value,
                    desc=f"rows:{csv_path.name}",
                    unit="row",
                    leave=True,
                    dynamic_ncols=True,
                    disable=False,
                    mininterval=0.2,
                )
                for row in rows_bar:
                    label_text = _normalize_label(row.get(label_column, ""))
                    if not label_text:
                        continue

                    feature_vector = [self._to_float(row.get(col, "0")) for col in feature_names]
                    if not np.isfinite(feature_vector).all():
                        continue
                    file_features.append(feature_vector)
                    file_labels.append(label_text)
                rows_bar.close()

                seqs, seq_labels = self._build_windows(file_features=file_features, labels=file_labels)
                sequences.extend(seqs)
                window_labels_raw.extend(seq_labels)
        files_bar.close()

        if not sequences:
            raise ValueError("No sequence windows were produced from CICIDS CSVs.")
        if feature_names is None:
            raise ValueError("Feature names could not be resolved.")

        class_names = _ordered_class_names(window_labels_raw)
        class_to_index = {name: idx for idx, name in enumerate(class_names)}
        labels = np.asarray([class_to_index[item] for item in window_labels_raw], dtype=np.int64)
        features = np.stack(sequences, axis=0).astype(np.float32, copy=False)

        split_indices = _stratified_split_indices(
            labels=labels,
            seed=self.config.seed,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
        )

        normalized, stats = self._normalize_by_train(features=features, train_indices=split_indices["train"])
        payload = self._build_payload(
            features=normalized,
            labels=labels,
            split_indices=split_indices,
            class_names=class_names,
            feature_names=feature_names,
            normalization=stats,
        )
        self._save_payload(payload)
        return self._build_summary(payload)

    def _validate_ratios(self) -> None:
        total = self.config.train_ratio + self.config.val_ratio + self.config.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must sum to 1.0. "
                f"Got {self.config.train_ratio} + {self.config.val_ratio} + {self.config.test_ratio}."
            )
        for ratio, name in (
            (self.config.train_ratio, "train_ratio"),
            (self.config.val_ratio, "val_ratio"),
            (self.config.test_ratio, "test_ratio"),
        ):
            if ratio < 0:
                raise ValueError(f"{name} must be non-negative.")

    def _select_feature_columns(self, fieldnames: Iterable[str], label_column: str) -> list[str]:
        columns: list[str] = []
        for name in fieldnames:
            normalized = name.strip().lower()
            if normalized == label_column.strip().lower():
                continue
            if normalized in self._ignored_columns:
                continue
            columns.append(name)
        return columns

    def _build_windows(
        self,
        *,
        file_features: Sequence[Sequence[float]],
        labels: Sequence[str],
    ) -> tuple[list[np.ndarray], list[str]]:
        if not file_features:
            return [], []

        features_np = np.asarray(file_features, dtype=np.float32)
        if features_np.ndim != 2:
            return [], []

        sequences: list[np.ndarray] = []
        sequence_labels: list[str] = []
        total = features_np.shape[0]
        seq_len = self.config.sequence_length

        if total < seq_len:
            pad_count = seq_len - total
            padded = np.concatenate(
                [np.zeros((pad_count, features_np.shape[1]), dtype=np.float32), features_np],
                axis=0,
            )
            sequences.append(padded)
            sequence_labels.append(labels[-1])
            return sequences, sequence_labels

        for end_idx in range(seq_len - 1, total, self.config.stride):
            start_idx = end_idx - seq_len + 1
            window = features_np[start_idx : end_idx + 1]
            sequences.append(window)
            sequence_labels.append(labels[end_idx])

        return sequences, sequence_labels

    def _normalize_by_train(
        self,
        *,
        features: np.ndarray,
        train_indices: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if train_indices.size == 0:
            raise ValueError("Train split is empty; cannot compute normalization stats.")

        train_flat = features[train_indices].reshape(-1, features.shape[-1])
        mean = train_flat.mean(axis=0)
        std = train_flat.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)

        # Normalize in-place to avoid allocating another full [N, seq_len, feat] array.
        normalized = features.astype(np.float32, copy=False)
        normalized -= mean.reshape(1, 1, -1)
        normalized /= std.reshape(1, 1, -1)
        return normalized, {"mean": mean, "std": std}

    def _build_payload(
        self,
        *,
        features: np.ndarray,
        labels: np.ndarray,
        split_indices: dict[str, np.ndarray],
        class_names: list[str],
        feature_names: list[str],
        normalization: dict[str, np.ndarray],
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "class_names": class_names,
            "feature_names": feature_names,
            "sequence_length": self.config.sequence_length,
            "normalization": {
                "mean": torch.from_numpy(normalization["mean"].astype(np.float32)),
                "std": torch.from_numpy(normalization["std"].astype(np.float32)),
            },
            "splits": {},
        }
        for split_name in ("train", "val", "test"):
            idx = split_indices[split_name]
            payload["splits"][split_name] = {
                "X": torch.from_numpy(features[idx]),
                "y": torch.from_numpy(labels[idx]),
            }
        return payload

    def _save_payload(self, payload: dict[str, object]) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.config.output_path
        if not output_path.is_absolute():
            output_path = self.config.output_dir / output_path.name
        torch.save(payload, output_path)

        splits = payload["splits"]
        for split_name in ("train", "val", "test"):
            split = splits[split_name]
            split_out = self.config.output_dir / f"cicids_{split_name}.pt"
            torch.save({"features": split["X"], "labels": split["y"]}, split_out)

        metadata = {
            "config": {
                **asdict(self.config),
                "input_dir": str(self.config.input_dir),
                "output_dir": str(self.config.output_dir),
                "output_path": str(output_path),
            },
            "class_names": payload["class_names"],
            "feature_count": len(payload["feature_names"]),
            "sequence_length": int(payload["sequence_length"]),
            "split_sizes": {
                split_name: int(splits[split_name]["y"].shape[0])
                for split_name in ("train", "val", "test")
            },
        }
        metadata_path = self.config.output_dir / "cicids_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _build_summary(self, payload: dict[str, object]) -> dict[str, object]:
        splits = payload["splits"]
        class_names: list[str] = list(payload["class_names"])
        split_class_counts: dict[str, dict[str, int]] = {}
        for split_name in ("train", "val", "test"):
            labels = splits[split_name]["y"]
            counts = torch.bincount(labels, minlength=len(class_names))
            split_class_counts[split_name] = {
                class_names[idx]: int(counts[idx].item())
                for idx in range(len(class_names))
            }
        return {
            "task": "prepare_cicids_network_dataset",
            "output_path": str(self.config.output_path),
            "sequence_length": int(payload["sequence_length"]),
            "num_classes": len(payload["class_names"]),
            "num_features": len(payload["feature_names"]),
            "split_sizes": {
                split_name: int(splits[split_name]["y"].shape[0])
                for split_name in ("train", "val", "test")
            },
            "split_class_counts": split_class_counts,
            "classes": payload["class_names"],
        }

    def _to_float(self, value: object) -> float:
        text = str(value).strip().replace(",", "")
        if not text:
            return 0.0
        lowered = text.lower()
        if lowered in {"nan", "na", "none"}:
            return 0.0
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return 0.0
        if lowered in {"-inf", "-infinity"}:
            return 0.0
        try:
            parsed = float(text)
        except ValueError:
            return 0.0
        if not np.isfinite(parsed):
            return 0.0
        return float(parsed)

    def _estimate_rows(self, csv_path: Path) -> int:
        """
        Fast row estimate for progress/ETA.

        Counts newline-delimited records and subtracts the CSV header line.
        """
        line_count = 0
        with csv_path.open("rb") as handle:
            for _ in handle:
                line_count += 1
        return max(0, line_count - 1)


def _normalize_label(raw: object) -> str:
    return str(raw).strip().strip('"').strip("'")


def _resolve_column(fieldnames: Iterable[str], candidates: Sequence[str]) -> str | None:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _ordered_class_names(labels: Sequence[str]) -> list[str]:
    unique = sorted(set(labels))
    benign = [label for label in unique if label.strip().lower() in {"benign", "normal"}]
    others = [label for label in unique if label not in benign]
    return benign + others


def _stratified_split_indices(
    *,
    labels: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for class_value in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_value)
        shuffled = rng.permutation(class_indices)
        train_count, val_count, test_count = _split_counts(
            total=len(class_indices),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        train_parts.append(shuffled[:train_count])
        val_parts.append(shuffled[train_count : train_count + val_count])
        test_parts.append(shuffled[train_count + val_count : train_count + val_count + test_count])

    train_idx = np.concatenate(train_parts, axis=0) if train_parts else np.empty((0,), dtype=np.int64)
    val_idx = np.concatenate(val_parts, axis=0) if val_parts else np.empty((0,), dtype=np.int64)
    test_idx = np.concatenate(test_parts, axis=0) if test_parts else np.empty((0,), dtype=np.int64)
    return {
        "train": rng.permutation(train_idx),
        "val": rng.permutation(val_idx),
        "test": rng.permutation(test_idx),
    }


def _split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        train_count = 1 if train_ratio >= 0.5 else 0
        val_count = 1 - train_count
        return train_count, val_count, total - train_count - val_count

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if train_count == 0:
        train_count = 1
    if val_count == 0:
        val_count = 1
    test_count = total - train_count - val_count
    if test_count <= 0:
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1
        test_count = total - train_count - val_count
    return train_count, val_count, test_count


def main() -> None:
    config = parse_args()
    preprocessor = CICIDSPreprocessor(config=config)
    summary = preprocessor.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
