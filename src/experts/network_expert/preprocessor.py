from __future__ import annotations

import argparse
import csv
import json
import itertools
import re
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from src.experts.network_expert.constants import ATTACK_FAMILY_CLASSES, CANONICAL_CICIDS_15_CLASSES

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
    target_feature_count: int = 80
    add_engineered_totals: bool = True
    label_schema: str = "binary"
    min_class_support: int = 25
    rare_class_bucket_name: str = "OtherAttack"
    force_15_class_schema: bool = False
    max_windows: int | None = 30000


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
    parser.add_argument("--target-feature-count", type=int, default=80)
    parser.add_argument(
        "--add-engineered-totals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add flow-level totals to reach a stable 80-feature tensor.",
    )
    parser.add_argument(
        "--label-schema",
        type=str,
        choices=("binary", "family", "fine"),
        default="binary",
        help=(
            "Label granularity: "
            "binary (Benign/Anomaly), "
            "family (Benign + attack families), "
            "fine (canonical CICIDS attack classes)."
        ),
    )
    parser.add_argument(
        "--min-class-support",
        type=int,
        default=25,
        help="Merge attack classes/families with fewer samples than this threshold into a single bucket.",
    )
    parser.add_argument(
        "--rare-class-bucket-name",
        type=str,
        default="OtherAttack",
        help="Name of the merged bucket used when classes are below min-class-support.",
    )
    parser.add_argument(
        "--force-15-class-schema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For --label-schema fine, force all canonical 15 classes even if some have zero support.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=30000,
        help="Maximum total sequence windows kept in memory (set 0 to disable cap).",
    )
    ns = parser.parse_args()
    max_windows = ns.max_windows if ns.max_windows > 0 else None
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
        target_feature_count=ns.target_feature_count,
        add_engineered_totals=ns.add_engineered_totals,
        label_schema=ns.label_schema,
        min_class_support=max(0, int(ns.min_class_support)),
        rare_class_bucket_name=str(ns.rare_class_bucket_name),
        force_15_class_schema=ns.force_15_class_schema,
        max_windows=max_windows,
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
        if self.config.target_feature_count <= 0:
            raise ValueError("target_feature_count must be positive.")
        if self.config.label_schema not in {"binary", "family", "fine"}:
            raise ValueError("label_schema must be one of: binary, family, fine.")
        if self.config.min_class_support < 0:
            raise ValueError("min_class_support must be >= 0.")
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
        if self.config.max_windows is not None:
            print(f"Window reservoir cap enabled: keeping at most {self.config.max_windows} windows.")
        source_feature_names: list[str] | None = None
        model_feature_names: list[str] | None = None
        sequences: list[np.ndarray] = []
        window_labels_raw: list[str] = []
        rng = np.random.default_rng(self.config.seed)
        total_windows_seen = 0

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

                if source_feature_names is None:
                    source_feature_names = self._select_feature_columns(reader.fieldnames, label_column)
                    if not source_feature_names:
                        raise ValueError(f"No usable feature columns found in {csv_path}")
                    model_feature_names = self._resolve_output_feature_names(source_feature_names)
                    if not model_feature_names:
                        raise ValueError(f"No model feature names resolved for {csv_path}")

                rolling_features: deque[np.ndarray] = deque(maxlen=self.config.sequence_length)
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
                valid_rows = 0
                windows_from_file = 0
                for row in rows_bar:
                    label_text = _canonicalize_label(
                        row.get(label_column, ""),
                        label_schema=self.config.label_schema,
                        force_15_class_schema=self.config.force_15_class_schema,
                    )
                    if not label_text:
                        continue

                    if source_feature_names is None:
                        raise RuntimeError("Feature schema was not initialized.")

                    feature_vector = [
                        self._to_float(row.get(col, "0"))
                        for col in source_feature_names
                    ]
                    if self.config.add_engineered_totals:
                        feature_vector.extend(self._engineered_totals(row))
                    feature_vector = _fit_feature_vector(
                        feature_vector,
                        target_count=self.config.target_feature_count,
                    )
                    feature_array = np.asarray(feature_vector, dtype=np.float32)
                    if not np.isfinite(feature_array).all():
                        continue
                    valid_rows += 1
                    rolling_features.append(feature_array)

                    if len(rolling_features) < self.config.sequence_length:
                        continue
                    if (valid_rows - self.config.sequence_length) % self.config.stride != 0:
                        continue

                    window = np.stack(rolling_features, axis=0)
                    total_windows_seen += 1
                    windows_from_file += 1
                    self._append_window_with_reservoir(
                        sequences=sequences,
                        labels=window_labels_raw,
                        window=window,
                        label=label_text,
                        seen_windows=total_windows_seen,
                        rng=rng,
                    )
                rows_bar.close()

                if valid_rows > 0 and windows_from_file == 0:
                    stacked = np.stack(rolling_features, axis=0)
                    pad_count = self.config.sequence_length - stacked.shape[0]
                    padded = np.concatenate(
                        [
                            np.zeros((pad_count, stacked.shape[1]), dtype=np.float32),
                            stacked,
                        ],
                        axis=0,
                    )
                    total_windows_seen += 1
                    self._append_window_with_reservoir(
                        sequences=sequences,
                        labels=window_labels_raw,
                        window=padded,
                        label=label_text,
                        seen_windows=total_windows_seen,
                        rng=rng,
                    )
        files_bar.close()

        if self.config.max_windows is not None and total_windows_seen > self.config.max_windows:
            dropped = total_windows_seen - len(sequences)
            print(
                "Window sampling summary | "
                f"seen={total_windows_seen} kept={len(sequences)} dropped={dropped}"
            )

        if not sequences:
            raise ValueError("No sequence windows were produced from CICIDS CSVs.")
        if model_feature_names is None:
            raise ValueError("Feature names could not be resolved.")

        adjusted_labels = _merge_rare_classes(
            labels=window_labels_raw,
            min_support=self.config.min_class_support,
            label_schema=self.config.label_schema,
            rare_class_bucket_name=self.config.rare_class_bucket_name,
        )
        class_names = _ordered_class_names(
            adjusted_labels,
            label_schema=self.config.label_schema,
            force_15_class_schema=self.config.force_15_class_schema,
            rare_class_bucket_name=self.config.rare_class_bucket_name,
        )
        class_to_index = {name: idx for idx, name in enumerate(class_names)}
        labels = np.asarray([class_to_index[item] for item in adjusted_labels], dtype=np.int64)
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
            feature_names=model_feature_names,
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

    def _resolve_output_feature_names(self, source_feature_names: Sequence[str]) -> list[str]:
        names = list(source_feature_names)
        if self.config.add_engineered_totals:
            names.extend(["Flow Pkts Total", "Flow Bytes Total"])
        return _fit_feature_names(names, target_count=self.config.target_feature_count)

    def _engineered_totals(self, row: dict[str, object]) -> list[float]:
        fwd_pkts = self._coalesce_float(
            row,
            candidates=(
                "Tot Fwd Pkts",
                "Total Fwd Packets",
                "Total Forward Packets",
            ),
        )
        bwd_pkts = self._coalesce_float(
            row,
            candidates=(
                "Tot Bwd Pkts",
                "Total Bwd Packets",
                "Total Backward Packets",
            ),
        )
        fwd_bytes = self._coalesce_float(
            row,
            candidates=(
                "TotLen Fwd Pkts",
                "Total Length of Fwd Packets",
            ),
        )
        bwd_bytes = self._coalesce_float(
            row,
            candidates=(
                "TotLen Bwd Pkts",
                "Total Length of Bwd Packets",
            ),
        )
        return [fwd_pkts + bwd_pkts, fwd_bytes + bwd_bytes]

    def _coalesce_float(self, row: dict[str, object], *, candidates: Sequence[str]) -> float:
        for name in candidates:
            if name in row:
                return self._to_float(row.get(name, "0"))
        return 0.0

    def _append_window_with_reservoir(
        self,
        *,
        sequences: list[np.ndarray],
        labels: list[str],
        window: np.ndarray,
        label: str,
        seen_windows: int,
        rng: np.random.Generator,
    ) -> None:
        cap = self.config.max_windows
        if cap is None or len(sequences) < cap:
            sequences.append(window)
            labels.append(label)
            return

        replace_index = int(rng.integers(0, seen_windows))
        if replace_index < cap:
            sequences[replace_index] = window
            labels[replace_index] = label

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
            "label_schema": self.config.label_schema,
            "min_class_support": int(self.config.min_class_support),
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


def _canonicalize_label(
    raw: object,
    *,
    label_schema: str = "fine",
    force_15_class_schema: bool = True,
) -> str:
    label = _normalize_label(raw)
    if not label:
        return ""

    compact = re.sub(r"[^a-z0-9]+", "", label.lower())
    if compact in {"label", "labels", "class", "attack"}:
        # Skip header-like/noise rows that leak into data.
        return ""

    fine_label = _to_fine_label(compact)
    if label_schema == "binary":
        if fine_label == "Benign":
            return "Benign"
        return "Anomaly"
    if label_schema == "family":
        if fine_label == "Benign":
            return "Benign"
        if fine_label:
            return _to_family_label(fine_label)
        return "OtherAttack"
    if fine_label:
        return fine_label
    if force_15_class_schema:
        return ""
    return label


def _resolve_column(fieldnames: Iterable[str], candidates: Sequence[str]) -> str | None:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _ordered_class_names(
    labels: Sequence[str],
    *,
    label_schema: str = "fine",
    force_15_class_schema: bool,
    rare_class_bucket_name: str = "OtherAttack",
) -> list[str]:
    unique = set(labels)
    if label_schema == "binary":
        return ["Benign", "Anomaly"]
    if label_schema == "family":
        ordered = [name for name in ATTACK_FAMILY_CLASSES if name in unique]
        if force_15_class_schema:
            return list(ATTACK_FAMILY_CLASSES)
        if rare_class_bucket_name in unique and rare_class_bucket_name not in ordered:
            ordered.append(rare_class_bucket_name)
        return ordered

    if force_15_class_schema:
        base = list(CANONICAL_CICIDS_15_CLASSES)
        extras = sorted(item for item in unique if item not in CANONICAL_CICIDS_15_CLASSES)
        return base + extras

    present_in_canonical = [name for name in CANONICAL_CICIDS_15_CLASSES if name in unique]
    extras = sorted(item for item in unique if item not in CANONICAL_CICIDS_15_CLASSES)
    return present_in_canonical + extras


def _merge_rare_classes(
    *,
    labels: Sequence[str],
    min_support: int,
    label_schema: str,
    rare_class_bucket_name: str,
) -> list[str]:
    if min_support <= 1:
        return list(labels)
    if label_schema == "binary":
        return list(labels)

    counts = Counter(labels)
    protected = {"Benign", "Anomaly"}
    rare_labels = {name for name, count in counts.items() if count < min_support and name not in protected}
    if not rare_labels:
        return list(labels)

    merged: list[str] = []
    for label in labels:
        if label in rare_labels:
            merged.append(rare_class_bucket_name)
        else:
            merged.append(label)
    return merged


def _to_fine_label(compact: str) -> str:
    if compact in {"benign", "normal"} or "benign" in compact:
        return "Benign"

    if "ddos" in compact:
        if "hoic" in compact:
            return "DDoS-HOIC"
        if "loic" in compact and "udp" in compact:
            return "DDoS-LOIC-UDP"
        if "loic" in compact and "http" in compact:
            return "DDoS-LOIC-HTTP"

    if compact.startswith("dos") or any(token in compact for token in ("hulk", "goldeneye", "slowloris", "slowhttp")):
        if "hulk" in compact:
            return "DoS-Hulk"
        if "goldeneye" in compact:
            return "DoS-GoldenEye"
        if "slowloris" in compact:
            return "DoS-Slowloris"
        if "slowhttp" in compact:
            return "DoS-SlowHTTPTest"

    if "ftp" in compact and ("brute" in compact or "patator" in compact):
        return "Brute Force-FTP"
    if "ssh" in compact and ("brute" in compact or "patator" in compact):
        return "Brute Force-SSH"

    if "xss" in compact:
        return "Web Attack-XSS"
    if "sql" in compact and "inject" in compact:
        return "Web Attack-SQL Injection"
    if "web" in compact and "brute" in compact:
        return "Web Attack-Brute Force"

    if "infiltration" in compact or "infilteration" in compact:
        return "Infiltration"
    if "bot" in compact:
        return "Botnet"

    if compact.startswith("label"):
        return ""
    return ""


def _to_family_label(fine_label: str) -> str:
    if fine_label.startswith("DDoS-"):
        return "DDoS"
    if fine_label.startswith("DoS-"):
        return "DoS"
    if fine_label.startswith("Brute Force"):
        return "BruteForce"
    if fine_label.startswith("Web Attack"):
        return "WebAttack"
    if fine_label == "Botnet":
        return "Botnet"
    if fine_label == "Infiltration":
        return "Infiltration"
    if fine_label == "Benign":
        return "Benign"
    return "OtherAttack"


def _fit_feature_vector(values: Sequence[float], *, target_count: int) -> list[float]:
    current = list(values)
    if len(current) >= target_count:
        return current[:target_count]
    return current + ([0.0] * (target_count - len(current)))


def _fit_feature_names(names: Sequence[str], *, target_count: int) -> list[str]:
    current = list(names)
    if len(current) >= target_count:
        return current[:target_count]
    padding = [f"Pad Feature {idx + 1}" for idx in range(target_count - len(current))]
    return current + padding


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
