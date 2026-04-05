from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm


DEFAULT_TRACE_PATH = Path("data/raw/hdfs/Event_traces.csv")
DEFAULT_LABEL_PATH = Path("data/raw/hdfs/anomaly_label.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_CACHE_PATH = DEFAULT_OUTPUT_DIR / "hdfs_cache.json"
DEFAULT_PROCESSED_PATH = DEFAULT_OUTPUT_DIR / "hdfs_processed.pt"

LABEL_NAMES = ["Normal", "Anomaly"]


@dataclass(slots=True)
class HDFSParseConfig:
    event_trace_path: Path = DEFAULT_TRACE_PATH
    anomaly_label_path: Path = DEFAULT_LABEL_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    cache_path: Path = DEFAULT_CACHE_PATH
    processed_path: Path = DEFAULT_PROCESSED_PATH
    sequence_length: int = 128
    max_vocab_size: int = 8192
    train_ratio: float = 0.75
    val_ratio: float = 0.10
    test_ratio: float = 0.15
    seed: int = 42
    min_token_frequency: int = 2


def parse_args() -> HDFSParseConfig:
    parser = argparse.ArgumentParser(
        description="Parse HDFS Event_traces.csv + anomaly_label.csv into training-ready tensors."
    )
    parser.add_argument("--event-trace-path", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--anomaly-label-path", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=8192)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-token-frequency", type=int, default=2)
    ns = parser.parse_args()
    return HDFSParseConfig(
        event_trace_path=ns.event_trace_path,
        anomaly_label_path=ns.anomaly_label_path,
        output_dir=ns.output_dir,
        cache_path=ns.cache_path,
        processed_path=ns.processed_path,
        sequence_length=ns.sequence_length,
        max_vocab_size=ns.max_vocab_size,
        train_ratio=ns.train_ratio,
        val_ratio=ns.val_ratio,
        test_ratio=ns.test_ratio,
        seed=ns.seed,
        min_token_frequency=ns.min_token_frequency,
    )


class HDFSEventParser:
    """Converts raw HDFS traces into split tensors for transformer training."""

    _token_pattern = re.compile(r"[A-Za-z0-9_:\-/\.]+")

    def __init__(self, config: HDFSParseConfig) -> None:
        self.config = config
        if self.config.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if self.config.max_vocab_size < 2:
            raise ValueError("max_vocab_size must be at least 2.")
        if self.config.min_token_frequency <= 0:
            raise ValueError("min_token_frequency must be at least 1.")
        self._validate_ratios()

    def run(self) -> dict[str, object]:
        print(f"Preparing HDFS using {self.config.event_trace_path.name} + {self.config.anomaly_label_path.name}")
        traces = self._load_event_traces(self.config.event_trace_path)
        labels = self._load_labels(self.config.anomaly_label_path)
        merged = self._join_records(traces=traces, labels=labels)
        if not merged:
            raise ValueError("No matched HDFS records found between trace and label files.")

        all_tokens = [tokens for _, tokens, _ in merged]
        vocab = self._build_vocab(all_tokens)

        block_ids: list[str] = []
        encoded_rows: list[np.ndarray] = []
        targets: list[int] = []
        for block_id, tokens, label in tqdm(
            merged,
            desc="Encoding HDFS sequences",
            unit="row",
            dynamic_ncols=True,
            disable=False,
            mininterval=0.2,
        ):
            block_ids.append(block_id)
            encoded_rows.append(self._encode_tokens(tokens=tokens, vocab=vocab))
            targets.append(label)

        features = np.stack(encoded_rows, axis=0).astype(np.int64, copy=False)
        labels_array = np.asarray(targets, dtype=np.int64)

        split_indices = self._stratified_split_indices(labels_array)
        split_payload = self._build_split_payload(
            features=features,
            labels=labels_array,
            block_ids=block_ids,
            split_indices=split_indices,
            vocab_size=len(vocab),
        )
        self._save_outputs(payload=split_payload, vocab=vocab)
        return self._build_summary(payload=split_payload, vocab=vocab)

    def _validate_ratios(self) -> None:
        total = self.config.train_ratio + self.config.val_ratio + self.config.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must sum to 1.0. "
                f"Got {self.config.train_ratio} + {self.config.val_ratio} + {self.config.test_ratio}."
            )
        for value, name in (
            (self.config.train_ratio, "train_ratio"),
            (self.config.val_ratio, "val_ratio"),
            (self.config.test_ratio, "test_ratio"),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")

    def _load_event_traces(self, path: Path) -> dict[str, list[str]]:
        if not path.exists():
            raise FileNotFoundError(f"HDFS event trace file not found: {path}")

        traces: dict[str, list[str]] = {}
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Missing CSV header in {path}")
            block_key = _resolve_column(
                reader.fieldnames,
                candidates=("BlockId", "blockid", "block_id", "blk_id", "blkid"),
            )
            sequence_key = _resolve_column(
                reader.fieldnames,
                candidates=(
                    "EventSequence",
                    "event_sequence",
                    "events",
                    "sequence",
                    "Features",
                    "features",
                ),
            )
            if block_key is None or sequence_key is None:
                raise KeyError(
                    f"Could not resolve BlockId/EventSequence columns in {path}. "
                    f"Available columns: {', '.join(reader.fieldnames)}"
                )

            rows_bar = tqdm(
                reader,
                desc="Reading Event_traces.csv",
                unit="row",
                dynamic_ncols=True,
                disable=False,
                mininterval=0.2,
            )
            for row in rows_bar:
                block_raw = row.get(block_key, "")
                sequence_raw = row.get(sequence_key, "")
                block_id = _normalize_block_id(block_raw)
                if not block_id:
                    continue
                tokens = self._tokenize_sequence(sequence_raw)
                if not tokens:
                    continue
                traces[block_id] = tokens
            rows_bar.close()

        return traces

    def _load_labels(self, path: Path) -> dict[str, int]:
        if not path.exists():
            raise FileNotFoundError(f"HDFS label file not found: {path}")

        labels: dict[str, int] = {}
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Missing CSV header in {path}")

            block_key = _resolve_column(
                reader.fieldnames,
                candidates=("BlockId", "blockid", "block_id", "blk_id", "blkid"),
            )
            label_key = _resolve_column(
                reader.fieldnames,
                candidates=("Label", "label", "anomaly", "is_anomaly", "class"),
            )
            if block_key is None or label_key is None:
                raise KeyError(
                    f"Could not resolve BlockId/Label columns in {path}. "
                    f"Available columns: {', '.join(reader.fieldnames)}"
                )

            rows_bar = tqdm(
                reader,
                desc="Reading anomaly_label.csv",
                unit="row",
                dynamic_ncols=True,
                disable=False,
                mininterval=0.2,
            )
            for row in rows_bar:
                block_id = _normalize_block_id(row.get(block_key, ""))
                if not block_id:
                    continue
                label = _parse_label(row.get(label_key, ""))
                if label is None:
                    continue
                labels[block_id] = label
            rows_bar.close()

        return labels

    def _join_records(
        self,
        *,
        traces: dict[str, list[str]],
        labels: dict[str, int],
    ) -> list[tuple[str, list[str], int]]:
        merged: list[tuple[str, list[str], int]] = []
        for block_id, tokens in traces.items():
            if block_id not in labels:
                continue
            merged.append((block_id, tokens, labels[block_id]))
        merged.sort(key=lambda item: item[0])
        return merged

    def _build_vocab(self, token_lists: Sequence[Sequence[str]]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for tokens in token_lists:
            counts.update(tokens)

        vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        for token, freq in counts.most_common():
            if len(vocab) >= self.config.max_vocab_size:
                break
            if freq < self.config.min_token_frequency:
                continue
            vocab[token] = len(vocab)
        return vocab

    def _encode_tokens(self, *, tokens: Sequence[str], vocab: dict[str, int]) -> np.ndarray:
        ids = [vocab.get(token, 1) for token in tokens]
        if len(ids) > self.config.sequence_length:
            ids = ids[-self.config.sequence_length :]
        else:
            ids = ([0] * (self.config.sequence_length - len(ids))) + ids
        return np.asarray(ids, dtype=np.int64)

    def _tokenize_sequence(self, raw: str) -> list[str]:
        return [token.lower() for token in self._token_pattern.findall(str(raw))]

    def _stratified_split_indices(self, labels: np.ndarray) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.config.seed)
        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []

        for class_value in np.unique(labels):
            class_indices = np.flatnonzero(labels == class_value)
            shuffled = rng.permutation(class_indices)
            train_count, val_count, test_count = _split_counts(
                total=len(class_indices),
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
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

    def _build_split_payload(
        self,
        *,
        features: np.ndarray,
        labels: np.ndarray,
        block_ids: Sequence[str],
        split_indices: dict[str, np.ndarray],
        vocab_size: int,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "class_names": LABEL_NAMES,
            "sequence_length": self.config.sequence_length,
            "vocab_size": int(vocab_size),
            "splits": {},
        }

        for split_name in ("train", "val", "test"):
            idx = split_indices[split_name]
            split_X = torch.from_numpy(features[idx])
            split_y = torch.from_numpy(labels[idx])
            split_blocks = [block_ids[int(i)] for i in idx]
            casted = {
                "X": split_X,
                "y": split_y,
                "block_ids": split_blocks,
            }
            payload["splits"][split_name] = casted

        return payload

    def _save_outputs(self, *, payload: dict[str, object], vocab: dict[str, int]) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        processed_path = self.config.processed_path
        if not processed_path.is_absolute():
            processed_path = self.config.output_dir / processed_path.name
        torch.save(payload, processed_path)

        split_payload = payload["splits"]
        for split_name in ("train", "val", "test"):
            split = split_payload[split_name]
            split_out = self.config.output_dir / f"hdfs_{split_name}.pt"
            torch.save({"features": split["X"], "labels": split["y"]}, split_out)

        cache = {
            "config": {
                **asdict(self.config),
                "event_trace_path": str(self.config.event_trace_path),
                "anomaly_label_path": str(self.config.anomaly_label_path),
                "output_dir": str(self.config.output_dir),
                "cache_path": str(self.config.cache_path),
                "processed_path": str(processed_path),
            },
            "label_names": LABEL_NAMES,
            "vocab_size": len(vocab),
            "vocab": vocab,
            "split_sizes": {
                split_name: int(split_payload[split_name]["y"].shape[0])
                for split_name in ("train", "val", "test")
            },
            "artifacts": {
                "processed_bundle": str(processed_path),
                "train_file": str(self.config.output_dir / "hdfs_train.pt"),
                "val_file": str(self.config.output_dir / "hdfs_val.pt"),
                "test_file": str(self.config.output_dir / "hdfs_test.pt"),
            },
        }
        self.config.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    def _build_summary(self, *, payload: dict[str, object], vocab: dict[str, int]) -> dict[str, object]:
        split_payload = payload["splits"]
        return {
            "task": "prepare_hdfs_system_dataset",
            "processed_bundle": str(self.config.processed_path),
            "cache_path": str(self.config.cache_path),
            "sequence_length": self.config.sequence_length,
            "vocab_size": len(vocab),
            "class_names": LABEL_NAMES,
            "split_sizes": {
                split_name: int(split_payload[split_name]["y"].shape[0])
                for split_name in ("train", "val", "test")
            },
        }


def _resolve_column(fieldnames: Iterable[str], candidates: Sequence[str]) -> str | None:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _normalize_block_id(raw: object) -> str:
    return str(raw).strip().strip('"').strip("'")


def _parse_label(raw: object) -> int | None:
    normalized = str(raw).strip().strip('"').strip("'").lower()
    if not normalized:
        return None
    if normalized in {"normal", "0", "false", "no", "benign"}:
        return 0
    if normalized in {"anomaly", "anomalous", "1", "true", "yes", "attack"}:
        return 1
    return None


def _split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        train_count = 1 if train_ratio >= 0.5 else 0
        val_count = 1 - train_count
        return train_count, val_count, 1 - train_count - val_count

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    used = train_count + val_count
    test_count = total - used

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
    parser = HDFSEventParser(config=config)
    summary = parser.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
