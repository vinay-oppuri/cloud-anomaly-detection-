from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.experts.system_expert.drain import DrainLikeParser, DrainLikeParserConfig, extract_block_ids


RAW_LOG_DEFAULT = Path("data/raw/hdfs/HDFS.log")
TRACE_PATH_DEFAULT = Path("data/raw/hdfs/Event_traces.csv")
LABEL_PATH_DEFAULT = Path("data/raw/hdfs/anomaly_label.csv")
OUT_DIR_DEFAULT = Path("data/processed")
PROCESSED_PATH_DEFAULT = OUT_DIR_DEFAULT / "hdfs_processed.pt"
CACHE_PATH_DEFAULT = OUT_DIR_DEFAULT / "hdfs_cache.json"
PARSER_STATE_PATH_DEFAULT = OUT_DIR_DEFAULT / "hdfs_parser_state.json"

CLASS_NAMES = ["Normal", "Anomaly"]
EVENT_TOKEN_PATTERN = re.compile(r"\bE\d+\b", flags=re.IGNORECASE)
RAW_HDFS_LINE_PATTERN = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+\d+\s+(?P<level>[A-Z]+)\s+(?P<component>[^:]+):\s+(?P<message>.*)$"
)


@dataclass(slots=True)
class PreprocessConfig:
    source: str
    raw_log_path: Path
    trace_path: Path
    label_path: Path
    out_dir: Path
    processed_path: Path
    cache_path: Path
    parser_state_path: Path
    sequence_length: int
    train_ratio: float
    val_ratio: float
    seed: int
    similarity_threshold: float


@dataclass(slots=True)
class BlockSequenceSample:
    block_id: str
    template_ids: list[str]
    label: int
    source: str


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="Prepare HDFS log sequences for anomaly detection.")
    parser.add_argument("--source", choices=("auto", "raw", "trace"), default="auto")
    parser.add_argument("--raw-log-path", type=Path, default=RAW_LOG_DEFAULT)
    parser.add_argument("--trace-path", type=Path, default=TRACE_PATH_DEFAULT)
    parser.add_argument("--label-path", type=Path, default=LABEL_PATH_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED_PATH_DEFAULT)
    parser.add_argument("--cache-path", type=Path, default=CACHE_PATH_DEFAULT)
    parser.add_argument("--parser-state-path", type=Path, default=PARSER_STATE_PATH_DEFAULT)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    args = parser.parse_args()

    return PreprocessConfig(
        source=args.source,
        raw_log_path=args.raw_log_path,
        trace_path=args.trace_path,
        label_path=args.label_path,
        out_dir=args.out_dir,
        processed_path=args.processed_path,
        cache_path=args.cache_path,
        parser_state_path=args.parser_state_path,
        sequence_length=max(8, int(args.sequence_length)),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        similarity_threshold=float(args.similarity_threshold),
    )


def main() -> None:
    config = parse_args()
    prepare_hdfs(config)


def prepare_hdfs(config: PreprocessConfig) -> None:
    print("=" * 64)
    print("  HDFS Log Parsing + Sequence Generation")
    print("=" * 64)

    labels = load_labels(config.label_path)
    source_mode = resolve_source_mode(config)

    if source_mode == "raw":
        parser = DrainLikeParser(
            DrainLikeParserConfig(similarity_threshold=config.similarity_threshold)
        )
        samples, parser_state = parse_raw_hdfs_logs(config.raw_log_path, labels, parser)
    else:
        samples, parser_state = load_event_trace_sequences(config.trace_path, labels)

    if not samples:
        raise ValueError("No HDFS block sequences were produced from the configured inputs.")

    vocab = build_vocab(samples)
    encoded = encode_samples(samples, vocab=vocab, sequence_length=config.sequence_length)
    split_payload = stratified_split(
        encoded=encoded,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    save_outputs(
        config=config,
        split_payload=split_payload,
        vocab=vocab,
        parser_state=parser_state,
        source_mode=source_mode,
    )

    print("\nDone. Next: uv run train_hdfs")


def resolve_source_mode(config: PreprocessConfig) -> str:
    if config.source == "raw":
        if not config.raw_log_path.exists():
            raise FileNotFoundError(f"Raw HDFS log not found: {config.raw_log_path}")
        return "raw"
    if config.source == "trace":
        if not config.trace_path.exists():
            raise FileNotFoundError(f"Event trace file not found: {config.trace_path}")
        return "trace"
    if config.raw_log_path.exists():
        return "raw"
    if config.trace_path.exists():
        return "trace"
    raise FileNotFoundError(
        f"Neither raw log ({config.raw_log_path}) nor event traces ({config.trace_path}) were found."
    )


def load_labels(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"HDFS label file not found: {path}")

    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Could not read headers from {path}.")
        block_col = resolve_column(reader.fieldnames, ("BlockId", "block_id", "blk_id", "blkid"))
        label_col = resolve_column(reader.fieldnames, ("Label", "label", "class"))
        if block_col is None or label_col is None:
            raise KeyError(f"Expected BlockId/Label columns in {path}. Found: {reader.fieldnames}")

        for row in tqdm(reader, desc="Loading anomaly labels", unit="row", dynamic_ncols=True):
            block_id = normalize_block_id(row.get(block_col, ""))
            label = normalize_label(row.get(label_col, ""))
            if block_id and label is not None:
                labels[block_id] = label
    return labels


def parse_raw_hdfs_logs(
    path: Path,
    labels: dict[str, int],
    parser: DrainLikeParser,
) -> tuple[list[BlockSequenceSample], dict[str, object]]:
    print(f"\n[Source] Raw HDFS log: {path}")
    block_sequences: dict[str, list[str]] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in tqdm(handle, desc="Parsing raw HDFS logs", unit="line", dynamic_ncols=True):
            message = extract_message(line)
            if not message:
                continue
            block_ids = extract_block_ids(message)
            if not block_ids:
                continue
            cluster = parser.parse_message(message)
            for block_id in block_ids:
                block_sequences.setdefault(block_id, []).append(cluster.template_id)

    samples: list[BlockSequenceSample] = []
    for block_id, template_ids in sorted(block_sequences.items()):
        if block_id not in labels or not template_ids:
            continue
        samples.append(
            BlockSequenceSample(
                block_id=block_id,
                template_ids=template_ids,
                label=labels[block_id],
                source="raw",
            )
        )

    parser_state = parser.export_state()
    parser_state["source"] = "raw"
    parser_state["matched_blocks"] = len(samples)
    parser_state["labeled_blocks"] = len(labels)
    return samples, parser_state


def load_event_trace_sequences(
    path: Path,
    labels: dict[str, int],
) -> tuple[list[BlockSequenceSample], dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"HDFS event trace file not found: {path}")

    print(f"\n[Source] Event traces: {path}")
    samples: list[BlockSequenceSample] = []
    template_vocab: set[str] = set()

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Could not read headers from {path}.")
        block_col = resolve_column(reader.fieldnames, ("BlockId", "block_id", "blk_id", "blkid"))
        seq_col = resolve_column(
            reader.fieldnames,
            ("Features", "EventSequence", "event_sequence", "sequence", "events"),
        )
        if block_col is None or seq_col is None:
            raise KeyError(
                f"Expected BlockId/Features columns in {path}. Found: {reader.fieldnames}"
            )

        for row in tqdm(reader, desc="Loading event traces", unit="row", dynamic_ncols=True):
            block_id = normalize_block_id(row.get(block_col, ""))
            if block_id not in labels:
                continue
            template_ids = [token.upper() for token in EVENT_TOKEN_PATTERN.findall(str(row.get(seq_col, "")))]
            if not template_ids:
                continue
            template_vocab.update(template_ids)
            samples.append(
                BlockSequenceSample(
                    block_id=block_id,
                    template_ids=template_ids,
                    label=labels[block_id],
                    source="trace",
                )
            )

    parser_state = {
        "parser": "prebuilt_event_traces",
        "source": "trace",
        "matched_blocks": len(samples),
        "labeled_blocks": len(labels),
        "templates": [
            {
                "template_id": template_id,
                "template": template_id,
                "template_tokens": [template_id.lower()],
                "occurrences": None,
            }
            for template_id in sorted(template_vocab)
        ],
    }
    return samples, parser_state


def extract_message(line: str) -> str:
    match = RAW_HDFS_LINE_PATTERN.match(str(line).strip())
    if match is not None:
        return match.group("message").strip()
    return str(line).strip()


def resolve_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {field.strip().lower(): field for field in fieldnames}
    for candidate in candidates:
        match = normalized.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def normalize_block_id(value: str) -> str:
    return str(value).strip().strip('"').strip("'").lower()


def normalize_label(value: str) -> int | None:
    text = str(value).strip().strip('"').strip("'").lower()
    if text in {"normal", "0", "false", "benign", "success"}:
        return 0
    if text in {"anomaly", "anomalous", "1", "true", "fail", "failure"}:
        return 1
    return None


def build_vocab(samples: list[BlockSequenceSample]) -> dict[str, int]:
    unique_templates = sorted({template_id for sample in samples for template_id in sample.template_ids})
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for template_id in unique_templates:
        vocab[template_id.lower()] = len(vocab)
    return vocab


def encode_samples(
    samples: list[BlockSequenceSample],
    *,
    vocab: dict[str, int],
    sequence_length: int,
) -> dict[str, np.ndarray | list[str]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    block_ids: list[str] = []
    lengths: list[int] = []
    sources: list[str] = []

    for sample in tqdm(samples, desc="Encoding HDFS sequences", unit="seq", dynamic_ncols=True):
        token_ids = [vocab.get(template_id.lower(), vocab["<UNK>"]) for template_id in sample.template_ids]
        lengths.append(min(len(token_ids), sequence_length))
        if len(token_ids) >= sequence_length:
            token_ids = token_ids[-sequence_length:]
        else:
            token_ids = ([vocab["<PAD>"]] * (sequence_length - len(token_ids))) + token_ids
        features.append(np.asarray(token_ids, dtype=np.int64))
        labels.append(sample.label)
        block_ids.append(sample.block_id)
        sources.append(sample.source)

    return {
        "X": np.stack(features, axis=0),
        "y": np.asarray(labels, dtype=np.int64),
        "block_ids": block_ids,
        "lengths": np.asarray(lengths, dtype=np.int64),
        "sources": sources,
    }


def stratified_split(
    *,
    encoded: dict[str, np.ndarray | list[str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, dict[str, object]]:
    labels = np.asarray(encoded["y"], dtype=np.int64)
    indices = np.arange(labels.shape[0])
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )
    temp_labels = labels[temp_idx]
    val_fraction = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_fraction),
        random_state=seed,
        stratify=temp_labels,
    )

    def make_split(split_idx: np.ndarray) -> dict[str, object]:
        return {
            "X": torch.from_numpy(np.asarray(encoded["X"])[split_idx]),
            "y": torch.from_numpy(labels[split_idx]),
            "block_ids": [encoded["block_ids"][int(i)] for i in split_idx],
            "lengths": torch.from_numpy(np.asarray(encoded["lengths"])[split_idx]),
            "sources": [encoded["sources"][int(i)] for i in split_idx],
        }

    return {
        "train": make_split(train_idx),
        "val": make_split(val_idx),
        "test": make_split(test_idx),
    }


def save_outputs(
    *,
    config: PreprocessConfig,
    split_payload: dict[str, dict[str, object]],
    vocab: dict[str, int],
    parser_state: dict[str, object],
    source_mode: str,
) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.processed_path.parent.mkdir(parents=True, exist_ok=True)
    config.cache_path.parent.mkdir(parents=True, exist_ok=True)
    config.parser_state_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "class_names": CLASS_NAMES,
        "sequence_length": config.sequence_length,
        "vocab_size": len(vocab),
        "splits": split_payload,
    }
    torch.save(bundle, config.processed_path)

    config.parser_state_path.write_text(json.dumps(parser_state, indent=2), encoding="utf-8")
    cache = {
        "class_names": CLASS_NAMES,
        "sequence_length": config.sequence_length,
        "vocab_size": len(vocab),
        "vocab": vocab,
        "processed_path": str(config.processed_path),
        "parser_state_path": str(config.parser_state_path),
        "source_mode": source_mode,
        "split_sizes": {
            split_name: int(split_payload[split_name]["y"].shape[0])
            for split_name in ("train", "val", "test")
        },
    }
    config.cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    label_counts = {
        split_name: {
            "normal": int((split_payload[split_name]["y"] == 0).sum().item()),
            "anomaly": int((split_payload[split_name]["y"] == 1).sum().item()),
        }
        for split_name in ("train", "val", "test")
    }

    print("\n[Summary]")
    print(f"  Source mode : {source_mode}")
    print(f"  Vocab size  : {len(vocab):,}")
    for split_name in ("train", "val", "test"):
        split_size = int(split_payload[split_name]["y"].shape[0])
        counts = label_counts[split_name]
        print(
            f"  {split_name:>5} : {split_size:,} "
            f"(normal={counts['normal']:,}, anomaly={counts['anomaly']:,})"
        )
    print(f"\nProcessed -> {config.processed_path}")
    print(f"Cache     -> {config.cache_path}")
    print(f"Parser    -> {config.parser_state_path}")


if __name__ == "__main__":
    main()
