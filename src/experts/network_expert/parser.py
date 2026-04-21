from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.experts.network_expert.constants import ATTACK_FAMILY_CLASSES


RAW_DIR_DEFAULT = Path("data/raw/cicids2018")
OUT_DIR_DEFAULT = Path("data/processed")
FEATURE_COLS_PATH_DEFAULT = OUT_DIR_DEFAULT / "feature_cols.json"
CLASS_INFO_PATH_DEFAULT = OUT_DIR_DEFAULT / "class_info.json"
META_PATH_DEFAULT = OUT_DIR_DEFAULT / "meta.json"
SCALER_PATH_DEFAULT = OUT_DIR_DEFAULT / "scaler.joblib"

BENIGN_LABELS = {"benign", "normal"}
IDENTIFIER_COLUMNS = {
    "Flow ID",
    "Src IP",
    "Source IP",
    "Dst IP",
    "Destination IP",
    "Src Port",
    "Source Port",
}


@dataclass(slots=True)
class PreprocessConfig:
    raw_dir: Path
    out_dir: Path
    sequence_length: int
    step: int
    train_ratio: float
    val_ratio: float
    label_scheme: str
    load_chunk_rows: int
    max_rows_per_file: int
    feature_cols_path: Path
    class_info_path: Path
    meta_path: Path
    scaler_path: Path


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="Prepare CICIDS2018 network sequences.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--label-scheme", choices=("family", "raw"), default="family")
    parser.add_argument("--load-chunk-rows", type=int, default=250_000)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--feature-cols-path", type=Path, default=FEATURE_COLS_PATH_DEFAULT)
    parser.add_argument("--class-info-path", type=Path, default=CLASS_INFO_PATH_DEFAULT)
    parser.add_argument("--meta-path", type=Path, default=META_PATH_DEFAULT)
    parser.add_argument("--scaler-path", type=Path, default=SCALER_PATH_DEFAULT)
    args = parser.parse_args()

    sequence_length = max(4, int(args.sequence_length))
    step = max(1, int(args.step))
    if step > sequence_length:
        step = sequence_length

    return PreprocessConfig(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        sequence_length=sequence_length,
        step=step,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        label_scheme=args.label_scheme,
        load_chunk_rows=max(10_000, int(args.load_chunk_rows)),
        max_rows_per_file=max(0, int(args.max_rows_per_file)),
        feature_cols_path=args.feature_cols_path,
        class_info_path=args.class_info_path,
        meta_path=args.meta_path,
        scaler_path=args.scaler_path,
    )


def main() -> None:
    config = parse_args()
    prepare_cicids(config)


def prepare_cicids(config: PreprocessConfig) -> None:
    print("=" * 64)
    print("  CICIDS2018 Preprocessing + Sequence Generation")
    print("=" * 64)

    csv_files = sorted(config.raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CICIDS2018 CSV files found in {config.raw_dir}")

    scaler = StandardScaler()
    observed_labels: set[str] = set()
    feature_cols: list[str] | None = None
    split_row_counts = {"train": 0, "val": 0, "test": 0}

    print("\n[Pass 1/2] Fitting scaler on train partitions")
    for csv_path in tqdm(csv_files, desc="Fitting scaler", unit="file", dynamic_ncols=True):
        frame = load_clean_cicids_file(csv_path, config)
        if frame.empty:
            continue
        observed_labels.update(frame["attack_label"].unique().tolist())
        feature_cols = resolve_feature_columns(frame, feature_cols)
        split_frames = split_frame(frame, config)
        train_features = split_frames["train"][feature_cols].to_numpy(dtype=np.float32, copy=False)
        scaler.partial_fit(train_features)
        for split_name, split_frame_df in split_frames.items():
            split_row_counts[split_name] += int(len(split_frame_df))

    if feature_cols is None:
        raise ValueError("No usable CICIDS features were found after preprocessing.")

    class_names = [label for label in ATTACK_FAMILY_CLASSES if label in observed_labels]
    if config.label_scheme == "raw":
        class_names = sorted(observed_labels, key=lambda value: (value != "Benign", value))
    if "Benign" not in class_names:
        class_names = ["Benign", *class_names]
    label_to_id = {label: idx for idx, label in enumerate(class_names)}

    print("\n[Pass 2/2] Transforming rows and building windows")
    sequence_buffers = {
        split_name: {"X": [], "y": [], "y_binary": [], "meta": []}
        for split_name in ("train", "val", "test")
    }
    for csv_path in tqdm(csv_files, desc="Building sequences", unit="file", dynamic_ncols=True):
        frame = load_clean_cicids_file(csv_path, config)
        if frame.empty:
            continue
        split_frames = split_frame(frame, config)
        for split_name, split_frame_df in split_frames.items():
            if split_frame_df.empty:
                continue
            transformed = split_frame_df.copy()
            transformed.loc[:, feature_cols] = scaler.transform(
                transformed[feature_cols].to_numpy(dtype=np.float32, copy=False)
            )
            transformed.loc[:, feature_cols] = transformed[feature_cols].clip(-8.0, 8.0)
            append_sequence_windows(
                split_buffer=sequence_buffers[split_name],
                frame=transformed,
                feature_cols=feature_cols,
                label_to_id=label_to_id,
                sequence_length=config.sequence_length,
                step=config.step,
            )

    save_outputs(
        config=config,
        sequence_buffers=sequence_buffers,
        feature_cols=feature_cols,
        class_names=class_names,
        label_to_id=label_to_id,
        scaler=scaler,
        split_row_counts=split_row_counts,
    )

    print("\nDone. Next: uv run train_cicids")


def load_clean_cicids_file(csv_path: Path, config: PreprocessConfig) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    total_rows = 0
    read_kwargs = {
        "low_memory": False,
        "on_bad_lines": "skip",
        "chunksize": config.load_chunk_rows,
    }

    try:
        reader = pd.read_csv(csv_path, encoding="utf-8", **read_kwargs)
    except UnicodeDecodeError:
        reader = pd.read_csv(csv_path, encoding="latin-1", **read_kwargs)

    for chunk in reader:
        prepared = prepare_chunk(chunk, csv_path.name, label_scheme=config.label_scheme)
        if prepared.empty:
            continue
        if config.max_rows_per_file > 0:
            remaining = config.max_rows_per_file - total_rows
            if remaining <= 0:
                break
            if len(prepared) > remaining:
                prepared = prepared.iloc[:remaining].copy()
        chunks.append(prepared)
        total_rows += int(len(prepared))
        if config.max_rows_per_file > 0 and total_rows >= config.max_rows_per_file:
            break

    if not chunks:
        return pd.DataFrame()

    frame = pd.concat(chunks, ignore_index=True)
    frame = frame.drop_duplicates(ignore_index=True)
    frame = frame.sort_values(["source_file", "timestamp"], kind="stable", na_position="last").reset_index(drop=True)
    return frame


def prepare_chunk(chunk: pd.DataFrame, source_file: str, *, label_scheme: str) -> pd.DataFrame:
    if chunk.empty:
        return pd.DataFrame()

    frame = chunk.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    label_col = resolve_label_column(frame.columns)
    if label_col is None:
        raise KeyError(f"No label column found in CICIDS file. Columns: {frame.columns.tolist()}")

    timestamp_col = resolve_timestamp_column(frame.columns)
    frame["attack_label"] = frame[label_col].map(lambda value: normalize_attack_label(value, label_scheme=label_scheme))
    frame["timestamp"] = (
        pd.to_datetime(frame[timestamp_col], errors="coerce", dayfirst=True)
        if timestamp_col is not None
        else pd.NaT
    )
    frame["source_file"] = source_file

    drop_columns = [column for column in IDENTIFIER_COLUMNS if column in frame.columns]
    drop_columns.append(label_col)
    if timestamp_col is not None:
        drop_columns.append(timestamp_col)
    frame = frame.drop(columns=drop_columns, errors="ignore")

    protected_columns = {"attack_label", "timestamp", "source_file"}
    for column in frame.columns:
        if column in protected_columns:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    numeric_columns = [column for column in frame.columns if column not in protected_columns]
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
    medians = frame[numeric_columns].median(numeric_only=True).fillna(0.0)
    frame[numeric_columns] = frame[numeric_columns].fillna(medians)
    for column in numeric_columns:
        lowered = column.lower()
        if "win_bytes" in lowered or "init_win" in lowered:
            frame[column] = frame[column].clip(lower=0)
    frame[numeric_columns] = frame[numeric_columns].astype(np.float32)
    frame = frame.dropna(subset=["attack_label"]).reset_index(drop=True)
    return frame


def resolve_label_column(columns: list[str]) -> str | None:
    normalized = {column.strip().lower(): column for column in columns}
    return normalized.get("label")


def resolve_timestamp_column(columns: list[str]) -> str | None:
    normalized = {column.strip().lower(): column for column in columns}
    for candidate in ("timestamp", "time", "datetime"):
        if candidate in normalized:
            return normalized[candidate]
    return None


def normalize_attack_label(value: object, *, label_scheme: str) -> str:
    raw = str(value).strip()
    lowered = raw.lower()
    if lowered in BENIGN_LABELS:
        return "Benign"
    if label_scheme == "raw":
        return raw
    if "ddos" in lowered:
        return "DDoS"
    if "dos" in lowered:
        return "DoS"
    if "web attack" in lowered or "sql injection" in lowered or "xss" in lowered:
        return "WebAttack"
    if "brute" in lowered or "ftp" in lowered or "ssh" in lowered:
        return "BruteForce"
    if "bot" in lowered:
        return "Botnet"
    if "infiltration" in lowered:
        return "Infiltration"
    return "OtherAttack"


def resolve_feature_columns(frame: pd.DataFrame, existing: list[str] | None) -> list[str]:
    protected_columns = {"attack_label", "timestamp", "source_file"}
    feature_cols = [column for column in frame.columns if column not in protected_columns]
    if existing is None:
        return feature_cols
    if feature_cols != existing:
        missing = sorted(set(existing) ^ set(feature_cols))
        raise ValueError(f"Inconsistent CICIDS feature schema across files. Mismatch: {missing[:10]}")
    return existing


def split_frame(frame: pd.DataFrame, config: PreprocessConfig) -> dict[str, pd.DataFrame]:
    total_rows = len(frame)
    if total_rows == 0:
        return {"train": frame.iloc[0:0], "val": frame.iloc[0:0], "test": frame.iloc[0:0]}

    train_end = max(1, int(total_rows * config.train_ratio))
    val_end = max(train_end + 1, int(total_rows * (config.train_ratio + config.val_ratio)))
    val_end = min(val_end, total_rows)

    if val_end >= total_rows:
        val_end = max(train_end, total_rows - 1)

    train_frame = frame.iloc[:train_end].reset_index(drop=True)
    val_frame = frame.iloc[train_end:val_end].reset_index(drop=True)
    test_frame = frame.iloc[val_end:].reset_index(drop=True)

    return {"train": train_frame, "val": val_frame, "test": test_frame}


def append_sequence_windows(
    *,
    split_buffer: dict[str, list[object]],
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_to_id: dict[str, int],
    sequence_length: int,
    step: int,
) -> None:
    if frame.empty:
        return

    values = frame[feature_cols].to_numpy(dtype=np.float32, copy=False)
    labels = frame["attack_label"].tolist()
    timestamps = frame["timestamp"].tolist()
    source_file = str(frame["source_file"].iloc[0])

    if len(frame) < sequence_length:
        padded = np.zeros((sequence_length, values.shape[1]), dtype=np.float32)
        padded[-len(values) :] = values
        window_label = resolve_window_label(labels)
        split_buffer["X"].append(padded)
        split_buffer["y"].append(label_to_id[window_label])
        split_buffer["y_binary"].append(0 if window_label == "Benign" else 1)
        split_buffer["meta"].append(
            {
                "source_file": source_file,
                "start_index": 0,
                "end_index": len(frame) - 1,
                "start_timestamp": timestamp_to_string(timestamps[0]),
                "end_timestamp": timestamp_to_string(timestamps[-1]),
                "window_label": window_label,
            }
        )
        return

    for start_idx in range(0, len(frame) - sequence_length + 1, step):
        end_idx = start_idx + sequence_length
        window_values = values[start_idx:end_idx]
        window_labels = labels[start_idx:end_idx]
        window_label = resolve_window_label(window_labels)
        split_buffer["X"].append(window_values.astype(np.float32, copy=False))
        split_buffer["y"].append(label_to_id[window_label])
        split_buffer["y_binary"].append(0 if window_label == "Benign" else 1)
        split_buffer["meta"].append(
            {
                "source_file": source_file,
                "start_index": start_idx,
                "end_index": end_idx - 1,
                "start_timestamp": timestamp_to_string(timestamps[start_idx]),
                "end_timestamp": timestamp_to_string(timestamps[end_idx - 1]),
                "window_label": window_label,
            }
        )


def resolve_window_label(labels: list[str]) -> str:
    attack_labels = [label for label in labels if label != "Benign"]
    if not attack_labels:
        return "Benign"
    counts = pd.Series(attack_labels).value_counts()
    return str(counts.index[0])


def timestamp_to_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def save_outputs(
    *,
    config: PreprocessConfig,
    sequence_buffers: dict[str, dict[str, list[object]]],
    feature_cols: list[str],
    class_names: list[str],
    label_to_id: dict[str, int],
    scaler: StandardScaler,
    split_row_counts: dict[str, int],
) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.feature_cols_path.parent.mkdir(parents=True, exist_ok=True)
    config.class_info_path.parent.mkdir(parents=True, exist_ok=True)
    config.meta_path.parent.mkdir(parents=True, exist_ok=True)
    config.scaler_path.parent.mkdir(parents=True, exist_ok=True)

    split_sizes: dict[str, int] = {}
    for split_name, buffer in sequence_buffers.items():
        if not buffer["X"]:
            raise ValueError(f"No sequence windows were generated for split '{split_name}'.")
        features = np.stack(buffer["X"], axis=0).astype(np.float32, copy=False)
        labels = np.asarray(buffer["y"], dtype=np.int64)
        anomaly_labels = np.asarray(buffer["y_binary"], dtype=np.int64)
        torch.save(torch.from_numpy(features), config.out_dir / f"{split_name}_X.pt")
        torch.save(torch.from_numpy(labels), config.out_dir / f"{split_name}_y.pt")
        torch.save(torch.from_numpy(anomaly_labels), config.out_dir / f"{split_name}_y_binary.pt")
        torch.save(buffer["meta"], config.out_dir / f"{split_name}_meta.pt")
        split_sizes[split_name] = int(features.shape[0])

    joblib.dump(scaler, config.scaler_path)
    config.feature_cols_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    class_info = {
        "class_names": class_names,
        "label_to_id": label_to_id,
        "seq_len": config.sequence_length,
        "step": config.step,
        "benign_label": "Benign",
        "label_scheme": config.label_scheme,
        "split_sizes": split_sizes,
    }
    config.class_info_path.write_text(json.dumps(class_info, indent=2), encoding="utf-8")

    meta = {
        "dataset": "CICIDS2018",
        "n_features": len(feature_cols),
        "seq_len": config.sequence_length,
        "step": config.step,
        "feature_columns": feature_cols,
        "class_names": class_names,
        "label_scheme": config.label_scheme,
        "row_split_sizes": split_row_counts,
        "sequence_split_sizes": split_sizes,
    }
    config.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n[Summary]")
    print(f"  Feature count : {len(feature_cols)}")
    print(f"  Classes       : {class_names}")
    print(f"  Row splits    : {split_row_counts}")
    print(f"  Seq splits    : {split_sizes}")
    print(f"\nScaler      -> {config.scaler_path}")
    print(f"Features    -> {config.feature_cols_path}")
    print(f"Class info  -> {config.class_info_path}")
    print(f"Meta        -> {config.meta_path}")


if __name__ == "__main__":
    main()
