"""
process.py
==========
Preprocesses CSE-CICIDS2018 for binary anomaly detection.
  Label: 0 = Benign (Normal)
         1 = Anomaly (any attack type)

Input  : data/raw/cicids2018/*.csv
Output : data/processed/
           train_X.pt, train_y.pt
           val_X.pt,   val_y.pt
           test_X.pt,  test_y.pt
           scaler.joblib
           meta.json

Run:
  uv run prepare_cicids
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
RAW_DIR  = "data/raw/cicids2018"
OUT_DIR  = "data/processed"
SEED     = 42
SEQ_LEN  = 20   # consecutive flows per sequence
CLEAN_DEDUP_CHUNK_ROWS = int(os.environ.get("CICIDS_CLEAN_DEDUP_CHUNK_ROWS", "250000"))
LOAD_CHUNK_ROWS = int(os.environ.get("CICIDS_LOAD_CHUNK_ROWS", "250000"))
MAX_ROWS = int(os.environ.get("CICIDS_MAX_ROWS", "0"))  # 0 = no cap

# ── Drop socket/identity columns — not behavioral features ─────────
DROP_COLS = [
    "Flow ID", "Src IP", "Source IP",
    "Dst IP",  "Destination IP",
    "Src Port","Source Port",
    "Dst Port","Destination Port",
    "Timestamp",
    "Protocol",
]

# ── All attack labels → 1, Benign → 0 ─────────────────────────────
BENIGN_LABELS = {"benign", "BENIGN", "Benign"}


def load_csvs(raw_dir):
    files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".csv")])
    if not files:
        raise FileNotFoundError(
            f"No CSV files in {raw_dir}\n"
            "Download: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv"
        )

    dfs: list[pd.DataFrame] = []
    running_rows = 0
    print(f"[Load] {len(files)} CSV files found")
    for fname in tqdm(files, desc="[Load] Reading CSVs", unit="file", dynamic_ncols=True):
        path = os.path.join(raw_dir, fname)
        file_rows = 0
        file_kept = 0

        read_kwargs = {
            "low_memory": False,
            "on_bad_lines": "skip",
            "chunksize": max(10_000, LOAD_CHUNK_ROWS),
        }
        try:
            reader = pd.read_csv(path, encoding="utf-8", **read_kwargs)
        except Exception:
            reader = pd.read_csv(path, encoding="latin-1", **read_kwargs)

        for chunk in reader:
            file_rows += len(chunk)
            chunk = _prepare_loaded_chunk(chunk)
            if chunk.empty:
                continue

            if MAX_ROWS > 0:
                remaining = MAX_ROWS - running_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.sample(n=remaining, random_state=SEED)

            dfs.append(chunk)
            kept = len(chunk)
            file_kept += kept
            running_rows += kept

            if MAX_ROWS > 0 and running_rows >= MAX_ROWS:
                break

        print(f"  {fname}: {file_rows:,} rows read | {file_kept:,} rows kept")
        if MAX_ROWS > 0 and running_rows >= MAX_ROWS:
            print(f"  Reached max row cap ({MAX_ROWS:,}).")
            break

    if not dfs:
        raise RuntimeError("No usable data rows were loaded from CSV files.")
    combined = pd.concat(dfs, ignore_index=True, copy=False)
    print(f"[Load] Total: {len(combined):,} rows\n")
    return combined


def find_label_col(df):
    for c in df.columns:
        if c.strip().lower() == "label":
            return c
    raise ValueError(f"No label column found. Columns: {df.columns.tolist()}")


def _prepare_loaded_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Early-clean each chunk so we never keep large object/string columns in memory.
    """
    if df.empty:
        return df

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    lbl_col = find_label_col(df)
    if lbl_col != "Label":
        df = df.rename(columns={lbl_col: "Label"})
    label_series = df["Label"].astype(str).str.strip()
    df["label_bin"] = np.where(label_series.isin(BENIGN_LABELS), 0, 1).astype(np.int8)

    to_drop = [c for c in DROP_COLS if c in df.columns] + ["Label"]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")

    feature_cols = [c for c in df.columns if c != "label_bin"]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # fill_non_numeric handled by to_numeric and fillna later

    # Use compact dtypes early to lower memory pressure.
    for col in df.columns:
        if col == "label_bin":
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(np.int32)

    return df


def _deduplicate_with_hash_progress(df: pd.DataFrame, chunk_rows: int) -> pd.DataFrame:
    """
    Chunked duplicate removal with progress and Ctrl+C responsiveness.
    Uses row hashes; collision risk is negligible for this use case.
    """
    n_rows = len(df)
    if n_rows <= 0:
        return df

    chunk_rows = max(10_000, int(chunk_rows))
    n_chunks = (n_rows + chunk_rows - 1) // chunk_rows
    seen_hashes: set[int] = set()
    keep_masks: list[np.ndarray] = []

    for start in tqdm(
        range(0, n_rows, chunk_rows),
        total=n_chunks,
        desc="[Clean] Deduplicating",
        unit="chunk",
        dynamic_ncols=True,
    ):
        end = min(start + chunk_rows, n_rows)
        chunk = df.iloc[start:end]
        hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype=np.uint64, copy=False)
        h_series = pd.Series(hashes, copy=False)

        keep_local = ~h_series.duplicated(keep="first").to_numpy()
        if seen_hashes:
            unseen_global = ~h_series.isin(seen_hashes).to_numpy()
            keep_local &= unseen_global

        if keep_local.any():
            seen_hashes.update(map(int, hashes[keep_local]))
        keep_masks.append(keep_local)

    keep_mask = np.concatenate(keep_masks, axis=0) if keep_masks else np.zeros((n_rows,), dtype=bool)
    return df.loc[keep_mask].reset_index(drop=True)


def clean(df):
    print("[Clean] Starting...")
    steps_total = 6
    pbar = tqdm(total=steps_total, desc="[Clean] Steps", unit="step", dynamic_ncols=True)

    # Find/create binary label.
    if "label_bin" not in df.columns:
        lbl_col = find_label_col(df)
        df = df.rename(columns={lbl_col: "Label"})
        df["Label"] = df["Label"].astype(str).str.strip()
        pbar.update(1)
        df["label_bin"] = df["Label"].apply(
            lambda x: 0 if x in BENIGN_LABELS else 1
        )
        pbar.update(1)
    else:
        # If chunks were preprocessed in loader, label_bin already exists.
        df["label_bin"] = pd.to_numeric(df["label_bin"], errors="coerce").fillna(1).astype(np.int8)
        pbar.update(2)

    n_benign = (df["label_bin"] == 0).sum()
    n_attack = (df["label_bin"] == 1).sum()
    print(f"  Benign : {n_benign:,}  ({100*n_benign/len(df):.1f}%)")
    print(f"  Attack : {n_attack:,}  ({100*n_attack/len(df):.1f}%)")

    # Drop identifier columns
    to_drop = [c for c in DROP_COLS if c in df.columns]
    if "Label" in df.columns:
        to_drop.append("Label")
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
    pbar.update(1)

    # Replace inf -> NaN -> median (column-wise to avoid huge temporary allocations)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "label_bin" in num_cols:
        num_cols.remove("label_bin")
    for col in num_cols:
        col_values = df[col]
        col_values = col_values.replace([np.inf, -np.inf], np.nan)
        median_value = col_values.median()
        if pd.isna(median_value):
            median_value = 0.0
        df[col] = col_values.fillna(median_value)
    pbar.update(1)

    # Clip negatives in win_bytes columns (CICFlowMeter bug)
    for c in df.columns:
        if "win_bytes" in c.lower() or "init_win" in c.lower():
            df[c] = df[c].clip(lower=0)
    pbar.update(1)

    # Remove duplicates
    before = len(df)
    try:
        df = _deduplicate_with_hash_progress(df, chunk_rows=CLEAN_DEDUP_CHUNK_ROWS)
    except KeyboardInterrupt:
        pbar.close()
        raise
    except MemoryError:
        print("  Dedup skipped due memory pressure; continuing without duplicate removal.")
    except np.core._exceptions._ArrayMemoryError:
        print("  Dedup skipped due memory pressure; continuing without duplicate removal.")
    print(f"  Removed {before-len(df):,} duplicates")
    print(f"  Clean rows: {len(df):,}\n")
    pbar.update(1)
    pbar.close()
    return df


def scale(df):
    print("[Scale] Applying StandardScaler...")
    feat_df = df.drop(columns=["label_bin"], errors="ignore").copy()

    # Coerce any unexpected string/object columns to numeric.
    for col in feat_df.columns:
        if not pd.api.types.is_numeric_dtype(feat_df[col]):
            feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")

    # Disable dropping feature columns in this pass

    if feat_df.shape[1] == 0:
        raise ValueError("No numeric feature columns available after preprocessing.")

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.fillna(feat_df.median(numeric_only=True)).fillna(0)

    feat_cols = feat_df.columns.tolist()
    X = feat_df.to_numpy(dtype=np.float32, copy=False)
    y = df["label_bin"].values.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.clip(X, -5, 5)   # clip extreme outliers gracefully for transformer LayerNorm
    print(f"  X shape: {X.shape}")
    return X, y, feat_cols, scaler


def build_sequences(X, y, seq_len):
    """
    Sliding window of seq_len flows.
    Label = anomaly if any flow in the window is anomalous.
    This avoids missing attack windows where the last flow is benign.
    Step  = seq_len // 2  (50% overlap for more training samples)
    """
    print(f"[Seq] Building sequences (len={seq_len}, step={seq_len//2})...")
    step = seq_len // 2
    Xs, ys = [], []
    n_windows = max(0, (len(X) - seq_len) // step + 1)
    for i in tqdm(
        range(0, len(X) - seq_len + 1, step),
        total=n_windows,
        desc="[Seq] Sliding windows",
        unit="win",
        dynamic_ncols=True,
    ):
        Xs.append(X[i:i+seq_len])
        ys.append(int(np.max(y[i:i+seq_len])))
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.int64)
    n_anom = ys.sum()
    print(f"  Sequences : {len(Xs):,}")
    print(f"  Normal    : {len(ys)-n_anom:,}  ({100*(len(ys)-n_anom)/len(ys):.1f}%)")
    print(f"  Anomaly   : {n_anom:,}  ({100*n_anom/len(ys):.1f}%)\n")
    return Xs, ys


def split_save(Xs, ys, feat_cols, scaler, out_dir):
    print("[Split] 70% train / 15% val / 15% test (stratified)...")
    os.makedirs(out_dir, exist_ok=True)

    X_tv, X_te, y_tv, y_te = train_test_split(
        Xs, ys, test_size=0.15, random_state=SEED, stratify=ys)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=0.15/0.85, random_state=SEED, stratify=y_tv)

    print(f"  Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,}")

    splits = [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]
    for name, X, y in tqdm(splits, desc="[Split] Saving tensors", unit="split", dynamic_ncols=True):
        torch.save(torch.tensor(X), os.path.join(out_dir, f"{name}_X.pt"))
        torch.save(torch.tensor(y), os.path.join(out_dir, f"{name}_y.pt"))

    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    # pos_weight for BCEWithLogitsLoss
    n_norm  = float((y_tr == 0).sum())
    n_anom  = float((y_tr == 1).sum())
    pos_w   = n_norm / max(n_anom, 1)

    meta = {
        "n_features"  : Xs.shape[2],
        "seq_len"     : SEQ_LEN,
        "pos_weight"  : pos_w,
        "n_train"     : len(X_tr),
        "n_val"       : len(X_va),
        "n_test"      : len(X_te),
        "feat_cols"   : feat_cols,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  pos_weight = {pos_w:.2f}")
    print(f"  Saved to {out_dir}/\n")


def main():
    print("=" * 55)
    print("  CICIDS2018 Binary Anomaly Detection — Preprocessing")
    print("=" * 55 + "\n")

    try:
        df  = load_csvs(RAW_DIR)
        df  = clean(df)
        X, y, feat_cols, scaler = scale(df)
        Xs, ys = build_sequences(X, y, SEQ_LEN)
        split_save(Xs, ys, feat_cols, scaler, OUT_DIR)
        print("Done! Next: uv run train_cicids")
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Preprocessing stopped cleanly.")


if __name__ == "__main__":
    main()
