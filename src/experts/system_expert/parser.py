import csv
import json
import os
import re
from collections import Counter

import numpy as np
import torch
from tqdm.auto import tqdm

# ── Paths ──────────────────────────────────────────────────────────
TRACE_PATH     = "data/raw/hdfs/Event_traces.csv"
LABEL_PATH     = "data/raw/hdfs/anomaly_label.csv"
OUT_DIR        = "data/processed"
PROCESSED_PATH = "data/processed/hdfs_processed.pt"
CACHE_PATH     = "data/processed/hdfs_cache.json"

# ── Config ─────────────────────────────────────────────────────────
SEQ_LEN       = 128
MAX_VOCAB     = 8192
MIN_FREQ      = 2
TRAIN_RATIO   = 0.75
VAL_RATIO     = 0.10
TEST_RATIO    = 0.15
SEED          = 42

CLASS_NAMES   = ["Normal", "Anomaly"]
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_:\-/\.]+")


# ══════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════

def _resolve_col(fieldnames, candidates):
    normalized = {n.strip().lower(): n for n in fieldnames}
    for c in candidates:
        if c.strip().lower() in normalized:
            return normalized[c.strip().lower()]
    return None


def _norm_block_id(raw):
    return str(raw).strip().strip('"').strip("'")


def _parse_label(raw):
    v = str(raw).strip().strip('"').strip("'").lower()
    if v in {"normal", "0", "false", "no", "benign"}:
        return 0
    if v in {"anomaly", "anomalous", "1", "true", "yes", "attack"}:
        return 1
    return None


def load_traces(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Event trace file not found: {path}")

    traces = {}
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        block_key = _resolve_col(reader.fieldnames,
                                 ("BlockId", "blockid", "block_id", "blk_id", "blkid"))
        seq_key   = _resolve_col(reader.fieldnames,
                                 ("EventSequence", "event_sequence",
                                  "events", "sequence", "Features", "features"))

        is_positional = False
        if block_key is None or seq_key is None:
            first = str(reader.fieldnames[0]) if reader.fieldnames else ""
            if first.startswith("blk_") or "Success" in reader.fieldnames:
                print(f"  [Parser] No headers in {os.path.basename(path)}, using positional indexing")
                block_key, seq_key, is_positional = 0, 3, True
                row_vals = list(reader.fieldnames)
                blk = _norm_block_id(row_vals[block_key])
                tokens = _tokenize(row_vals[seq_key])
                if blk and tokens:
                    traces[blk] = tokens
            else:
                raise KeyError(
                    f"Could not resolve BlockId/EventSequence columns in {path}. "
                    f"Available: {', '.join(reader.fieldnames)}"
                )

        for row in tqdm(reader, desc="Reading Event_traces.csv", unit="row", dynamic_ncols=True):
            if is_positional:
                vals = list(row.values())
                blk_raw, seq_raw = vals[block_key], vals[seq_key]
            else:
                blk_raw, seq_raw = row.get(block_key, ""), row.get(seq_key, "")
            blk = _norm_block_id(blk_raw)
            tokens = _tokenize(seq_raw)
            if blk and tokens:
                traces[blk] = tokens

    return traces


def load_labels(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")

    labels = {}
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        block_key = _resolve_col(reader.fieldnames,
                                 ("BlockId", "blockid", "block_id", "blk_id", "blkid"))
        label_key = _resolve_col(reader.fieldnames,
                                 ("Label", "label", "anomaly", "is_anomaly", "class"))
        if block_key is None or label_key is None:
            raise KeyError(
                f"Could not resolve BlockId/Label columns in {path}. "
                f"Available: {', '.join(reader.fieldnames)}"
            )

        for row in tqdm(reader, desc="Reading anomaly_label.csv", unit="row", dynamic_ncols=True):
            blk = _norm_block_id(row.get(block_key, ""))
            lbl = _parse_label(row.get(label_key, ""))
            if blk and lbl is not None:
                labels[blk] = lbl

    return labels


# ══════════════════════════════════════════════════════════════════
#  ENCODE
# ══════════════════════════════════════════════════════════════════

def _tokenize(raw):
    return [t.lower() for t in TOKEN_PATTERN.findall(str(raw))]


def build_vocab(token_lists):
    counts = Counter(t for tokens in token_lists for t in tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counts.most_common():
        if len(vocab) >= MAX_VOCAB:
            break
        if freq < MIN_FREQ:
            continue
        vocab[token] = len(vocab)
    return vocab


def encode(tokens, vocab):
    ids = [vocab.get(t, 1) for t in tokens]
    if len(ids) > SEQ_LEN:
        ids = ids[-SEQ_LEN:]
    else:
        ids = [0] * (SEQ_LEN - len(ids)) + ids
    return np.asarray(ids, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════
#  SPLIT
# ══════════════════════════════════════════════════════════════════

def split_counts(total):
    if total <= 0: return 0, 0, 0
    train_n = max(1, int(total * TRAIN_RATIO))
    val_n   = max(1, int(total * VAL_RATIO))
    test_n  = total - train_n - val_n
    if test_n <= 0:
        val_n  -= 1
        test_n  = total - train_n - val_n
    return train_n, val_n, test_n


def stratified_split(labels):
    rng = np.random.default_rng(SEED)
    trains, vals, tests = [], [], []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        idx = rng.permutation(idx)
        tr, va, _ = split_counts(len(idx))
        trains.append(idx[:tr])
        vals.append(idx[tr:tr+va])
        tests.append(idx[tr+va:])
    return (
        rng.permutation(np.concatenate(trains)),
        rng.permutation(np.concatenate(vals)),
        rng.permutation(np.concatenate(tests)),
    )


# ══════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════

def save_outputs(features, labels, block_ids, vocab, train_idx, val_idx, test_idx):
    os.makedirs(OUT_DIR, exist_ok=True)

    payload = {
        "class_names"    : CLASS_NAMES,
        "sequence_length": SEQ_LEN,
        "vocab_size"     : len(vocab),
        "splits": {
            split: {
                "X"        : torch.from_numpy(features[idx]),
                "y"        : torch.from_numpy(labels[idx]),
                "block_ids": [block_ids[int(i)] for i in idx],
            }
            for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]
        },
    }
    torch.save(payload, PROCESSED_PATH)

    cache = {
        "class_names": CLASS_NAMES,
        "vocab_size" : len(vocab),
        "vocab"      : vocab,
        "split_sizes": {
            s: int(payload["splits"][s]["y"].shape[0])
            for s in ("train", "val", "test")
        },
        "processed_path": PROCESSED_PATH,
    }
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\n  Train : {len(train_idx):,}")
    print(f"  Val   : {len(val_idx):,}")
    print(f"  Test  : {len(test_idx):,}")
    print(f"  Vocab : {len(vocab):,}")
    print(f"\nSaved -> {PROCESSED_PATH}")
    print(f"Cache  -> {CACHE_PATH}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  System Expert - HDFS Log Preprocessing")
    print("=" * 55 + "\n")

    traces = load_traces(TRACE_PATH)
    labels = load_labels(LABEL_PATH)

    # Join
    merged = sorted(
        [(blk, tokens, labels[blk]) for blk, tokens in traces.items() if blk in labels],
        key=lambda x: x[0],
    )
    if not merged:
        raise ValueError("No matched HDFS records found between trace and label files.")
    print(f"\n[Join] {len(merged):,} matched block IDs")

    # Build vocab
    vocab = build_vocab([tokens for _, tokens, _ in merged])

    # Encode
    block_ids, rows, targets = [], [], []
    for blk, tokens, lbl in tqdm(merged, desc="Encoding sequences", unit="row", dynamic_ncols=True):
        block_ids.append(blk)
        rows.append(encode(tokens, vocab))
        targets.append(lbl)

    features = np.stack(rows, axis=0)
    labels_arr = np.asarray(targets, dtype=np.int64)

    n_anom = labels_arr.sum()
    print(f"\n  Normal  : {len(labels_arr)-n_anom:,}  ({100*(len(labels_arr)-n_anom)/len(labels_arr):.1f}%)")
    print(f"  Anomaly : {n_anom:,}  ({100*n_anom/len(labels_arr):.1f}%)")

    # Split and save
    train_idx, val_idx, test_idx = stratified_split(labels_arr)
    save_outputs(features, labels_arr, block_ids, vocab, train_idx, val_idx, test_idx)

    print("\nDone! Next: uv run train_system")


if __name__ == "__main__":
    main()
