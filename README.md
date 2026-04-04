## Modular Cloud Anomaly Detection (Production Scope)

This project is a production-focused anomaly detection system for cloud workloads using deep learning.

Current scope intentionally includes only two specialized experts:
- `NetworkExpert` for VPC/flow-log anomaly detection (`CNN + LSTM`)
- `SystemExpert` for host/system-log anomaly detection (`Bi-LSTM`)

No CloudTrail or LLM interpreter modules are included in the runtime path.

## Architecture

Pipeline flow:
1. Collect logs:
   - `src/collectors/vpc_collector.py`
   - `src/collectors/system_collector.py`
2. Preprocess:
   - `src/processing/encoders.py`
   - `src/processing/normalizers.py`
3. Inference:
   - `src/experts/network_model.py`
   - `src/experts/system_model.py`
4. Aggregate decisions:
   - `src/aggregator/ensemble.py`
5. Orchestrate end-to-end:
   - `src/pipeline.py`
   - `src/main.py`

## Run

```bash
uv sync
uv run python -m src.main
```

## Train

Processed dataset format:
- Network: `X` shape `[N, seq_len, feature_dim]`, `y` shape `[N]`
- System: `X` shape `[N, seq_len]` token IDs, `y` shape `[N]`
- Store as `.npz` (keys `X`, `y`) or `.pt` dict (`X`/`y` or `features`/`labels`)

Example commands:

```bash
uv run train-network ^
  --train-data data/processed/network_train.npz ^
  --val-data data/processed/network_val.npz ^
  --test-data data/processed/network_test.npz ^
  --class-names-path data/processed/network_classes.txt ^
  --output-dir models
```

```bash
uv run train-system ^
  --train-data data/processed/system_train.npz ^
  --val-data data/processed/system_val.npz ^
  --test-data data/processed/system_test.npz ^
  --class-names-path data/processed/system_classes.txt ^
  --output-dir models
```

Artifacts produced in `models/`:
- `network_expert_best.pth`, `network_expert_last.pth`, `network_expert_metrics.json`
- `system_expert_best.pth`, `system_expert_last.pth`, `system_expert_metrics.json`

## Test

```bash
uv run pytest -q
```

## Notes

- Inference pipeline expects `models/network_expert_best.pth` and `models/system_expert_best.pth`.
- If files are absent, the pipeline still runs with initialized model weights for smoke-testing only.
