## Cloud Anomaly Detection

Primary source code is organized under `src/`:

- `src/experts/network_expert/` for network anomaly detection
- `src/experts/system_expert/` for system anomaly detection
- `src/correlation/` for cross-layer correlation
- `src/api/` for FastAPI entry points
- `src/validation/` for end-to-end model checks

The expert pipelines follow the same structure:

- `parser.py` for preprocessing
- `train.py` for model training
- `test.py` for evaluation/inference

## Setup

```bash
uv sync
```

## System Log Anomaly Detection (HDFS)

Code path:

- `src/experts/system_expert/parser.py`
- `src/experts/system_expert/train.py`
- `src/experts/system_expert/test.py`
- `src/experts/system_expert/service.py`

Commands:

```bash
uv run prepare_hdfs
uv run train_hdfs
uv run test_hdfs --mode evaluate --split test --device cpu
```

Analyze uploaded log file:

```bash
uv run test_hdfs --mode analyze --log-file data/uploads/system.log --device cpu
```

## Network Anomaly Detection (CICIDS, Binary)

Code path:

- `src/experts/network_expert/parser.py`
- `src/experts/network_expert/train.py`
- `src/experts/network_expert/test.py`

Commands:

```bash
uv run prepare_cicids
uv run train_cicids
uv run test_cicids --log-file data/uploads/raw_network_normal_long.log --device cuda
```

Binary aliases are also available:

```bash
uv run preprocess_cicids_binary
uv run train_cicids_binary
uv run test_cicids_binary --log-file data/uploads/raw_network_anomaly_long.log --device cuda
```

## Cross-Layer Correlation

Code path:

- `src/correlation/train.py`
- `src/correlation/pipeline.py`
- `src/correlation/realworld.py`

Commands:

```bash
uv run train_correlation
uv run run_correlation --split test --device cpu
uv run serve_cross_layer_api
```

## Validation

Run the end-to-end expert checks:

```bash
uv run run_model_tests
```

## API

Start system API:

```bash
uv run serve_system_api
```
