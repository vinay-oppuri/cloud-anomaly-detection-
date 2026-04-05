## Cloud Anomaly Detection (API-Ready)

This project keeps one end-to-end flow for cloud logs:
- parse + clean raw logs
- detect anomaly
- predict anomaly type
- generate reason and action using Gemini (`google-genai`)

## Core Modules

1. Log parsing/cleaning:
   - `src/collectors/vpc_collector.py`
   - `src/collectors/system_collector.py`
   - `src/processing/encoders.py`
   - `src/processing/normalizers.py`
2. Expert models:
   - `src/experts/network_expert/*` (CICIDS / CNN-LSTM)
   - `src/experts/system_expert/*` (HDFS / Transformer)
3. Fusion + advice:
   - `src/aggregator/ensemble.py`
   - `src/interpreter/advisor.py`
   - `src/pipeline.py`

## Setup

```bash
uv sync
```

For Gemini explanations, set one of:
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

## Data Preparation

```bash
uv run prepare_hdfs
uv run prepare_cicids
```

## Training

```bash
uv run train_hdfs
uv run train_cicids
```

## Testing

```bash
uv run test_hdfs
uv run test_cicids
```

## Inference (Raw Logs -> Result)

Demo events:

```bash
uv run analyze_logs --demo
```

Single event:

```bash
uv run analyze_logs --event-name e1 --vpc-flow-line "2 111122223333 eni-1 10.0.0.1 10.0.0.2 50000 443 6 10 1200 1700000000 1700000060 ACCEPT OK" --system-log-line "host=i-1 service=sshd severity=warning msg='Failed password' user=admin ip=185.10.10.10"
```

Output includes:
- `anomaly_detected`
- `anomaly_type`
- `reason`
- `action`
