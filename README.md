## System Log Anomaly Detection (HDFS)

This project is now organized around one reusable system-log flow:

1. user uploads raw HDFS logs  
2. API/CLI preprocesses logs into event sequence (`E*`)  
3. trained HDFS transformer predicts anomaly  
4. output returns:
   - `anomaly_detected`
   - `anomaly_type`
   - `reason`
   - `action`

`reason` and `action` are generated using Gemini (`google-genai`) with fallback heuristics.

This repository is currently system-log focused, while network modules are also available for later expansion.

## Network Flow Training (CICIDS)

Network expert training now follows a CNN-LSTM pipeline for CICIDS:

1. preprocess CICIDS CSVs into fixed windows (`seq_len x 80` flow features)
2. run 1D CNN over each flow vector to learn feature interactions
3. run LSTM across flow windows to learn temporal attack evolution
4. classify with softmax over the CICIDS attack taxonomy (15 classes)

Commands:

```bash
uv run prepare_cicids
uv run train_cicids
uv run test_cicids --split test --device cpu
```

## Setup

```bash
uv sync
```

For Gemini explanations, set one of:
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

`.env` is also supported.

## Project Structure (System Side)

- `src/experts/system_expert/parser.py`  
  prepare HDFS dataset from `Event_traces.csv` + `anomaly_label.csv`
- `src/experts/system_expert/train.py`  
  train transformer checkpoint
- `src/experts/system_expert/service.py`  
  single reusable service for:
  - split evaluation
  - uploaded-log analysis
- `src/experts/system_expert/test.py`  
  unified command for both evaluation and real-world log testing
- `src/api/system_api.py`  
  production API (JSON + file upload)

## Data Preparation

```bash
uv run prepare_hdfs
```

## Training

```bash
uv run train_hdfs
```

## Testing (Single Unified Command)

### 1. Dataset evaluation (offline metrics)

```bash
uv run test_hdfs --mode evaluate --split test --device cpu
```

### 2. Real-world log testing (uploaded/raw logs)

Create an example upload log:

```bash
powershell -Command "New-Item -ItemType Directory -Path data/uploads -Force | Out-Null; @'
2026-04-07T02:10:01Z INFO FSNamesystem: BLOCK* allocate blk_107 for /warehouse/events/file-1
2026-04-07T02:10:03Z INFO DataNode: Receiving block BP-1:blk_107 src=/10.20.1.5:50010
2026-04-07T02:10:05Z ERROR FSNamesystem: BLOCK* invalidateBlocks blk_107 due to corrupt replica
'@ | Set-Content data/uploads/system.log -Encoding UTF8"
```

```bash
uv run test_hdfs --mode analyze --log-file data/uploads/system.log --device cpu
```

### 3. Direct event sequence testing

```bash
uv run test_hdfs --mode analyze --event-sequence "E5 E26 E11 E9 E21 E23 E22 E3 E1 E17 E19 E24 E29 E13" --device cpu
```

## API (Production Path)

Start API:

```bash
uv run serve_system_api
```

Live workflow progress in terminal is enabled by default. You will see:

1. `Step 1/6: Validate API input`
2. `Step 2/6: Preprocess and extract events`
3. `Step 3/6: Encode event sequence`
4. `Step 4/6: Run transformer inference`
5. `Step 5/6: Classify anomaly type (rule-based)`
6. `Step 6/6: Generate LLM reason/action and build response`

To disable terminal progress bars:

```bash
set SYSTEM_SHOW_WORKFLOW_PROGRESS=false
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Analyze raw log file upload:

```bash
curl -X POST "http://127.0.0.1:8000/v1/system/analyze-file" ^
  -F "event_name=hdfs-upload-1" ^
  -F "log_file=@data/uploads/system.log"
```

Analyze raw text JSON:

```bash
curl -X POST "http://127.0.0.1:8000/v1/system/analyze" ^
  -H "Content-Type: application/json" ^
  -d "{\"event_name\":\"hdfs-json-1\",\"log_text\":\"2026-04-07T02:10:01Z ...\"}"
```

## Output Contract

Both CLI and API return:

- `anomaly_detected` (bool)
- `anomaly_type` (string)
- `reason` (string)
- `action` (string)
- `metadata` (severity, confidence, source, etc.)
