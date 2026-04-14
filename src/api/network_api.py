from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.experts.network_expert.test import BinaryAnalyzeConfig, run

FIXED_MODEL_PATH = Path("models/network_expert.pth")
FIXED_PREPROCESSED_DIR = Path("data/processed")
FIXED_METRICS_PATH = Path("models/network_meta.json")
FIXED_DEVICE = "cuda"
FIXED_BATCH_SIZE = 2048
FIXED_WINDOW_STEP = 1
FIXED_MAX_REPORT_ITEMS = 20
FIXED_HOST = "0.0.0.0"
FIXED_PORT = 8001

app = FastAPI(
    title="Network Anomaly Detection API",
    version="1.0.0",
    description=(
        "Binary network anomaly detection API for CICIDS-based models. "
        "Upload one file and the service returns whether it is benign or anomalous."
    ),
)


def _make_temp_upload_path(original_name: str) -> Path:
    suffix = Path(original_name or "upload.log").suffix or ".log"
    upload_dir = Path("data/uploads/api")
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir / f"network_{uuid4().hex}{suffix}"


def _looks_structured_file(path: Path) -> bool:
    return path.suffix.lower() in {".csv", ".json", ".jsonl", ".ndjson"}


def _build_config(
    *,
    input_file: Path | None = None,
    log_file: Path | None = None,
) -> BinaryAnalyzeConfig:
    return BinaryAnalyzeConfig(
        model_path=FIXED_MODEL_PATH,
        preprocessed_dir=FIXED_PREPROCESSED_DIR,
        metrics_path=FIXED_METRICS_PATH,
        dataset_split=None,
        input_file=input_file,
        log_file=log_file,
        log_text=None,
        input_format="auto",
        interactive=False,
        threshold=None,
        device=FIXED_DEVICE,
        batch_size=FIXED_BATCH_SIZE,
        window_step=FIXED_WINDOW_STEP,
        max_report_items=FIXED_MAX_REPORT_ITEMS,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "network-anomaly-api"}


@app.post("/v1/network/analyze-file")
async def analyze_network_file(
    upload_file: UploadFile = File(...),
) -> dict:
    temp_path = _make_temp_upload_path(upload_file.filename or "upload.log")
    try:
        raw = await upload_file.read()
        temp_path.write_bytes(raw)
        config = _build_config(
            input_file=temp_path if _looks_structured_file(temp_path) else None,
            log_file=temp_path if not _looks_structured_file(temp_path) else None,
        )
        return run(config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> None:
    import uvicorn

    uvicorn.run("src.api.network_api:app", host=FIXED_HOST, port=FIXED_PORT, reload=False)


if __name__ == "__main__":
    main()
