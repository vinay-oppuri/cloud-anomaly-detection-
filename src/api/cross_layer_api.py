from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, model_validator

from src.correlation.realworld import CrossLayerRealWorldAnalyzer, RealWorldAnalyzerConfig


app = FastAPI(
    title="Cross-Layer Anomaly Correlation API",
    version="1.0.0",
    description=(
        "Unified end-to-end API for raw network-log detection, raw HDFS log detection, "
        "and cross-layer anomaly correlation."
    ),
)


class CrossLayerAnalyzeRequest(BaseModel):
    incident_name: str = Field(
        default="uploaded-incident",
        description="Logical identifier for this incident analysis request.",
    )
    network_log_text: str = Field(
        description="Raw network log text, one flow/log line per line.",
    )
    system_log_text: str = Field(
        description="Raw HDFS/system log text, one log line per line.",
    )
    use_correlation_model: bool = Field(default=True)
    use_gemini: bool = Field(default=False)
    gemini_model: str = Field(default="gemini-2.5-flash")
    temporal_window_minutes: int = Field(default=20, ge=1, le=180)
    edge_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    align_on_relative_start: bool = Field(
        default=False,
        description=(
            "Optional clock-offset normalization. Useful when two raw captures belong to the same incident "
            "but start at different absolute times."
        ),
    )

    @model_validator(mode="after")
    def validate_payload(self) -> "CrossLayerAnalyzeRequest":
        if not self.network_log_text.strip():
            raise ValueError("network_log_text must not be empty.")
        if not self.system_log_text.strip():
            raise ValueError("system_log_text must not be empty.")
        return self


@lru_cache(maxsize=1)
def get_cross_layer_analyzer() -> CrossLayerRealWorldAnalyzer:
    correlation_model_path = Path(os.getenv("CORRELATION_MODEL_PATH", "models/correlation_attention.pth"))
    if not correlation_model_path.exists():
        correlation_model_path = None
    config = RealWorldAnalyzerConfig(
        device=os.getenv("CROSS_LAYER_DEVICE", "cuda"),
        network_model_path=Path(os.getenv("NETWORK_MODEL_PATH", "models/network_expert.pth")),
        network_preprocessed_dir=Path(os.getenv("NETWORK_PREPROCESSED_DIR", "data/processed")),
        network_metrics_path=Path(os.getenv("NETWORK_METRICS_PATH", "models/network_meta.json")),
        system_processed_path=Path(os.getenv("SYSTEM_PROCESSED_DATA", "data/processed/hdfs_processed.pt")),
        system_cache_path=Path(os.getenv("SYSTEM_CACHE_PATH", "data/processed/hdfs_cache.json")),
        system_model_path=Path(os.getenv("SYSTEM_MODEL_PATH", "models/system_expert_best.pth")),
        system_metrics_path=Path(os.getenv("SYSTEM_METRICS_PATH", "models/system_expert_metrics.json")),
        correlation_model_path=correlation_model_path,
        batch_size=int(os.getenv("CROSS_LAYER_BATCH_SIZE", "512")),
        window_step=int(os.getenv("CROSS_LAYER_WINDOW_STEP", "1")),
        max_report_items=int(os.getenv("CROSS_LAYER_MAX_REPORT_ITEMS", "10")),
        temporal_window_minutes=int(os.getenv("CROSS_LAYER_TEMPORAL_WINDOW_MINUTES", "20")),
        edge_threshold=float(os.getenv("CROSS_LAYER_EDGE_THRESHOLD", "0.62")),
        use_correlation_model=os.getenv("CROSS_LAYER_USE_CORRELATION_MODEL", "true").strip().lower() != "false",
        use_gemini=os.getenv("CROSS_LAYER_USE_GEMINI", "false").strip().lower() == "true",
        gemini_model=os.getenv("CROSS_LAYER_GEMINI_MODEL", "gemini-2.5-flash"),
        session_gap_seconds=int(os.getenv("CROSS_LAYER_SESSION_GAP_SECONDS", "90")),
    )
    return CrossLayerRealWorldAnalyzer(config)


@app.get("/health")
def health() -> dict[str, Any]:
    analyzer = get_cross_layer_analyzer()
    return {
        "status": "ok",
        "service": "cross-layer-anomaly-api",
        "device": str(analyzer.device),
    }


@app.post("/v1/cross-layer/analyze")
def analyze_cross_layer(payload: CrossLayerAnalyzeRequest) -> dict[str, Any]:
    try:
        analyzer = get_cross_layer_analyzer()
        return analyzer.analyze_texts(
            network_log_text=payload.network_log_text,
            system_log_text=payload.system_log_text,
            incident_name=payload.incident_name,
            use_correlation_model=payload.use_correlation_model,
            use_gemini=payload.use_gemini,
            gemini_model=payload.gemini_model,
            temporal_window_minutes=payload.temporal_window_minutes,
            edge_threshold=payload.edge_threshold,
            align_on_relative_start=payload.align_on_relative_start,
        )
    except Exception as exc:
        status_code = 503 if isinstance(exc, FileNotFoundError) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@app.post("/v1/cross-layer/analyze-files")
async def analyze_cross_layer_files(
    incident_name: str = Form("uploaded-incident"),
    use_correlation_model: bool = Form(True),
    use_gemini: bool = Form(False),
    gemini_model: str = Form("gemini-2.5-flash"),
    temporal_window_minutes: int = Form(20),
    edge_threshold: float = Form(0.62),
    align_on_relative_start: bool = Form(False),
    network_log_file: UploadFile = File(...),
    system_log_file: UploadFile = File(...),
) -> dict[str, Any]:
    try:
        analyzer = get_cross_layer_analyzer()
        network_text = (await network_log_file.read()).decode("utf-8", errors="ignore")
        system_text = (await system_log_file.read()).decode("utf-8", errors="ignore")
        return analyzer.analyze_texts(
            network_log_text=network_text,
            system_log_text=system_text,
            incident_name=incident_name,
            use_correlation_model=use_correlation_model,
            use_gemini=use_gemini,
            gemini_model=gemini_model,
            temporal_window_minutes=temporal_window_minutes,
            edge_threshold=edge_threshold,
            align_on_relative_start=align_on_relative_start,
        )
    except Exception as exc:
        status_code = 503 if isinstance(exc, FileNotFoundError) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    host = os.getenv("CROSS_LAYER_API_HOST", "127.0.0.1")
    port = int(os.getenv("CROSS_LAYER_API_PORT", "8002"))
    uvicorn.run("src.api.cross_layer_api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
