from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, model_validator

from src.experts.system_expert.service import SystemAnomalyService, SystemServiceConfig

app = FastAPI(
    title="System Log Anomaly API",
    version="1.0.0",
    description=(
        "System-log anomaly detection API for HDFS logs. "
        "Input can be uploaded raw log file, raw log text, or event sequence."
    ),
)


class AnalyzeSystemLogRequest(BaseModel):
    event_name: str = Field(default="uploaded-log", description="Logical name for this uploaded log batch.")
    event_sequence: str | None = Field(
        default=None,
        description="Optional event-id sequence, e.g. 'E5 E26 E11 E9'.",
    )
    log_text: str | None = Field(
        default=None,
        description="Optional raw multiline system log text.",
    )
    log_lines: list[str] | None = Field(
        default=None,
        description="Optional list of raw system log lines.",
    )

    @model_validator(mode="after")
    def validate_one_input(self) -> AnalyzeSystemLogRequest:
        provided = [
            self.event_sequence is not None and self.event_sequence.strip() != "",
            self.log_text is not None and self.log_text.strip() != "",
            self.log_lines is not None and len(self.log_lines) > 0,
        ]
        if sum(1 for item in provided if item) != 1:
            raise ValueError(
                "Provide exactly one of: event_sequence, log_text, log_lines."
            )
        return self


@lru_cache(maxsize=1)
def get_system_service() -> SystemAnomalyService:
    show_progress = os.getenv("SYSTEM_SHOW_WORKFLOW_PROGRESS", "true").strip().lower() != "false"
    config = SystemServiceConfig(
        processed_data=Path(os.getenv("SYSTEM_PROCESSED_DATA", "data/processed/hdfs_processed.pt")),
        cache_path=Path(os.getenv("SYSTEM_CACHE_PATH", "data/processed/hdfs_cache.json")),
        model_path=Path(os.getenv("SYSTEM_MODEL_PATH", "models/system_expert_best.pth")),
        device=os.getenv("SYSTEM_DEVICE", "cpu"),
        normal_class_index=int(os.getenv("SYSTEM_NORMAL_CLASS_INDEX", "0")),
        use_gemini=os.getenv("SYSTEM_USE_GEMINI", "true").strip().lower() != "false",
        gemini_model=os.getenv("SYSTEM_GEMINI_MODEL", "gemini-2.5-flash"),
        show_workflow_progress=show_progress,
    )
    return SystemAnomalyService.from_config(config)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "system-log-anomaly-api"}


@app.post("/v1/system/analyze")
def analyze_system_log(payload: AnalyzeSystemLogRequest) -> dict[str, Any]:
    service = get_system_service()
    try:
        if payload.event_sequence is not None and payload.event_sequence.strip():
            return service.analyze_event_sequence(
                payload.event_sequence,
                event_name=payload.event_name,
            )
        if payload.log_text is not None and payload.log_text.strip():
            lines = payload.log_text.splitlines()
            return service.analyze_log_lines(lines, event_name=payload.event_name)
        if payload.log_lines is not None and len(payload.log_lines) > 0:
            return service.analyze_log_lines(payload.log_lines, event_name=payload.event_name)
        raise ValueError("No input content found.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/system/analyze-file")
async def analyze_system_log_file(
    event_name: str = Form("uploaded-log"),
    log_file: UploadFile = File(...),
) -> dict[str, Any]:
    service = get_system_service()
    try:
        raw = await log_file.read()
        text = raw.decode("utf-8", errors="ignore")
        lines = text.splitlines()
        return service.analyze_log_lines(lines, event_name=event_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    host = os.getenv("SYSTEM_API_HOST", "0.0.0.0")
    port = int(os.getenv("SYSTEM_API_PORT", "8000"))
    uvicorn.run("src.api.system_api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
