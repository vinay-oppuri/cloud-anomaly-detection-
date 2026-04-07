from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experts.system_expert.service import (
    DEFAULT_CACHE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_PATH,
    SystemAnomalyService,
    SystemServiceConfig,
)


@dataclass(slots=True)
class SystemTestConfig:
    mode: str
    processed_data: Path
    cache_path: Path
    model_path: Path
    split: str
    batch_size: int
    device: str
    normal_class_index: int
    use_gemini: bool
    gemini_model: str
    show_workflow_progress: bool
    log_file: Path | None
    event_sequence: str | None
    event_name: str


def parse_args() -> SystemTestConfig:
    parser = argparse.ArgumentParser(
        description=(
            "HDFS system expert testing command.\n"
            "- mode=evaluate: compute dataset metrics on train/val/test split\n"
            "- mode=analyze: analyze uploaded real-world logs or event sequence"
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="evaluate",
        choices=("evaluate", "analyze"),
    )
    parser.add_argument("--processed-data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normal-class-index", type=int, default=0)
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--disable-gemini", action="store_true")
    parser.add_argument("--hide-workflow-progress", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--event-sequence", type=str, default=None)
    parser.add_argument("--event-name", type=str, default="uploaded-log")
    ns = parser.parse_args()

    return SystemTestConfig(
        mode=ns.mode,
        processed_data=ns.processed_data,
        cache_path=ns.cache_path,
        model_path=ns.model_path,
        split=ns.split,
        batch_size=ns.batch_size,
        device=ns.device,
        normal_class_index=ns.normal_class_index,
        use_gemini=not ns.disable_gemini,
        gemini_model=ns.gemini_model,
        show_workflow_progress=not ns.hide_workflow_progress,
        log_file=ns.log_file,
        event_sequence=ns.event_sequence,
        event_name=ns.event_name,
    )


def main() -> None:
    config = parse_args()
    output = run(config)
    print(json.dumps(output, indent=2))


def run(config: SystemTestConfig) -> dict[str, Any]:
    service = SystemAnomalyService.from_config(
        SystemServiceConfig(
            processed_data=config.processed_data,
            cache_path=config.cache_path,
            model_path=config.model_path,
            device=config.device,
            normal_class_index=config.normal_class_index,
            use_gemini=config.use_gemini,
            gemini_model=config.gemini_model,
            show_workflow_progress=config.show_workflow_progress,
        )
    )

    if config.mode == "evaluate":
        return service.evaluate_split(
            split=config.split,
            batch_size=config.batch_size,
        )

    if config.event_sequence is not None and config.event_sequence.strip():
        return service.analyze_event_sequence(
            config.event_sequence,
            event_name=config.event_name,
        )
    if config.log_file is not None:
        return service.analyze_log_file(
            config.log_file,
            event_name=config.event_name,
        )
    raise ValueError(
        "For --mode analyze, pass either --event-sequence or --log-file."
    )


if __name__ == "__main__":
    main()
