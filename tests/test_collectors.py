from __future__ import annotations

from src.collectors.system_collector import SystemLogCollector
from src.collectors.vpc_collector import VPCFlowCollector


def test_vpc_collector_parses_positional_line() -> None:
    collector = VPCFlowCollector()
    line = (
        "2 111122223333 eni-00abc123 10.0.1.10 10.0.2.8 51512 443 6 12 824 "
        "1700000000 1700000060 ACCEPT OK"
    )
    record = collector.parse_line(line)

    assert record is not None
    assert record.src_ip == "10.0.1.10"
    assert record.dst_port == 443
    assert record.action == "ACCEPT"


def test_system_collector_parses_key_value_line() -> None:
    collector = SystemLogCollector()
    line = (
        "host=i-93 service=sshd severity=warning msg='Failed password for admin' "
        "user=admin ip=185.23.44.91"
    )
    record = collector.parse_line(line)

    assert record is not None
    assert record.host == "i-93"
    assert record.service == "sshd"
    assert record.severity == "WARNING"
    assert record.user == "admin"
    assert record.source_ip == "185.23.44.91"

