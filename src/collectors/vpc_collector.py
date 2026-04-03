from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterable, AsyncIterator, Mapping, Sequence


@dataclass(slots=True)
class VPCFlowRecord:
    """Normalized VPC flow log record."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    packets: int
    bytes_transferred: int
    action: str
    log_status: str
    raw: str


class VPCFlowCollector:
    """
    Collector/parser for AWS VPC flow logs.

    Supports both:
    - key=value style lines
    - default positional flow-log format
    """

    def collect_from_lines(self, lines: Sequence[str]) -> list[VPCFlowRecord]:
        records: list[VPCFlowRecord] = []
        for line in lines:
            parsed = self.parse_line(line)
            if parsed is not None:
                records.append(parsed)
        return records

    async def stream_from_iterable(
        self,
        lines: AsyncIterable[str],
        poll_interval_seconds: float = 0.0,
    ) -> AsyncIterator[VPCFlowRecord]:
        async for line in lines:
            parsed = self.parse_line(line)
            if parsed is not None:
                yield parsed
            if poll_interval_seconds > 0:
                await asyncio.sleep(poll_interval_seconds)

    def parse_line(self, line: str) -> VPCFlowRecord | None:
        stripped = line.strip()
        if not stripped:
            return None

        if "=" in stripped:
            kv = self._parse_key_values(stripped)
            return VPCFlowRecord(
                src_ip=kv.get("src") or kv.get("srcaddr") or "0.0.0.0",
                dst_ip=kv.get("dst") or kv.get("dstaddr") or "0.0.0.0",
                src_port=self._to_int(kv.get("spt") or kv.get("srcport")),
                dst_port=self._to_int(kv.get("dpt") or kv.get("dstport")),
                protocol=self._to_int(kv.get("proto") or kv.get("protocol"), default=6),
                packets=self._to_int(kv.get("packets"), default=1),
                bytes_transferred=self._to_int(kv.get("bytes"), default=0),
                action=(kv.get("action") or "ACCEPT").upper(),
                log_status=(kv.get("log_status") or kv.get("log-status") or "OK").upper(),
                raw=stripped,
            )

        tokens = stripped.split()
        if len(tokens) < 14:
            return None

        # AWS default VPC flow log order:
        # version account-id interface-id srcaddr dstaddr srcport dstport protocol packets bytes start end action log-status
        return VPCFlowRecord(
            src_ip=tokens[3],
            dst_ip=tokens[4],
            src_port=self._to_int(tokens[5]),
            dst_port=self._to_int(tokens[6]),
            protocol=self._to_int(tokens[7], default=6),
            packets=self._to_int(tokens[8], default=1),
            bytes_transferred=self._to_int(tokens[9], default=0),
            action=tokens[12].upper(),
            log_status=tokens[13].upper(),
            raw=stripped,
        )

    def _parse_key_values(self, line: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            result[key.strip().lower()] = value.strip()
        return result

    def _to_int(self, value: str | None, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

