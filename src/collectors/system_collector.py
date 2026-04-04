from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import AsyncIterable, AsyncIterator, Sequence


@dataclass(slots=True)
class SystemLogRecord:
    """Normalized system log record."""

    host: str
    service: str
    severity: str
    user: str | None
    source_ip: str | None
    message: str
    raw: str


class SystemLogCollector:
    """Collector/parser for syslog-like host logs."""

    _kv_token = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)=([^ ]+)")
    _ip_token = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

    def collect_from_lines(self, lines: Sequence[str]) -> list[SystemLogRecord]:
        records: list[SystemLogRecord] = []
        for line in lines:
            parsed = self.parse_line(line)
            if parsed is not None:
                records.append(parsed)
        return records

    async def stream_from_iterable(
        self,
        lines: AsyncIterable[str],
        poll_interval_seconds: float = 0.0,
    ) -> AsyncIterator[SystemLogRecord]:
        async for line in lines:
            parsed = self.parse_line(line)
            if parsed is not None:
                yield parsed
            if poll_interval_seconds > 0:
                await asyncio.sleep(poll_interval_seconds)

    def parse_line(self, line: str) -> SystemLogRecord | None:
        raw = line.strip()
        if not raw:
            return None

        kv: dict[str, str] = {}
        for key, value in self._kv_token.findall(raw):
            kv[key.lower()] = value.strip("'\"")

        host = kv.get("host", "unknown-host")
        service = kv.get("service") or kv.get("app") or kv.get("proc") or "system"
        severity = (kv.get("severity") or kv.get("level") or "info").upper()
        user = kv.get("user")
        ip = kv.get("ip") or kv.get("src") or kv.get("source_ip")
        if ip is None:
            match = self._ip_token.search(raw)
            ip = match.group(0) if match else None

        message = kv.get("msg", raw)

        return SystemLogRecord(
            host=host,
            service=service,
            severity=severity,
            user=user,
            source_ip=ip,
            message=message,
            raw=raw,
        )

