from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterable, AsyncIterator, Mapping, Sequence


@dataclass(slots=True)
class CloudTrailEvent:
    """Normalized CloudTrail event."""

    event_name: str
    event_source: str
    aws_region: str
    user_identity: str
    source_ip: str
    user_agent: str
    error_code: str | None
    error_message: str | None
    raw: Mapping[str, Any]


class CloudTrailCollector:
    """Collector/parser for CloudTrail JSON logs."""

    def collect_from_json_lines(self, lines: Sequence[str]) -> list[CloudTrailEvent]:
        records: list[CloudTrailEvent] = []
        for line in lines:
            event = self.parse_json_line(line)
            if event is not None:
                records.append(event)
        return records

    def collect_from_records(self, records: Sequence[Mapping[str, Any]]) -> list[CloudTrailEvent]:
        parsed: list[CloudTrailEvent] = []
        for record in records:
            event = self.parse_record(record)
            if event is not None:
                parsed.append(event)
        return parsed

    async def stream_from_iterable(
        self,
        records: AsyncIterable[Mapping[str, Any] | str],
        poll_interval_seconds: float = 0.0,
    ) -> AsyncIterator[CloudTrailEvent]:
        async for item in records:
            parsed = self.parse_json_line(item) if isinstance(item, str) else self.parse_record(item)
            if parsed is not None:
                yield parsed
            if poll_interval_seconds > 0:
                await asyncio.sleep(poll_interval_seconds)

    def parse_json_line(self, line: str) -> CloudTrailEvent | None:
        stripped = line.strip()
        if not stripped:
            return None
        payload = json.loads(stripped)

        # Some exports wrap events in {"Records": [...]}
        records = payload.get("Records")
        if isinstance(records, list) and records:
            first = records[0]
            if isinstance(first, Mapping):
                return self.parse_record(first)

        if isinstance(payload, Mapping):
            return self.parse_record(payload)
        return None

    def parse_record(self, record: Mapping[str, Any]) -> CloudTrailEvent | None:
        event_name = str(record.get("eventName", "")).strip()
        event_source = str(record.get("eventSource", "")).strip()
        if not event_name and not event_source:
            return None

        user_identity_raw = record.get("userIdentity", {})
        user_identity = self._extract_user_identity(user_identity_raw)

        return CloudTrailEvent(
            event_name=event_name or "UnknownEvent",
            event_source=event_source or "unknown.amazonaws.com",
            aws_region=str(record.get("awsRegion", "unknown")),
            user_identity=user_identity,
            source_ip=str(record.get("sourceIPAddress", "0.0.0.0")),
            user_agent=str(record.get("userAgent", "unknown")),
            error_code=self._optional_str(record.get("errorCode")),
            error_message=self._optional_str(record.get("errorMessage")),
            raw=record,
        )

    def _extract_user_identity(self, user_identity: Any) -> str:
        if not isinstance(user_identity, Mapping):
            return "unknown"

        for key in ("arn", "userName", "principalId", "type"):
            value = user_identity.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
        return "unknown"

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

