from __future__ import annotations

import hashlib
import math
import re
from collections import deque
from typing import Deque, Sequence

import torch

from src.collectors.system_collector import SystemLogRecord
from src.collectors.vpc_collector import VPCFlowRecord


class NetworkFeatureEncoder:
    """
    Converts parsed VPC flow records into fixed-length tensor windows.

    Output shape: [window_size, feature_dim]
    Default feature_dim is 16 to match NetworkExpert input.
    """

    def __init__(self, window_size: int = 48, feature_dim: int = 16) -> None:
        self.window_size = window_size
        self.feature_dim = feature_dim
        self._buffer: Deque[VPCFlowRecord] = deque(maxlen=window_size)

    def append(self, record: VPCFlowRecord) -> None:
        self._buffer.append(record)

    def encode_current_window(self) -> torch.Tensor:
        records = list(self._buffer)
        return self.encode_records(records)

    def encode_records(self, records: Sequence[VPCFlowRecord]) -> torch.Tensor:
        vectors: list[torch.Tensor] = [self._encode_record(record) for record in records]
        if len(vectors) > self.window_size:
            vectors = vectors[-self.window_size :]

        if not vectors:
            return torch.zeros((self.window_size, self.feature_dim), dtype=torch.float32)

        while len(vectors) < self.window_size:
            vectors.insert(0, torch.zeros(self.feature_dim, dtype=torch.float32))

        matrix = torch.stack(vectors, dim=0)
        return matrix

    def _encode_record(self, record: VPCFlowRecord) -> torch.Tensor:
        src_parts = self._ip_to_parts(record.src_ip)
        dst_parts = self._ip_to_parts(record.dst_ip)

        protocol = float(record.protocol) / 255.0
        src_port = float(record.src_port) / 65535.0
        dst_port = float(record.dst_port) / 65535.0
        packets = self._log_scale(record.packets)
        byte_count = self._log_scale(record.bytes_transferred)
        avg_packet_size = self._safe_ratio(record.bytes_transferred, max(1, record.packets))
        avg_packet_size = self._log_scale(avg_packet_size)
        action_accept = 1.0 if record.action.upper() == "ACCEPT" else 0.0
        status_ok = 1.0 if record.log_status.upper() == "OK" else 0.0

        base = [
            *src_parts,
            *dst_parts,
            src_port,
            dst_port,
            protocol,
            packets,
            byte_count,
            avg_packet_size,
            action_accept,
            status_ok,
        ]

        if len(base) < self.feature_dim:
            base.extend([0.0] * (self.feature_dim - len(base)))
        elif len(base) > self.feature_dim:
            base = base[: self.feature_dim]

        return torch.tensor(base, dtype=torch.float32)

    def _ip_to_parts(self, ip: str) -> list[float]:
        parts = ip.split(".")
        if len(parts) != 4:
            return [0.0, 0.0, 0.0, 0.0]

        values: list[float] = []
        for part in parts:
            try:
                value = int(part)
            except ValueError:
                value = 0
            values.append(float(min(max(value, 0), 255)) / 255.0)
        return values

    def _log_scale(self, value: int | float) -> float:
        return float(math.log1p(max(float(value), 0.0)) / 16.0)

    def _safe_ratio(self, numerator: int | float, denominator: int | float) -> float:
        denom = float(denominator)
        if denom <= 0.0:
            return 0.0
        return float(numerator) / denom


class SystemLogEncoder:
    """
    Tokenizes system log text into integer ids for Bi-LSTM input.

    Output shape: [sequence_length]
    """

    _token_pattern = re.compile(r"[A-Za-z0-9_\-:.\/]+")

    def __init__(self, vocab_size: int = 5000, sequence_length: int = 48) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2.")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

    def encode_text(self, text: str) -> torch.Tensor:
        tokens = self._tokenize(text)
        token_ids = [self._hash_token(token) for token in tokens]

        if len(token_ids) > self.sequence_length:
            token_ids = token_ids[-self.sequence_length :]
        else:
            token_ids = ([0] * (self.sequence_length - len(token_ids))) + token_ids

        return torch.tensor(token_ids, dtype=torch.long)

    def encode_record(self, event: SystemLogRecord) -> torch.Tensor:
        composed = (
            f"host={event.host} service={event.service} severity={event.severity} "
            f"user={event.user or 'none'} source_ip={event.source_ip or 'none'} "
            f"msg={event.message}"
        )
        return self.encode_text(composed)

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in self._token_pattern.findall(text)]

    def _hash_token(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
        # Reserve 0 for padding
        return (int(digest[:8], 16) % (self.vocab_size - 1)) + 1
