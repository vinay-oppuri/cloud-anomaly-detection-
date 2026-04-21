from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:-]+")
BLOCK_ID_PATTERN = re.compile(r"blk_-?\d+", flags=re.IGNORECASE)
IP_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?$")
HEX_PATTERN = re.compile(r"^(?:0x)?[0-9a-f]{6,}$", flags=re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
PATH_PATTERN = re.compile(r"^(?:/|[A-Za-z]:\\).+")
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class LogTemplateCluster:
    template_id: str
    tokens: list[str]
    size: int = 0
    examples: list[str] = field(default_factory=list)

    def merge(self, tokens: list[str], raw_message: str, max_examples: int = 5) -> None:
        self.size += 1
        if len(self.examples) < max_examples and raw_message not in self.examples:
            self.examples.append(raw_message)
        if len(tokens) != len(self.tokens):
            return
        self.tokens = [
            current if current == incoming else "<*>"
            for current, incoming in zip(self.tokens, tokens, strict=False)
        ]

    @property
    def template_text(self) -> str:
        return " ".join(self.tokens)


@dataclass(slots=True)
class DrainLikeParserConfig:
    similarity_threshold: float = 0.5
    max_examples_per_cluster: int = 5


class DrainLikeParser:
    """
    Lightweight Drain-style parser.

    This keeps the mechanics simple enough for the current repo:
    - logs are tokenized and variable-looking fields are normalized to `<*>`
    - candidate templates are bucketed by token count
    - the best cluster above a similarity threshold is updated in place
    """

    def __init__(self, config: DrainLikeParserConfig | None = None) -> None:
        self.config = config or DrainLikeParserConfig()
        self._buckets: dict[int, list[LogTemplateCluster]] = {}
        self._next_cluster_id = 1

    def parse_message(self, raw_message: str) -> LogTemplateCluster:
        tokens = self.normalize_tokens(tokenize_message(raw_message))
        bucket = self._buckets.setdefault(len(tokens), [])
        best_cluster = self._find_best_cluster(tokens, bucket)
        if best_cluster is None:
            cluster = LogTemplateCluster(
                template_id=f"E{self._next_cluster_id}",
                tokens=tokens or ["<EMPTY>"],
            )
            self._next_cluster_id += 1
            cluster.merge(tokens or ["<EMPTY>"], raw_message, self.config.max_examples_per_cluster)
            bucket.append(cluster)
            return cluster

        best_cluster.merge(tokens, raw_message, self.config.max_examples_per_cluster)
        return best_cluster

    def parse_messages(self, messages: Iterable[str]) -> list[LogTemplateCluster]:
        return [self.parse_message(message) for message in messages]

    def export_state(self) -> dict[str, object]:
        templates: list[dict[str, object]] = []
        for bucket_size in sorted(self._buckets):
            for cluster in self._buckets[bucket_size]:
                templates.append(
                    {
                        "template_id": cluster.template_id,
                        "token_count": bucket_size,
                        "template_tokens": cluster.tokens,
                        "template": cluster.template_text,
                        "occurrences": cluster.size,
                        "examples": cluster.examples,
                    }
                )
        return {
            "parser": "drain_like",
            "similarity_threshold": self.config.similarity_threshold,
            "templates": templates,
        }

    def normalize_tokens(self, tokens: list[str]) -> list[str]:
        normalized: list[str] = []
        for token in tokens:
            lowered = token.lower()
            if _looks_variable(lowered):
                normalized.append("<*>")
            else:
                normalized.append(lowered)
        return normalized

    def _find_best_cluster(
        self,
        tokens: list[str],
        bucket: list[LogTemplateCluster],
    ) -> LogTemplateCluster | None:
        best_cluster: LogTemplateCluster | None = None
        best_score = -1.0
        for cluster in bucket:
            score = _template_similarity(cluster.tokens, tokens)
            if score > best_score:
                best_score = score
                best_cluster = cluster
        if best_cluster is None:
            return None
        if best_score < self.config.similarity_threshold:
            return None
        return best_cluster


def tokenize_message(message: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(message))


def extract_block_ids(message: str) -> list[str]:
    return [match.lower() for match in BLOCK_ID_PATTERN.findall(str(message))]


def _looks_variable(token: str) -> bool:
    return bool(
        token == ""
        or token.startswith("blk_")
        or token.startswith("e") and token[1:].isdigit()
        or IP_PATTERN.match(token)
        or UUID_PATTERN.match(token)
        or HEX_PATTERN.match(token)
        or NUMBER_PATTERN.match(token)
        or PATH_PATTERN.match(token)
        or any(char.isdigit() for char in token)
    )


def _template_similarity(template_tokens: list[str], tokens: list[str]) -> float:
    if len(template_tokens) != len(tokens):
        return 0.0
    exact_matches = 0
    fixed_positions = 0
    for template_token, token in zip(template_tokens, tokens, strict=False):
        if template_token == "<*>":
            continue
        fixed_positions += 1
        if template_token == token:
            exact_matches += 1
    if fixed_positions == 0:
        return 1.0
    return exact_matches / fixed_positions


__all__ = [
    "DrainLikeParser",
    "DrainLikeParserConfig",
    "LogTemplateCluster",
    "extract_block_ids",
    "tokenize_message",
]
