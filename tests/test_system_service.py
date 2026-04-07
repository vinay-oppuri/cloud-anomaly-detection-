from __future__ import annotations

from src.experts.system_expert.service import extract_event_tokens_from_lines


def test_extract_event_tokens_from_explicit_event_ids() -> None:
    lines = [
        "2026-04-07T02:10:01Z ... event_id=E5",
        "2026-04-07T02:10:03Z ... event_id=E26",
        "2026-04-07T02:10:04Z ... event_id=E11",
    ]
    extraction = extract_event_tokens_from_lines(lines)
    assert extraction.event_tokens == ["e5", "e26", "e11"]
    assert extraction.extracted_from_event_id == 3
    assert extraction.inferred_from_templates == 0
    assert extraction.unmatched_lines == 0


def test_extract_event_tokens_from_raw_templates() -> None:
    lines = [
        "INFO FSNamesystem: BLOCK* allocate blk_107 for /warehouse/events/file-1",
        "INFO DataNode: Receiving block BP-1:blk_107 src=/10.20.1.5:50010",
        "ERROR FSNamesystem: BLOCK* invalidateBlocks blk_107 due to corrupt replica",
    ]
    extraction = extract_event_tokens_from_lines(lines)
    assert extraction.event_tokens == ["e5", "e26", "e24"]
    assert extraction.extracted_from_event_id == 0
    assert extraction.inferred_from_templates == 3
    assert extraction.unmatched_lines == 0
