from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Iterable, TextIO


@dataclass
class TraceData:
    ids: list[int]
    tokens: list[int]
    request_starts: list[int]
    block_size: int
    request_count: int
    unique_raw_blocks: int
    total_input_tokens: int
    average_input_tokens: float
    parse_errors: int
    skipped_records: int


def block_tokens(input_length: int, block_size: int, index: int, count: int) -> int:
    if count <= 0:
        return 0
    if input_length <= 0:
        return block_size
    if block_size <= 0:
        return max(1, round(input_length / count))
    remaining = input_length - index * block_size
    if remaining <= 0:
        return 1
    return max(1, min(block_size, remaining))


def _open_text(path: str | Path) -> TextIO:
    if str(path) == "-":
        return sys.stdin
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


class _Interner:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def intern(self, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("hash_ids must be decimal integers or strings, not booleans")
        if isinstance(value, int):
            key = str(value)
        elif isinstance(value, str):
            key = value
        else:
            raise ValueError("hash_ids must be decimal integers or strings")
        existing = self._ids.get(key)
        if existing is not None:
            return existing
        assigned = len(self._ids)
        self._ids[key] = assigned
        return assigned

    def __len__(self) -> int:
        return len(self._ids)


def parse_trace_lines(
    lines: Iterable[str],
    *,
    block_size: int | None = None,
    max_records: int = 0,
    max_events: int = 0,
) -> TraceData:
    fallback_block_size = int(block_size or 0)
    trace_block_size = 0
    interner = _Interner()
    raw_requests: list[tuple[list[int], int, list[int] | None]] = []
    request_count = 0
    parse_errors = 0
    skipped = 0
    records_without_block_size = 0
    invalid_block_size = 0
    inconsistent_block_size = 0
    missing_input_length = 0
    invalid_block_tokens = 0
    invalid_hash_id = 0
    raw_event_count = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if not isinstance(record, dict):
            skipped += 1
            continue
        hash_ids = record.get("hash_ids")
        if not isinstance(hash_ids, list) or not hash_ids:
            skipped += 1
            continue
        try:
            input_length = int(record.get("input_length", 0))
        except (TypeError, ValueError):
            input_length = 0
        if input_length <= 0:
            missing_input_length += 1
            continue

        has_record_block_size = "block_size" in record and record.get("block_size") is not None
        try:
            record_block_size = int(record.get("block_size", 0)) if has_record_block_size else 0
        except (TypeError, ValueError):
            record_block_size = 0
        if has_record_block_size and record_block_size <= 0:
            invalid_block_size += 1
            continue
        if record_block_size > 0:
            if not trace_block_size:
                trace_block_size = record_block_size
            if record_block_size != trace_block_size:
                inconsistent_block_size += 1
                continue
        else:
            records_without_block_size += 1

        parsed_ids: list[int] = []
        try:
            for hash_id in hash_ids:
                parsed_ids.append(interner.intern(hash_id))
        except ValueError:
            invalid_hash_id += 1
            continue

        explicit_block_tokens: list[int] | None = None
        if "block_tokens" in record and record.get("block_tokens") is not None:
            block_token_values = record.get("block_tokens")
            if not isinstance(block_token_values, list) or len(block_token_values) != len(parsed_ids):
                invalid_block_tokens += 1
                continue
            explicit_block_tokens = []
            valid_tokens = True
            for value in block_token_values:
                try:
                    token_count = int(value)
                except (TypeError, ValueError):
                    token_count = 0
                if token_count <= 0:
                    valid_tokens = False
                    break
                explicit_block_tokens.append(token_count)
            if not valid_tokens:
                invalid_block_tokens += 1
                continue

        raw_requests.append((parsed_ids, input_length, explicit_block_tokens))
        request_count += 1
        raw_event_count += len(parsed_ids)

        if (max_records and request_count >= max_records) or (max_events and raw_event_count >= max_events):
            break

    selected_block_size = trace_block_size or fallback_block_size
    if records_without_block_size and not selected_block_size:
        raise ValueError(f"Trace records without block_size need --block-size. Found {records_without_block_size} record(s).")
    if invalid_block_size:
        raise ValueError(f"Trace block_size must be a positive integer. Found {invalid_block_size} invalid record(s).")
    if inconsistent_block_size:
        raise ValueError(f"Trace block_size must be consistent. Found {inconsistent_block_size} record(s) with a different value.")
    if missing_input_length:
        raise ValueError(f"Trace records must include positive input_length. Found {missing_input_length} invalid record(s).")
    if invalid_block_tokens:
        raise ValueError(f"Trace block_tokens must be a positive integer list matching hash_ids. Found {invalid_block_tokens} invalid record(s).")
    if invalid_hash_id:
        raise ValueError(f"Trace hash_ids must be decimal integers or strings. Found {invalid_hash_id} invalid record(s).")
    if not request_count:
        raise ValueError("No valid trace records found. Each line must contain hash_ids, input_length, and block_size.")

    ids: list[int] = []
    tokens: list[int] = []
    request_starts: list[int] = []
    total_input_tokens = 0
    for parsed_ids, input_length, explicit_block_tokens in raw_requests:
        request_starts.append(len(ids))
        for index, int_id in enumerate(parsed_ids):
            tok = explicit_block_tokens[index] if explicit_block_tokens else block_tokens(input_length, selected_block_size, index, len(parsed_ids))
            ids.append(int_id)
            tokens.append(tok)
            total_input_tokens += tok

    request_starts.append(len(ids))
    return TraceData(
        ids=ids,
        tokens=tokens,
        request_starts=request_starts,
        block_size=selected_block_size,
        request_count=request_count,
        unique_raw_blocks=len(interner),
        total_input_tokens=total_input_tokens,
        average_input_tokens=total_input_tokens / request_count if request_count else 0.0,
        parse_errors=parse_errors,
        skipped_records=skipped,
    )


def parse_trace_file(
    path: str | Path,
    *,
    block_size: int | None = None,
    max_records: int = 0,
    max_events: int = 0,
) -> TraceData:
    handle = _open_text(path)
    close = handle is not sys.stdin
    try:
        return parse_trace_lines(handle, block_size=block_size, max_records=max_records, max_events=max_events)
    finally:
        if close:
            handle.close()
