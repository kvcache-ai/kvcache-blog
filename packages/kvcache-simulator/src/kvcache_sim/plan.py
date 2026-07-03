from __future__ import annotations

from dataclasses import dataclass
import math

from .trace import TraceData


@dataclass
class ExecutionPlan:
    ids: list[int]
    tokens: list[int]
    request_starts: list[int]
    node_for_event: list[int]
    next_request_for_event: list[int]
    parent: list[int]
    block_size: int
    request_count: int
    unique_raw_blocks: int
    unique_blocks: int
    warmup_requests: int
    total_measured_tokens: int
    total_input_tokens: int
    average_input_tokens: float


def build_execution_plan(trace: TraceData, warmup_fraction: float = 0.5) -> ExecutionPlan:
    node_for_event = [0] * len(trace.ids)
    parent = [0]
    edges: dict[tuple[int, int], int] = {}

    for request_index in range(trace.request_count):
        parent_node = 0
        start = trace.request_starts[request_index]
        end = trace.request_starts[request_index + 1]
        for index in range(start, end):
            key = (parent_node, trace.ids[index])
            node = edges.get(key)
            if node is None:
                node = len(parent)
                edges[key] = node
                parent.append(parent_node)
            node_for_event[index] = node
            parent_node = node

    never_request = trace.request_count + 1
    next_request_for_event = [never_request] * len(trace.ids)
    last_use = [never_request] * len(parent)
    for request_index in range(trace.request_count - 1, -1, -1):
        start = trace.request_starts[request_index]
        end = trace.request_starts[request_index + 1]
        for index in range(start, end):
            next_request_for_event[index] = last_use[node_for_event[index]]
        for index in range(start, end):
            last_use[node_for_event[index]] = request_index

    warmup_requests = max(0, min(trace.request_count, math.floor(trace.request_count * warmup_fraction)))
    total_measured_tokens = sum(
        trace.tokens[index]
        for request_index in range(warmup_requests, trace.request_count)
        for index in range(trace.request_starts[request_index], trace.request_starts[request_index + 1])
    )
    return ExecutionPlan(
        ids=trace.ids,
        tokens=trace.tokens,
        request_starts=trace.request_starts,
        node_for_event=node_for_event,
        next_request_for_event=next_request_for_event,
        parent=parent,
        block_size=trace.block_size,
        request_count=trace.request_count,
        unique_raw_blocks=trace.unique_raw_blocks,
        unique_blocks=max(0, len(parent) - 1),
        warmup_requests=warmup_requests,
        total_measured_tokens=total_measured_tokens,
        total_input_tokens=trace.total_input_tokens,
        average_input_tokens=trace.average_input_tokens,
    )
