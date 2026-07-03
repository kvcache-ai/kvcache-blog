from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq

from .plan import ExecutionPlan


@dataclass
class PolicyResult:
    policy: str
    cacheBlocks: int
    warmupRequests: int
    measurementStartRequest: int | None
    measurementMode: str
    hitTokens: int
    totalTokens: int
    hitRate: float

    def to_json(self) -> dict[str, int | float | str | None]:
        return asdict(self)


def _finish(policy: str, capacity: int, plan: ExecutionPlan, hit_tokens: int, total_tokens: int, mode: str = "fixed_window") -> PolicyResult:
    return PolicyResult(
        policy=policy,
        cacheBlocks=capacity,
        warmupRequests=plan.warmup_requests,
        measurementStartRequest=plan.warmup_requests if mode == "fixed_window" else None,
        measurementMode=mode,
        hitTokens=hit_tokens,
        totalTokens=total_tokens,
        hitRate=(hit_tokens / total_tokens) if total_tokens else 0.0,
    )


def _no_cache(policy: str, capacity: int, plan: ExecutionPlan) -> PolicyResult:
    return _finish(policy, capacity, plan, 0, plan.total_measured_tokens, "fixed_window")


def _underfilled(policy: str, capacity: int, plan: ExecutionPlan) -> PolicyResult:
    return _finish(policy, capacity, plan, 0, plan.total_measured_tokens, "underfilled_at_window")


def simulate_ceiling(plan: ExecutionPlan) -> PolicyResult:
    seen = bytearray(len(plan.parent))
    hit_tokens = 0
    total_tokens = 0
    for request_index in range(plan.request_count):
        start = plan.request_starts[request_index]
        end = plan.request_starts[request_index + 1]
        measured = request_index >= plan.warmup_requests
        prefix_alive = True
        for index in range(start, end):
            node = plan.node_for_event[index]
            hit = seen[node] == 1
            if measured:
                total_tokens += plan.tokens[index]
                if prefix_alive and hit:
                    hit_tokens += plan.tokens[index]
            if not hit:
                prefix_alive = False
        for index in range(start, end):
            seen[plan.node_for_event[index]] = 1
    return _finish("ceiling", max(plan.unique_blocks, 1), plan, hit_tokens, total_tokens)


def simulate_fifo(plan: ExecutionPlan, capacity: int) -> PolicyResult:
    if capacity <= 0 or not plan.ids:
        return _no_cache("fifo", capacity, plan)
    in_cache = bytearray(len(plan.parent))
    queue: list[int] = []
    head = 0
    cache_size = 0
    full_before_measurement = False
    hit_tokens = 0
    total_tokens = 0

    for request_index in range(plan.request_count):
        if request_index >= plan.warmup_requests and not full_before_measurement:
            return _underfilled("fifo", capacity, plan)
        start = plan.request_starts[request_index]
        end = plan.request_starts[request_index + 1]
        measured = request_index >= plan.warmup_requests
        prefix_alive = True
        for index in range(start, end):
            node = plan.node_for_event[index]
            hit = in_cache[node] == 1
            if measured:
                total_tokens += plan.tokens[index]
                if prefix_alive and hit:
                    hit_tokens += plan.tokens[index]
            if not hit:
                prefix_alive = False
        for index in range(start, end):
            node = plan.node_for_event[index]
            if in_cache[node]:
                continue
            if cache_size >= capacity:
                while head < len(queue):
                    victim = queue[head]
                    head += 1
                    if in_cache[victim]:
                        in_cache[victim] = 0
                        cache_size -= 1
                        break
            if cache_size < capacity:
                in_cache[node] = 1
                cache_size += 1
                queue.append(node)
                if cache_size >= capacity and request_index < plan.warmup_requests:
                    full_before_measurement = True
            if head > 1_000_000 and head * 2 > len(queue):
                del queue[:head]
                head = 0

    if not full_before_measurement or total_tokens <= 0:
        return _underfilled("fifo", capacity, plan)
    return _finish("fifo", capacity, plan, hit_tokens, total_tokens)


class _LeafHeap:
    def __init__(self, max_heap: bool) -> None:
        self.max_heap = max_heap
        self.items: list[tuple[int, int, int, int]] = []

    def push(self, node: int, key: int, version: int) -> None:
        if self.max_heap:
            item = (-key, -node, version, node)
        else:
            item = (key, -node, version, node)
        heapq.heappush(self.items, item)

    def pop(self) -> tuple[int, int, int] | None:
        if not self.items:
            return None
        key_part, _neg_node, version, node = heapq.heappop(self.items)
        key = -key_part if self.max_heap else key_part
        return node, key, version


def simulate_trie_policy(plan: ExecutionPlan, capacity: int, *, optimal: bool) -> PolicyResult:
    policy = "optimal" if optimal else "lru"
    if capacity <= 0 or not plan.ids:
        return _no_cache(policy, capacity, plan)

    node_count = len(plan.parent)
    present = bytearray(node_count)
    child_count = [0] * node_count
    state_version = [0] * node_count
    state_key = [0] * node_count
    protected_mark = [0] * node_count
    heap = _LeafHeap(max_heap=optimal)
    present[0] = 1
    cache_size = 0
    clock = 0
    mark_value = 1
    full_before_measurement = False
    hit_tokens = 0
    total_tokens = 0

    def push_leaf(node: int) -> None:
        if node == 0 or not present[node] or child_count[node] != 0:
            return
        state_version[node] += 1
        heap.push(node, state_key[node], state_version[node])

    def touch_lru(node: int) -> None:
        nonlocal clock
        if node == 0 or not present[node]:
            return
        clock += 1
        state_key[node] = clock
        push_leaf(node)

    def update_optimal(node: int, next_use: int) -> None:
        if node == 0 or not present[node]:
            return
        state_key[node] = next_use
        push_leaf(node)

    def add_node(node: int, event_index: int) -> None:
        nonlocal cache_size
        parent = plan.parent[node]
        present[node] = 1
        cache_size += 1
        child_count[parent] += 1
        if optimal:
            update_optimal(node, plan.next_request_for_event[event_index])
        else:
            touch_lru(node)

    def restore(skipped: list[tuple[int, int, int]]) -> None:
        for node, key, version in skipped:
            heap.push(node, key, version)

    def evict_leaf(candidate_key: int) -> bool:
        nonlocal cache_size
        skipped: list[tuple[int, int, int]] = []
        while True:
            top = heap.pop()
            if top is None:
                restore(skipped)
                return False
            node, key, version = top
            if not present[node] or child_count[node] != 0 or state_version[node] != version or state_key[node] != key:
                continue
            if protected_mark[node] == mark_value:
                skipped.append(top)
                continue
            if optimal and key <= candidate_key:
                heap.push(node, key, version)
                restore(skipped)
                return False
            present[node] = 0
            cache_size -= 1
            parent = plan.parent[node]
            child_count[parent] -= 1
            if parent > 0 and present[parent] and child_count[parent] == 0:
                push_leaf(parent)
            restore(skipped)
            return True

    def mark_protected_path(start: int, end: int) -> None:
        nonlocal mark_value
        mark_value += 1
        if mark_value == 0x7FFFFFFF:
            protected_mark[:] = [0] * len(protected_mark)
            mark_value = 1
        protected_mark[0] = mark_value
        for index in range(start, end):
            node = plan.node_for_event[index]
            if present[node]:
                protected_mark[node] = mark_value

    for request_index in range(plan.request_count):
        if request_index >= plan.warmup_requests and not full_before_measurement:
            return _underfilled(policy, capacity, plan)
        start = plan.request_starts[request_index]
        end = plan.request_starts[request_index + 1]
        measured = request_index >= plan.warmup_requests
        prefix_alive = True
        for index in range(start, end):
            node = plan.node_for_event[index]
            hit = present[node] == 1
            if measured:
                total_tokens += plan.tokens[index]
                if prefix_alive and hit:
                    hit_tokens += plan.tokens[index]
            if prefix_alive and hit:
                if optimal:
                    update_optimal(node, plan.next_request_for_event[index])
                else:
                    touch_lru(node)
            elif not hit:
                prefix_alive = False

        mark_protected_path(start, end)
        for index in range(start, end):
            node = plan.node_for_event[index]
            if present[node]:
                continue
            if cache_size >= capacity:
                candidate_key = plan.next_request_for_event[index] if optimal else 0
                if not evict_leaf(candidate_key):
                    break
            if cache_size < capacity and present[plan.parent[node]]:
                add_node(node, index)
                if cache_size >= capacity and request_index < plan.warmup_requests:
                    full_before_measurement = True
            else:
                break

    if not full_before_measurement or total_tokens <= 0:
        return _underfilled(policy, capacity, plan)
    return _finish(policy, capacity, plan, hit_tokens, total_tokens)


def simulate_policy(plan: ExecutionPlan, policy: str, capacity: int) -> PolicyResult:
    if policy == "fifo":
        return simulate_fifo(plan, capacity)
    if policy == "lru":
        return simulate_trie_policy(plan, capacity, optimal=False)
    if policy == "optimal":
        return simulate_trie_policy(plan, capacity, optimal=True)
    raise ValueError(f"Unsupported policy: {policy}")
