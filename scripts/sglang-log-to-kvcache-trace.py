#!/usr/bin/env python3
"""Convert an SGLang request log into a KV Cache Lab upload trace (Mooncake schema).

Each ``Finish:`` line of an SGLang worker log already contains the tokenized
prompt (``input_ids=[...]``), so no tokenizer is needed. We split each prompt
into fixed-size blocks and assign every block a *prefix-aware* content hash:
the hash of all tokens up to and including that block. Two requests that share
a prompt prefix therefore share the leading block ids -- exactly the identity
that a RadixAttention / prefix KV cache exploits -- so the resulting trace
reproduces realistic cache reuse.

Output: JSONL, one record per request, in the Mooncake schema consumed by the
KV Cache Lab upload feature:

    {"id": "...", "timestamp": 1.78e9, "hash_ids": [int, ...],
     "input_length": N, "output_length": M}

Usage:
    python3 sglang-log-to-kvcache-trace.py LOG [LOG ...] \
        --block-size 64 --sort --jobs 32 --out trace.jsonl

The block size must match the trace granularity you want to evaluate; the lab
reads the generated ``block_size`` field and does not expose it as a UI knob.
With many log files, ``--jobs`` fans the (CPU-bound) block hashing across that
many worker processes; the parent then does a single global ``--sort`` by
request_received_ts before writing.
"""

import argparse
import hashlib
import json
import re
import struct
import sys
from multiprocessing import Pool

# Capture the integer list inside input_ids=[...] on a "Finish:" line.
INPUT_IDS_RE = re.compile(r"input_ids=\[([0-9,\s]*)\]")
RID_RE = re.compile(r"rid='([^']+)'")
RECEIVED_TS_RE = re.compile(r"'request_received_ts':\s*([0-9.]+)")
COMPLETION_RE = re.compile(r"'completion_tokens':\s*([0-9]+)")


def parse_int_list(blob):
    # blob is "163594, 20960, 35556, ..." -> list[int]
    return [int(tok) for tok in blob.split(",") if tok.strip()]


def block_hashes(token_ids, block_size):
    """Prefix-aware 64-bit hash per block (hash of the prefix up to block end)."""
    running = hashlib.blake2b(digest_size=8)
    ids = []
    for start in range(0, len(token_ids), block_size):
        block = token_ids[start : start + block_size]
        running.update(struct.pack(f"<{len(block)}I", *(t & 0xFFFFFFFF for t in block)))
        ids.append(int.from_bytes(running.copy().digest(), "little"))
    return ids


def iter_records(paths, start_ts=None, end_ts=None):
    handles = [sys.stdin] if not paths else [open(p, "r", errors="replace") for p in paths]
    try:
        for handle in handles:
            for line in handle:
                if "Finish: input_ids_len=" not in line:
                    continue
                # Filter on request_received_ts first (cheap) before parsing the
                # potentially huge input_ids list.
                ts = RECEIVED_TS_RE.search(line)
                ts_val = float(ts.group(1)) if ts else None
                if start_ts is not None or end_ts is not None:
                    if ts_val is None:
                        continue
                    if start_ts is not None and ts_val < start_ts:
                        continue
                    if end_ts is not None and ts_val >= end_ts:
                        continue
                m = INPUT_IDS_RE.search(line)
                if not m:
                    continue
                token_ids = parse_int_list(m.group(1))
                if not token_ids:
                    continue
                rid = RID_RE.search(line)
                comp = COMPLETION_RE.search(line)
                yield {
                    "rid": rid.group(1) if rid else None,
                    "ts": ts_val,
                    "input_ids": token_ids,
                    "completion": int(comp.group(1)) if comp else 0,
                }
    finally:
        for handle in handles:
            if handle is not sys.stdin:
                handle.close()


def convert_one(record, block_size):
    """Turn a parsed record into a (timestamp, json_line) pair."""
    ids = record["input_ids"]
    line = {
        "id": record["rid"],
        "timestamp": record["ts"],
        "block_size": block_size,
        "hash_ids": block_hashes(ids, block_size),
        "input_length": len(ids),
        "output_length": record["completion"],
    }
    return record["ts"], record["rid"] or "", len(ids), json.dumps(line)


def process_file(task):
    """Worker entry point: convert a single log file to a list of rows.

    Returns (timestamp, rid, input_len, json_line) tuples so the parent can do a
    global timestamp sort and tally token counts without re-parsing. The rid is
    the sort tie-breaker, making output deterministic regardless of -j.
    """
    path, block_size, start_ts, end_ts = task
    return [convert_one(r, block_size) for r in iter_records([path], start_ts, end_ts)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="*", help="SGLang log file(s); reads stdin if omitted")
    parser.add_argument("--block-size", type=int, default=64, help="tokens per cache block written into each trace row (default 64; set to your actual service granularity)")
    parser.add_argument("--out", default="-", help="output JSONL path, or - for stdout")
    parser.add_argument("--sort", action="store_true", help="sort by request_received_ts (recommended for cache order)")
    parser.add_argument("--start-ts", type=float, default=None, help="keep only requests with request_received_ts >= this epoch seconds")
    parser.add_argument("--end-ts", type=float, default=None, help="keep only requests with request_received_ts < this epoch seconds")
    parser.add_argument("-j", "--jobs", type=int, default=32,
                        help="worker processes for parallel file conversion (default 32); "
                             "blake2b block hashing is CPU-bound so processes (not threads) scale it")
    args = parser.parse_args()

    # rows are (timestamp, input_len, json_line). Parallelize across files when
    # we have real files and more than one job; fall back to serial for stdin or -j1.
    jobs = max(1, args.jobs)
    if args.logs and jobs > 1 and len(args.logs) > 1:
        jobs = min(jobs, len(args.logs))
        with Pool(processes=jobs) as pool:
            rows = [row for chunk in pool.imap_unordered(
                        process_file, [(p, args.block_size, args.start_ts, args.end_ts) for p in args.logs])
                        for row in chunk]
    else:
        rows = [convert_one(r, args.block_size) for r in iter_records(args.logs, args.start_ts, args.end_ts)]

    if args.sort:
        rows.sort(key=lambda row: (row[0] is None, row[0], row[1]))

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    n = 0
    total_tokens = 0
    try:
        for _ts, _rid, ilen, line in rows:
            out.write(line + "\n")
            n += 1
            total_tokens += ilen
    finally:
        if out is not sys.stdout:
            out.close()

    avg = total_tokens / n if n else 0
    sys.stderr.write(f"[ok] {n} requests, {total_tokens} input tokens, avg {avg:.0f} tokens/req, "
                     f"block_size={args.block_size}, jobs={jobs}\n")


if __name__ == "__main__":
    main()
