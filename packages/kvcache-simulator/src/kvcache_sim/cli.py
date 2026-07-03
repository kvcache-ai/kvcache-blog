from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .calculator import load_models_data, models_by_id
from .formatting import render_json, render_table
from .progress import ProgressBar
from .simulator import DEFAULT_BUDGETS_GIB, DEFAULT_POLICIES, run_sweep
from .trace import parse_trace_file


def _parse_csv_numbers(value: str) -> list[float]:
    if not value:
        return []
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvcache-simulator", description="Analyze KV cache hit rate for JSONL traces.")
    subparsers = parser.add_subparsers(dest="command")

    def add_run_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--trace", required=True, help="Trace path (.jsonl or .jsonl.gz), or - for stdin")
        command.add_argument("--model", required=True, help="Model id from the bundled KV Cache Size Calculator model catalog")
        command.add_argument("--kv-precision", dest="kv_precision", default=None, help="KV precision id, e.g. bf16_fp16, fp8_int8, fp4_int4")
        command.add_argument("--precision", dest="kv_precision", help=argparse.SUPPRESS)
        command.add_argument("--indexer-precision", default=None, help="Indexer precision id for models with an indexer cache")
        command.add_argument("--include-draft-kv-cache", action="store_true", help="Include draft/MTP KV cache where the model config supports it")
        command.add_argument("--block-size", type=int, default=None, help="Fallback block size when trace records omit block_size; record block_size overrides this value")
        command.add_argument("--estimate-tokens", type=int, default=None, help="Override calculator token count used for bytes/token")
        command.add_argument("--budgets-gib", default=",".join(str(v) for v in DEFAULT_BUDGETS_GIB), help="Comma-separated GiB budgets")
        command.add_argument("--policies", default=",".join(DEFAULT_POLICIES), help="Comma-separated policies: fifo,lru,optimal")
        command.add_argument("--backend", choices=["cpp", "python"], default="cpp", help="Simulation backend (default: cpp)")
        command.add_argument("--jobs", type=int, default=1, help="Worker processes for the Python backend; ignored by the C++ backend")
        command.add_argument("--no-progress", action="store_true", help="Disable terminal progress output")
        command.add_argument("--format", choices=["table", "json"], default="table", help="Output format (default: table)")
        command.add_argument("--output", "-o", default="-", help="Output path, or - for stdout")
        command.add_argument("--models-yaml", default=None, help="Override models.yaml path")
        command.add_argument("--max-records", type=int, default=0, help="Stop after this many valid requests (debug/testing)")
        command.add_argument("--max-events", type=int, default=0, help="Stop after this many trace blocks (debug/testing)")

    run = subparsers.add_parser("run", help="Run hit-rate analysis over configured KV cache budgets")
    add_run_arguments(run)

    sweep = subparsers.add_parser("sweep", help="Alias for run; scans a set of KV cache memory budgets")
    add_run_arguments(sweep)

    list_models = subparsers.add_parser("list-models", help="List supported model ids")
    list_models.add_argument("--models-yaml", default=None, help="Override models.yaml path")

    return parser


def _write_output(text: str, output: str) -> None:
    if output == "-":
        print(text)
        return
    Path(output).write_text(text + "\n", encoding="utf-8")


def run_sweep_command(args: argparse.Namespace) -> int:
    progress = ProgressBar(enabled=(not args.no_progress and sys.stderr.isatty()))
    try:
        data = load_models_data(args.models_yaml)
        progress.update(0, 4, "reading trace")
        trace = parse_trace_file(args.trace, block_size=args.block_size, max_records=args.max_records, max_events=args.max_events)
        progress.update(1, 4, "trace loaded")
        result = run_sweep(
            trace,
            model_id=args.model,
            precision=args.kv_precision,
            indexer_precision=args.indexer_precision,
            budgets_gib=_parse_csv_numbers(args.budgets_gib),
            policies=_parse_csv_strings(args.policies),
            jobs=args.jobs,
            backend=args.backend,
            progress=progress.update,
            estimate_tokens=args.estimate_tokens,
            include_draft_kv_cache=args.include_draft_kv_cache,
            models_data=data,
        )
        rendered = render_json(result) if args.format == "json" else render_table(result)
        _write_output(rendered, args.output)
        progress.finish()
        return 0
    except Exception:
        progress.close()
        raise


def run_list_models(args: argparse.Namespace) -> int:
    data = load_models_data(args.models_yaml)
    models = models_by_id(data)
    for model in sorted(models.values(), key=lambda item: (item.get("family", ""), item.get("label", ""))):
        print(f"{model['id']}\t{model.get('label', model['id'])}\t{model.get('family', '')}\t{model.get('formula', '')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"run", "sweep"}:
        return run_sweep_command(args)
    if args.command == "list-models":
        return run_list_models(args)
    parser.print_help(sys.stderr)
    return 2
