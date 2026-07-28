from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kvcache_sim.calculator import calculate_cache_size, load_models_data, models_by_id
from kvcache_sim.cpp_backend import _build_path_for_source
from kvcache_sim.plan import build_execution_plan
from kvcache_sim.policies import simulate_policy
from kvcache_sim._resources import package_resource_path, user_temp_suffix
from kvcache_sim.simulator import run_sweep
from kvcache_sim.trace import parse_trace_file, parse_trace_lines


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]


def make_trace(*hash_paths: list[str], block_size: int = 1):
    lines = [
        json.dumps({
            "block_size": block_size,
            "hash_ids": path,
            "input_length": len(path) * block_size,
        })
        for path in hash_paths
    ]
    return parse_trace_lines(lines)


class CalculatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models_data = load_models_data()
        cls.models = models_by_id(cls.models_data)

    def test_standard_gqa_matches_web_calculator_constant(self) -> None:
        result = calculate_cache_size(
            self.models["qwen3-32b"],
            tokens=128000,
            precision="bf16_fp16",
            models_data=self.models_data,
        )

        self.assertEqual(result.bytes_per_token, 262144)
        self.assertAlmostEqual(result.total_gib, 31.25)

    def test_glm_5_2_indexer_precision_matches_web_calculator_constant(self) -> None:
        result = calculate_cache_size(
            self.models["glm-5.2"],
            tokens=128000,
            precision="fp8_int8",
            indexer_precision="fp4_int4",
            models_data=self.models_data,
        )

        self.assertEqual(result.indexer_precision, "fp4_int4")
        self.assertEqual(result.bytes_per_token, 46272)
        self.assertAlmostEqual(result.total_gib, 5.51605224609375)

    def test_minimax_m3_keeps_fixed_bf16_indexer_precision(self) -> None:
        result = calculate_cache_size(
            self.models["minimax-m3"],
            tokens=1048576,
            precision="fp8_int8",
            indexer_precision="fp4_int4",
            models_data=self.models_data,
        )

        self.assertEqual(result.indexer_precision, "bf16_fp16")
        self.assertEqual(result.indexer_precision_label, "BF16 / FP16")
        self.assertAlmostEqual(result.total_gib, 74.25)

    def test_kimi_k3_uses_mla_and_kda_state_layout(self) -> None:
        result = calculate_cache_size(
            self.models["kimi-k3"],
            tokens=1048576,
            precision="fp8_int8",
            include_linear_attention_state=True,
            models_data=self.models_data,
        )

        self.assertEqual(result.hit_rate_bytes_per_token, 13824)
        self.assertAlmostEqual(result.total_gib, 13.918510437011719)

    def test_kimi_k3_mla_cache_uses_exact_logical_token_count(self) -> None:
        result = calculate_cache_size(
            self.models["kimi-k3"],
            tokens=1,
            precision="fp8_int8",
            include_linear_attention_state=False,
            models_data=self.models_data,
        )

        self.assertEqual(result.total_bytes, 13824)

    def test_bundled_models_match_web_calculator_catalog_when_in_repo(self) -> None:
        web_catalog = REPO_ROOT / "data" / "kv_cache_calculator" / "models.yaml"
        bundled_catalog = PACKAGE_ROOT / "src" / "kvcache_sim" / "resources" / "models.yaml"
        if not web_catalog.exists():
            self.skipTest("web calculator catalog is not present")

        self.assertEqual(bundled_catalog.read_text(encoding="utf-8"), web_catalog.read_text(encoding="utf-8"))


class TraceParserTests(unittest.TestCase):
    def test_jsonl_trace_parses_required_fields(self) -> None:
        trace = make_trace(["A", "B"], ["A", "C"], block_size=64)

        self.assertEqual(trace.request_count, 2)
        self.assertEqual(trace.block_size, 64)
        self.assertEqual(trace.total_input_tokens, 256)
        self.assertEqual(trace.unique_raw_blocks, 3)

    def test_missing_block_size_needs_cli_fallback(self) -> None:
        line = json.dumps({"hash_ids": ["A"], "input_length": 1})

        with self.assertRaisesRegex(ValueError, "without block_size need --block-size"):
            parse_trace_lines([line])

        trace = parse_trace_lines([line], block_size=64)
        self.assertEqual(trace.block_size, 64)

    def test_trace_block_size_overrides_cli_fallback(self) -> None:
        lines = [
            json.dumps({"hash_ids": ["A", "B"], "input_length": 32}),
            json.dumps({"block_size": 16, "hash_ids": ["A", "B"], "input_length": 32}),
        ]

        trace = parse_trace_lines(lines, block_size=64)

        self.assertEqual(trace.block_size, 16)
        self.assertEqual(trace.tokens, [16, 16, 16, 16])

    def test_explicit_block_tokens_preserve_per_block_weights(self) -> None:
        line = json.dumps({
            "block_size": 64,
            "hash_ids": ["A", "B", "C"],
            "input_length": 130,
            "block_tokens": [64, 32, 34],
        })

        trace = parse_trace_lines([line])

        self.assertEqual(trace.tokens, [64, 32, 34])
        self.assertEqual(trace.total_input_tokens, 130)

    def test_gzip_jsonl_trace_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"block_size": 1, "hash_ids": ["A"], "input_length": 1}) + "\n")

            trace = parse_trace_file(path)

        self.assertEqual(trace.request_count, 1)
        self.assertEqual(trace.ids, [0])


class PrefixPolicyTests(unittest.TestCase):
    def test_context_aware_identity_does_not_reuse_same_raw_block_under_different_parent(self) -> None:
        trace = make_trace(["A", "B"], ["C", "B"])
        plan = build_execution_plan(trace)

        self.assertEqual(trace.unique_raw_blocks, 3)
        self.assertEqual(plan.unique_blocks, 4)
        self.assertEqual(len(set(plan.node_for_event)), 4)

    def test_middle_miss_stops_prefix_hit_but_later_blocks_still_enter_cache(self) -> None:
        trace = make_trace(["A", "B"], ["A", "C", "B"])
        plan = build_execution_plan(trace)

        lru = simulate_policy(plan, "lru", 2)
        fifo = simulate_policy(plan, "fifo", 2)

        self.assertEqual(lru.hitTokens, 1)
        self.assertEqual(lru.totalTokens, 3)
        self.assertEqual(lru.hitRate, 1 / 3)
        self.assertEqual(fifo.hitTokens, 1)
        self.assertEqual(fifo.totalTokens, 3)

    def test_repeated_prefix_can_hit_after_warmup(self) -> None:
        trace = make_trace(["A", "B"], ["A", "B"], ["A", "B"])
        plan = build_execution_plan(trace)

        result = simulate_policy(plan, "lru", 2)

        self.assertEqual(result.hitTokens, 4)
        self.assertEqual(result.totalTokens, 4)
        self.assertEqual(result.hitRate, 1)

    def test_optimal_bypasses_polluting_leaf(self) -> None:
        trace = make_trace(["A"], ["B"], ["A"])
        plan = build_execution_plan(trace)

        result = simulate_policy(plan, "optimal", 1)

        self.assertEqual(result.hitTokens, 1)
        self.assertEqual(result.totalTokens, 2)
        self.assertEqual(result.hitRate, 0.5)

    def test_underfilled_capacity_is_reported_when_not_full_before_measurement(self) -> None:
        trace = make_trace(["A"], ["A"], ["A"], ["A"])
        plan = build_execution_plan(trace)

        result = simulate_policy(plan, "lru", 2)

        self.assertEqual(result.measurementMode, "underfilled_at_window")


class SweepAndCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models_data = load_models_data()

    def test_run_sweep_uses_fixed_50_percent_window_and_omits_underfilled_larger_budgets(self) -> None:
        trace = make_trace(["A"], ["A"], ["B"], ["C"])
        result = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025, 0.0005, 0.001],
            policies=["fifo", "lru", "optimal"],
            backend="python",
            models_data=self.models_data,
        )

        self.assertEqual(result["metadata"]["warmupFraction"], 0.5)
        self.assertEqual(result["metadata"]["totalMeasuredTokens"], 2)
        self.assertEqual([point["cacheBlocks"] for point in result["points"]], [1])
        self.assertEqual(result["points"][0]["results"]["fifo"]["totalTokens"], 2)

    def test_run_sweep_sorts_budgets_before_early_breaks(self) -> None:
        trace = make_trace(["A"], ["B"], ["A"], ["B"])
        result = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.001, 0.00025, 0.00025],
            policies=["fifo"],
            backend="python",
            models_data=self.models_data,
        )

        self.assertEqual([point["cacheBlocks"] for point in result["points"]], [1])
        self.assertEqual(result["points"][0]["gib"], 0.00025)

    def test_multiprocess_sweep_matches_single_process(self) -> None:
        trace = make_trace(["A"], ["B"], ["A"], ["B"])
        serial = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025],
            jobs=1,
            backend="python",
            models_data=self.models_data,
        )
        parallel = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025],
            jobs=2,
            backend="python",
            models_data=self.models_data,
        )

        self.assertEqual(serial["points"], parallel["points"])

    def test_cpp_backend_matches_python_backend_when_compiler_is_available(self) -> None:
        if not shutil.which("c++"):
            self.skipTest("c++ compiler is not available")
        trace = make_trace(["A"], ["B"], ["A"], ["B"])
        python_result = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025],
            jobs=1,
            backend="python",
            models_data=self.models_data,
        )
        cpp_result = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025],
            backend="cpp",
            models_data=self.models_data,
        )

        self.assertEqual(cpp_result["metadata"]["backend"], "cpp")
        self.assertEqual(python_result["points"], cpp_result["points"])
        self.assertEqual(python_result["hitRateCeiling"], cpp_result["hitRateCeiling"])

    def test_cli_outputs_table_by_default_and_json_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            trace_path.write_text(
                "\n".join([
                    json.dumps({"block_size": 1, "hash_ids": ["A"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["B"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["A"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["B"], "input_length": 1}),
                ])
                + "\n",
                encoding="utf-8",
            )
            base = [
                sys.executable,
                "-m",
                "kvcache_sim",
                "sweep",
                "--trace",
                str(trace_path),
                "--model",
                "qwen3-32b",
                "--kv-precision",
                "bf16_fp16",
                "--budgets-gib",
                "0.00025",
                "--backend",
                "python",
            ]

            table = subprocess.run(base, cwd=Path(__file__).resolve().parents[1], check=True, text=True, capture_output=True)
            as_json = subprocess.run(base + ["--format", "json"], cwd=Path(__file__).resolve().parents[1], check=True, text=True, capture_output=True)

        self.assertIn("Budget", table.stdout)
        self.assertIn("FIFO hit", table.stdout)
        parsed = json.loads(as_json.stdout)
        self.assertEqual(parsed["metadata"]["modelId"], "qwen3-32b")
        self.assertEqual(parsed["points"][0]["cacheBlocks"], 1)
        self.assertIn("Measurement: hit rates use the last 50% of requests", table.stdout)
        self.assertIn("Speedup: 1.0x means no-cache prefill throughput", table.stdout)

    def test_cli_run_command_matches_sweep_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            trace_path.write_text(
                "\n".join([
                    json.dumps({"block_size": 1, "hash_ids": ["A"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["B"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["A"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["B"], "input_length": 1}),
                ])
                + "\n",
                encoding="utf-8",
            )
            args = [
                "--trace",
                str(trace_path),
                "--model",
                "qwen3-32b",
                "--kv-precision",
                "bf16_fp16",
                "--budgets-gib",
                "0.00025",
                "--backend",
                "python",
                "--format",
                "json",
            ]

            run_result = subprocess.run([sys.executable, "-m", "kvcache_sim", "run", *args], cwd=Path(__file__).resolve().parents[1], check=True, text=True, capture_output=True)
            sweep_result = subprocess.run([sys.executable, "-m", "kvcache_sim", "sweep", *args], cwd=Path(__file__).resolve().parents[1], check=True, text=True, capture_output=True)

        self.assertEqual(json.loads(run_result.stdout)["points"], json.loads(sweep_result.stdout)["points"])


class TempPathTests(unittest.TestCase):
    def test_temp_paths_include_user_suffix_when_available(self) -> None:
        suffix = user_temp_suffix()
        if not suffix:
            self.skipTest("platform does not expose os.getuid")

        resource_path = package_resource_path("models.yaml")
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "sim.cc"
            source.write_text("int main() { return 0; }\n", encoding="utf-8")
            binary_path = _build_path_for_source(source)

        if tempfile.gettempdir() in str(resource_path):
            self.assertIn(suffix, resource_path.name)
        self.assertIn(suffix, binary_path.name)


if __name__ == "__main__":
    unittest.main()
