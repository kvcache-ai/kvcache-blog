import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  allModelSettings,
  buildTrace,
  infiniteCacheReuse,
  kvArchitectureGroups,
  loadModelsData,
  modelSweepKey,
  normalizeBailianRecord,
  normalizeExgenticAgentRecord,
  normalizeLmcacheAgenticRecord,
  normalizeMooncakeRecord,
  normalizeRagPulseRecord,
  normalizeWekaSessionRecord,
  precomputeSweep,
  selectModels,
} from "../scripts/lib/kv-cache-lab-traces.mjs";
import {
  PRECOMPUTED_SCHEMA_VERSION,
  SIMULATION_SEMANTICS_VERSION,
  compactPrecomputed,
  compactTraceMatchesSimulation,
  simulationResultsFromCompactTrace,
} from "../scripts/lib/kv-cache-lab-precomputed.mjs";
import { mergePrecomputed } from "../scripts/kv-cache-lab-merge-precomputed.mjs";

const BYTES_PER_GIB = 1024 ** 3;
const here = path.dirname(fileURLToPath(import.meta.url));

const tinyModel = {
  id: "tiny-standard",
  label: "Tiny Standard",
  formula: "standard_gqa",
  default_tokens: 16,
  fields: {
    num_hidden_layers: 1,
    num_key_value_heads: 1,
    head_dim: 1,
  },
};

test("Mooncake parser uses 512-token source blocks and partial last-block tokens", () => {
  const request = normalizeMooncakeRecord(
    { timestamp: 3, input_length: 1025, output_length: 7, hash_ids: [10, 11, 12] },
    { id: "mooncake_fast25", nativeBlockSize: 512 },
  );

  assert.equal(request.inputTokens, 1025);
  assert.deepEqual(
    request.inputBlocks.map((block) => block.tokens),
    [512, 512, 1],
  );
  assert.equal(request.inputBlocks[0].id, "mooncake_fast25:10");
});

test("Bailian parser keeps 16-token bucket accounting", () => {
  const request = normalizeBailianRecord(
    { chat_id: 1, parent_chat_id: -1, timestamp: 0.2, input_length: 17, output_length: 3, type: "text", turn: 1, hash_ids: [1, 2] },
    { id: "bailian_qwen_trace_a", nativeBlockSize: 16 },
  );

  assert.deepEqual(
    request.inputBlocks.map((block) => block.tokens),
    [16, 1],
  );
  assert.equal(request.type, "text");
});

test("RAGPulse parser namespaces hash-id categories", () => {
  const request = normalizeRagPulseRecord(
    {
      timestamp: "82",
      input_length: 1000,
      output_length: 1,
      session_id: "session-1",
      hash_ids: {
        sys_prompt: [1],
        passages_ids: [1],
        history: [2],
        web_search: [],
        user_input: [1],
      },
    },
    { nativeBlockSize: 512 },
  );

  assert.deepEqual(
    request.inputBlocks.map((block) => block.id),
    ["sys_prompt:1", "passages_ids:1", "history:2", "user_input:1"],
  );
});

test("agent text parsers are deterministic approximate block converters", () => {
  const lmcache = normalizeLmcacheAgenticRecord(
    {
      session_id: "s1",
      input: [
        { role: "system", content: "You are coding." },
        { role: "user", content: "Fix the failing test." },
      ],
      output_length: 32,
      pre_gap: 1.5,
    },
    { id: "lmcache_agentic_sample", nativeBlockSize: 64 },
  );
  const lmcacheAgain = normalizeLmcacheAgenticRecord(
    {
      session_id: "s1",
      input: [
        { role: "system", content: "You are coding." },
        { role: "user", content: "Fix the failing test." },
      ],
      output_length: 32,
      pre_gap: 1.5,
    },
    { id: "lmcache_agentic_sample", nativeBlockSize: 64 },
  );
  const exgentic = normalizeExgenticAgentRecord(
    {
      session_id: "e1",
      spans: JSON.stringify([{ name: "chat", input: "observe", output: "act" }]),
      collected_at: "2026-05-07T09:04:59.116973",
    },
    { id: "exgentic_agent_sample", nativeBlockSize: 64 },
  );

  assert.deepEqual(lmcache.inputBlocks, lmcacheAgain.inputBlocks);
  assert.ok(lmcache.inputTokens > 0);
  assert.ok(exgentic.inputBlocks[0].id.startsWith("exgentic_agent_sample:e1:"));
});

test("Weka session parser expands native hash-id request streams", () => {
  const requests = normalizeWekaSessionRecord(
    {
      id: "trace-a",
      block_size: 64,
      hash_id_scope: "local",
      requests: [
        { t: 0, model: "claude-opus", in: 130, out: 7, hash_ids: [1, 2, 3] },
        { t: 3, model: "claude-opus", in: 64, out: 2, hash_ids: [2] },
      ],
    },
    { id: "semianalysis_weka_no_subagents", nativeBlockSize: 64 },
  );

  assert.equal(requests.length, 2);
  assert.deepEqual(
    requests[0].inputBlocks.map((block) => block.tokens),
    [64, 64, 2],
  );
  assert.equal(requests[0].inputBlocks[0].id, "semianalysis_weka_no_subagents:trace-a:1");
  assert.equal(requests[1].inputBlocks[0].id, "semianalysis_weka_no_subagents:trace-a:2");
});

test("Weka session parser flattens nested sub-agent requests with parent namespace", () => {
  const requests = normalizeWekaSessionRecord(
    {
      id: "trace-parent",
      block_size: 64,
      requests: [
        { t: 10, model: "claude-opus", in: 64, out: 1, hash_ids: [1] },
        {
          t: 20,
          type: "subagent",
          requests: [
            { t: 2, model: "claude-haiku", in: 65, out: 1, hash_ids: [100, 101] },
          ],
        },
      ],
    },
    { id: "kv_cache_tester_claude_code", nativeBlockSize: 64 },
  );

  assert.equal(requests.length, 2);
  assert.equal(requests[1].timestamp, 22);
  assert.deepEqual(
    requests[1].inputBlocks.map((block) => block.id),
    ["kv_cache_tester_claude_code:trace-parent:100", "kv_cache_tester_claude_code:trace-parent:101"],
  );
  assert.deepEqual(
    requests[1].inputBlocks.map((block) => block.tokens),
    [64, 1],
  );
});

test("infinite-cache reuse ceiling uses the fixed measurement window", () => {
  const trace = buildTrace(
    { id: "fixture", label: "Fixture", nativeBlockSize: 1, sourceKind: "hash" },
    [
      { timestamp: 0, inputBlocks: [{ id: "A", tokens: 1 }], appendBlocks: [] },
      { timestamp: 1, inputBlocks: [{ id: "A", tokens: 1 }], appendBlocks: [] },
      { timestamp: 2, inputBlocks: [{ id: "B", tokens: 1 }], appendBlocks: [] },
    ],
  );

  const ceiling = infiniteCacheReuse(trace, { warmupRequests: 1 });

  assert.equal(ceiling.warmupRequests, 1);
  assert.equal(ceiling.hitTokens, 1);
  assert.equal(ceiling.totalTokens, 2);
  assert.equal(ceiling.hitRate, 1 / 2);
});

test("native full-precompute optimal uses Belady bypass admission", (t) => {
  try {
    execFileSync("c++", ["--version"], { stdio: "ignore" });
  } catch {
    t.skip("c++ is unavailable");
    return;
  }

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "kv-cache-lab-native-"));
  try {
    const binary = path.join(tmp, "kv-cache-lab-native-sim");
    execFileSync("c++", [
      "-std=c++17",
      "-O2",
      path.resolve(here, "../scripts/kv-cache-lab-native-sim.cc"),
      "-o",
      binary,
    ]);

    const ids = Buffer.alloc(3 * 4);
    [0, 1, 0].forEach((id, index) => ids.writeUInt32LE(id, index * 4));
    const tokens = Buffer.alloc(3 * 2);
    [1, 1, 1].forEach((token, index) => tokens.writeUInt16LE(token, index * 2));
    const next = Buffer.alloc(3 * 4);
    [2, 4, 4].forEach((nextUse, index) => next.writeUInt32LE(nextUse, index * 4));
    const requestEnds = Buffer.alloc(3 * 4);
    [1, 2, 3].forEach((end, index) => requestEnds.writeUInt32LE(end, index * 4));

    fs.writeFileSync(path.join(tmp, "ids.u32.bin"), ids);
    fs.writeFileSync(path.join(tmp, "tokens.u16.bin"), tokens);
    fs.writeFileSync(path.join(tmp, "next.u32.bin"), next);
    fs.writeFileSync(path.join(tmp, "request-ends.u32.bin"), requestEnds);

    const output = execFileSync(binary, [
      "--policy",
      "optimal",
      "--ids",
      path.join(tmp, "ids.u32.bin"),
      "--tokens",
      path.join(tmp, "tokens.u16.bin"),
      "--next",
      path.join(tmp, "next.u32.bin"),
      "--request-ends",
      path.join(tmp, "request-ends.u32.bin"),
      "--total-blocks",
      "3",
      "--warmup-event-start",
      "0",
      "--capacity",
      "1",
      "--request-count",
      "3",
      "--warmup-requests",
      "1",
    ], { encoding: "utf8" });
    const result = JSON.parse(output);

    // With capacity 1 on A, B, A, Belady-with-bypass skips B and hits the final
    // A. Measurement starts after A first fills the cache, so B and the final A
    // are in the denominator.
    assert.equal(result.hitTokens, 1);
    assert.equal(result.totalTokens, 2);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("full-precompute rebases copied event-cache paths", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "kv-cache-lab-rebase-"));
  try {
    const traceDir = path.join(tmp, "kv-cache-lab-full", "bailian_qwen_trace_a");
    fs.mkdirSync(traceDir, { recursive: true });
    const files = {
      idsPath: "ids.bin",
      tokensPath: "tokens.u16.bin",
      requestEndsPath: "request-ends.u32.bin",
      nextPath: "next.u32.bin",
    };
    Object.values(files).forEach((file) => fs.writeFileSync(path.join(traceDir, file), Buffer.alloc(4)));
    fs.writeFileSync(path.join(traceDir, "events.json"), JSON.stringify({
      id: "bailian_qwen_trace_a",
      label: "Fixture",
      scenario: "Fixture",
      sourceKind: "hash",
      blockSize: 16,
      requestCount: 1,
      warmupRequests: 0,
      warmupEventStart: 0,
      totalBlocks: 1,
      totalInputTokens: 16,
      averageInputTokens: 16,
      uniqueBlocks: 1,
      ...Object.fromEntries(Object.keys(files).map((field) => [field, `/stale/machine/${files[field]}`])),
    }));

    const output = execFileSync("node", [
      path.resolve(here, "../scripts/kv-cache-lab-full-precompute.mjs"),
      "--events-only",
      "--trace",
      "bailian_qwen_trace_a",
      "--cache-dir",
      tmp,
      "--output",
      path.join(tmp, "unused.json"),
    ], { encoding: "utf8" });
    const metadata = JSON.parse(output).metadata;
    for (const [field, file] of Object.entries(files)) {
      assert.equal(metadata[field], path.join(traceDir, file));
    }
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("precompute sweep is deterministic and uses source-native block size", () => {
  const trace = buildTrace(
    { id: "fixture", label: "Fixture", nativeBlockSize: 16, sourceKind: "hash" },
    [
      { timestamp: 0, inputBlocks: [{ id: "A", tokens: 16 }], appendBlocks: [] },
      { timestamp: 1, inputBlocks: [{ id: "B", tokens: 16 }], appendBlocks: [] },
      { timestamp: 2, inputBlocks: [{ id: "A", tokens: 16 }], appendBlocks: [] },
      { timestamp: 3, inputBlocks: [{ id: "B", tokens: 16 }], appendBlocks: [] },
    ],
  );
  const settings = {
    precision: "bf16_fp16",
    blockSize: 16,
    capacityGiBValues: [64 / BYTES_PER_GIB, 128 / BYTES_PER_GIB],
    warmupFraction: 0.5,
  };

  const first = precomputeSweep(trace, tinyModel, settings, { generatedAt: "fixed" });
  const second = precomputeSweep(trace, tinyModel, settings, { generatedAt: "fixed" });

  assert.deepEqual(first, second);
  assert.equal(first.blockSize, 16);
  assert.equal(first.points.length, 1);
  assert.equal(first.points[0].results.lru.hitRate, 0);
  assert.equal(first.reuseCeiling, 1);
  assert.ok(
    first.points.every((point) =>
      Object.values(point.results).every((result) => result.measurementMode !== "ceiling_no_pressure"),
    ),
  );
});

test("model sweep keys encode model and precision settings", () => {
  assert.equal(
    modelSweepKey({ modelId: "qwen3-32b", precision: "bf16_fp16", includeDraftKvCache: false }),
    "qwen3-32b|precision=bf16_fp16|indexer=|draft=0",
  );
  assert.equal(
    modelSweepKey({ modelId: "deepseek-v4-pro", precision: "fp8_int8", indexerPrecision: "fp4_int4", includeDraftKvCache: true }),
    "deepseek-v4-pro|precision=fp8_int8|indexer=fp4_int4|draft=1",
  );
});

test("all model settings mirror calculator precision/indexer/draft controls", () => {
  const dsaModel = {
    id: "tiny-dsa",
    label: "Tiny DSA",
    formula: "dsa_mla",
    fields: {
      num_hidden_layers: 2,
      kv_lora_rank: 4,
      qk_rope_head_dim: 2,
      index_head_dim: 1,
      num_nextn_predict_layers: 1,
    },
  };
  const settings = allModelSettings([tinyModel, dsaModel], {
    precisionOptions: [{ id: "bf16_fp16" }, { id: "fp8_int8" }, { id: "fp4_int4" }],
    indexerPrecisionOptions: [{ id: "bf16_fp16" }, { id: "fp8_int8" }, { id: "fp4_int4" }],
  });

  const standardSettings = settings.filter((setting) => setting.modelId === "tiny-standard");
  const dsaSettings = settings.filter((setting) => setting.modelId === "tiny-dsa");

  assert.equal(standardSettings.length, 3);
  assert.deepEqual(standardSettings.map((setting) => setting.precision), ["bf16_fp16", "fp8_int8", "fp4_int4"]);
  assert.ok(standardSettings.every((setting) => setting.indexerPrecision === undefined));
  assert.ok(standardSettings.every((setting) => setting.includeDraftKvCache === false));

  assert.equal(dsaSettings.length, 18);
  assert.deepEqual(dsaSettings[0], {
    modelId: "tiny-dsa",
    precision: "bf16_fp16",
    indexerPrecision: "bf16_fp16",
    includeDraftKvCache: false,
  });
  assert.ok(dsaSettings.some((setting) => setting.precision === "fp8_int8" && setting.indexerPrecision === "fp4_int4" && setting.includeDraftKvCache === true));
});

test("selected production families dedupe to expected KV architecture settings", () => {
  const modelsData = loadModelsData(path.resolve(here, "../data/kv_cache_calculator/models.yaml"));
  const selectedModels = selectModels(modelsData.models, {
    includeFamilies: ["DeepSeek", "GLM", "Kimi", "MiMo", "MiniMax"],
  });
  const groups = kvArchitectureGroups(selectedModels);
  const settingCount = groups.reduce((total, group) => total + allModelSettings([group.models[0]], {
    precisionOptions: modelsData.precision_options,
    indexerPrecisionOptions: modelsData.indexer_precision_options,
    includeDraftKvCache: false,
  }).length, 0);
  const groupedLabels = groups.map((group) => group.models.map((model) => model.label).join(" / "));

  assert.equal(selectedModels.length, 17);
  assert.equal(groups.length, 11);
  assert.equal(settingCount, 69);
  assert.ok(groupedLabels.includes("DeepSeek V3 / DeepSeek R1"));
  assert.ok(groupedLabels.includes("GLM-5 / GLM-5.1"));
  assert.ok(groupedLabels.includes("GLM-5.2"));
  assert.ok(groupedLabels.includes("Kimi K2.5 / Kimi K2.6"));
  assert.ok(groupedLabels.includes("MiniMax M2 / MiniMax M2.1 / MiniMax M2.5 / MiniMax M2.7"));
  assert.ok(groupedLabels.includes("MiniMax M3"));
  assert.ok(selectedModels.every((model) => !["Qwen", "Cohere", "Llama", "Gemma"].some((family) => model.family.startsWith(family))));
});

test("merge precomputed data preserves existing sweeps and adds inputs", () => {
  const result = (hitTokens) => ({
    policy: "fifo",
    cacheBlocks: 10,
    warmupRequests: 1,
    measurementMode: "fixed_window",
    hitTokens,
    totalTokens: 100,
    hitRate: hitTokens / 100,
  });
  const sweep = (modelId, hitTokens) => ({
    modelId,
    bytesPerToken: 8,
    points: [{
      gib: 1,
      cacheBlocks: 10,
      results: {
        fifo: result(hitTokens),
        lru: { ...result(hitTokens + 10), policy: "lru" },
        optimal: { ...result(hitTokens + 20), policy: "optimal" },
      },
    }],
  });
  const base = {
    metadata: { mode: "base", warmup_fraction: 0.5, policies: ["fifo", "lru", "optimal"], sources: { a: "base" } },
    traces: {
      trace: {
        id: "trace",
        nativeBlockSize: 64,
        summary: { warmupRequests: 1 },
        modelSweeps: {
          "model-a|precision=bf16_fp16|indexer=|draft=0": sweep("model-a", 10),
        },
      },
    },
  };
  const input = {
    metadata: { mode: "input", warmup_fraction: 0.5, policies: ["fifo", "lru", "optimal"], sources: { b: "input" } },
    traces: {
      trace: {
        id: "trace",
        label: "Trace",
        nativeBlockSize: 64,
        summary: { warmupRequests: 1 },
        modelSweeps: {
          "model-b|precision=bf16_fp16|indexer=|draft=0": sweep("model-b", 10),
        },
      },
    },
  };

  const merged = mergePrecomputed(base, [input]);

  assert.equal(merged.metadata.mode, "input");
  assert.equal(merged.metadata.schema_version, PRECOMPUTED_SCHEMA_VERSION);
  assert.equal(merged.metadata.simulation_semantics, SIMULATION_SEMANTICS_VERSION);
  assert.deepEqual(merged.metadata.sources, { a: "base", b: "input" });
  assert.equal(merged.traces.trace.label, "Trace");
  assert.equal(Object.keys(merged.traces.trace.settings).length, 2);
  assert.deepEqual(merged.traces.trace.capacityResults[10], [10, 20, 30]);
});

test("compact precomputed data deduplicates capacity results without losing hit tokens", () => {
  const policyResult = (policy, hitTokens) => ({ policy, hitTokens, totalTokens: 100, hitRate: hitTokens / 100 });
  const point = {
    gib: 1,
    cacheBlocks: 10,
    results: {
      fifo: policyResult("fifo", 10),
      lru: policyResult("lru", 20),
      optimal: policyResult("optimal", 30),
    },
  };
  const compact = compactPrecomputed({
    metadata: { warmup_fraction: 0.5, policies: ["fifo", "lru", "optimal"] },
    traces: {
      trace: {
        nativeBlockSize: 64,
        summary: { requests: 2, totalBlocks: 2, totalInputTokens: 128, warmupRequests: 1 },
        modelSweeps: {
          "model-a|precision=bf16_fp16|indexer=|draft=0": { bytesPerToken: 8, points: [point] },
          "model-b|precision=bf16_fp16|indexer=|draft=0": { bytesPerToken: 8, points: [point] },
        },
      },
    },
  });
  const trace = compact.traces.trace;

  assert.equal(Object.keys(trace.settings).length, 2);
  assert.equal(Object.keys(trace.capacityResults).length, 1);
  assert.deepEqual(trace.capacityResults[10], [10, 20, 30]);
  assert.equal(trace.summary.totalMeasuredTokens, 100);
  const restored = simulationResultsFromCompactTrace(trace);
  assert.equal(restored.get("optimal|10").hitRate, 0.3);
});

test("versioned compact cache only matches the same semantics, warmup, and trace shape", () => {
  const dataset = compactPrecomputed({
    metadata: { warmup_fraction: 0.5, policies: ["fifo", "lru", "optimal"] },
    traces: {
      trace: {
        nativeBlockSize: 64,
        summary: {
          requests: 2,
          totalBlocks: 2,
          totalInputTokens: 128,
          totalMeasuredTokens: 64,
          warmupRequests: 1,
        },
        settings: { "model-a|precision=bf16_fp16|indexer=|draft=0": [8, []] },
        capacityResults: {},
      },
    },
  });
  const runtime = {
    blockSize: 64,
    requestCount: 2,
    totalBlocks: 2,
    totalInputTokens: 128,
    warmupRequests: 1,
    warmupFraction: 0.5,
  };

  assert.equal(compactTraceMatchesSimulation(dataset.traces.trace, dataset.metadata, runtime), true);
  assert.equal(compactTraceMatchesSimulation(dataset.traces.trace, dataset.metadata, { ...runtime, warmupFraction: 0.25 }), false);
  assert.equal(compactTraceMatchesSimulation(dataset.traces.trace, dataset.metadata, { ...runtime, totalBlocks: 3 }), false);
  assert.equal(
    compactTraceMatchesSimulation(dataset.traces.trace, { ...dataset.metadata, simulation_semantics: "old" }, runtime),
    false,
  );
});
