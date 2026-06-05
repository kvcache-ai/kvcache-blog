import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_CAPACITY_GIB_VALUES,
  allModelSettings,
  buildTrace,
  infiniteCacheReuse,
  modelSweepKey,
  normalizeBailianRecord,
  normalizeExgenticAgentRecord,
  normalizeLmcacheAgenticRecord,
  normalizeMooncakeRecord,
  normalizeRagPulseRecord,
  normalizeWekaSessionRecord,
  precomputeSweep,
} from "../scripts/lib/kv-cache-lab-traces.mjs";

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

test("infinite-cache reuse ceiling uses warmup for state but not stats", () => {
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
  assert.equal(ceiling.hitRate, 0.5);
});

test("precompute sweep is deterministic and uses source-native block size", () => {
  const trace = buildTrace(
    { id: "fixture", label: "Fixture", nativeBlockSize: 16, sourceKind: "hash" },
    [
      { timestamp: 0, inputBlocks: [{ id: "A", tokens: 16 }], appendBlocks: [] },
      { timestamp: 1, inputBlocks: [{ id: "A", tokens: 16 }], appendBlocks: [] },
    ],
  );
  const settings = {
    precision: "bf16_fp16",
    blockSize: 16,
    capacityGiBValues: DEFAULT_CAPACITY_GIB_VALUES.slice(0, 2),
    warmupFraction: 0,
  };

  const first = precomputeSweep(trace, tinyModel, settings, { generatedAt: "fixed" });
  const second = precomputeSweep(trace, tinyModel, settings, { generatedAt: "fixed" });

  assert.deepEqual(first, second);
  assert.equal(first.blockSize, 16);
  assert.equal(first.points.length, 2);
  assert.equal(first.points[0].results.lru.hitRate, 0.5);
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
