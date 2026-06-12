import assert from "node:assert/strict";
import fs from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";
import { Worker } from "node:worker_threads";

const require = createRequire(import.meta.url);
const {
  BYTES_PER_GIB,
  DEFAULT_CAPACITY_GIB_VALUES,
  DEFAULT_WARMUP_FRACTION,
  cacheBlocksForGiB,
  createCacheKey,
  estimateBytesPerToken,
  generateTrace,
  inspectUploadedTraceHeadText,
  modelSweepKey,
  parseUploadedTrace,
  parseUploadedTraceStreaming,
  precomputedResultFor,
  runLabComputation,
  simulatePolicy,
  shouldApplyJobResult,
  sweepCapacity,
  throughputFromHitRate,
} = require("../assets/js/kv-cache-lab.js");

const unitTrace = {
  blockSize: 1,
  requests: ["A", "B", "A", "C", "A", "B"].map((id) => ({
    inputBlocks: [{ id, tokens: 1 }],
    appendBlocks: [],
  })),
};

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

const tinyDsaModel = {
  id: "tiny-dsa",
  label: "Tiny DSA",
  formula: "dsa_mla",
  default_tokens: 16,
  fields: {
    num_hidden_layers: 1,
    kv_lora_rank: 2,
    qk_rope_head_dim: 1,
    index_head_dim: 4,
    num_nextn_predict_layers: 1,
  },
};

const tinySlidingModel = {
  id: "tiny-sliding",
  label: "Tiny Sliding",
  formula: "mixed_full_sliding_gqa",
  default_tokens: 4,
  fields: {
    num_hidden_layers: 2,
    full_attention_layers: 1,
    sliding_attention_layers: 1,
    num_key_value_heads: 1,
    head_dim: 1,
    sliding_window: 4,
  },
};

const tinyPreset = {
  id: "chat",
  label: "Chat",
  defaults: {
    sessions: 4,
    average_turns: 3,
    shared_prefix_tokens: 16,
    document_tokens: 8,
    per_turn_input_tokens: 10,
    output_tokens: 12,
    reuse_skew: 0.8,
    burstiness: 0.5,
  },
};

test("lab warmup defaults match precomputed metadata", () => {
  const presetsYaml = fs.readFileSync(new URL("../data/kv_cache_lab/presets.yaml", import.meta.url), "utf8");
  const precomputed = JSON.parse(fs.readFileSync(new URL("../data/kv_cache_lab/precomputed.json", import.meta.url), "utf8"));
  const match = presetsYaml.match(/warmup_fraction:\s*([0-9.]+)/);

  assert.equal(DEFAULT_WARMUP_FRACTION, 0.5);
  assert.equal(Number(match && match[1]), 0.5);
  assert.equal(precomputed.metadata.warmup_fraction, 0.5);
});

test("FIFO, LRU, and optimal policies produce known block hit rates", () => {
  const fifo = simulatePolicy(unitTrace, 2, "fifo", { warmupRequests: 0 });
  const lru = simulatePolicy(unitTrace, 2, "lru", { warmupRequests: 0 });
  const optimal = simulatePolicy(unitTrace, 2, "optimal", { warmupRequests: 0 });

  assert.equal(fifo.hitTokens, 1);
  assert.equal(lru.hitTokens, 2);
  assert.equal(optimal.hitTokens, 3);
  assert.equal(fifo.totalTokens, 6);
  assert.equal(lru.totalTokens, 6);
  assert.equal(optimal.totalTokens, 6);
});

test("uploaded traces use precompute-style contiguous prefix hit semantics by default", () => {
  const jsonl = [
    JSON.stringify({ block_size: 1, input_length: 2, hash_ids: ["A", "B"] }),
    JSON.stringify({ block_size: 1, input_length: 3, hash_ids: ["A", "C", "B"] }),
  ].join("\n");
  const prefixTrace = parseUploadedTrace(jsonl, { blockSize: 0 });
  const blockTrace = parseUploadedTrace(jsonl, { blockSize: 0, cacheSemantics: "block" });

  const prefix = simulatePolicy(prefixTrace, 2, "lru", { warmupRequests: 1 });
  const block = simulatePolicy(blockTrace, 3, "lru", { warmupRequests: 1 });

  assert.equal(prefixTrace.cacheSemantics, "prefix");
  assert.equal(prefix.hitTokens, 1);
  assert.equal(prefix.totalTokens, 3);
  assert.equal(prefix.hitRate, 1 / 3);
  assert.equal(block.hitTokens, 2);
  assert.equal(block.totalTokens, 3);
  assert.equal(block.hitRate, 2 / 3);
});

test("warmup requests affect cache state but are excluded from hit-rate stats", () => {
  const trace = {
    blockSize: 1,
    requests: ["A", "A", "A"].map((id) => ({
      inputBlocks: [{ id, tokens: 1 }],
      appendBlocks: [],
    })),
  };

  const result = simulatePolicy(trace, 1, "lru", { warmupRequests: 1 });

  assert.equal(result.warmupRequests, 1);
  assert.equal(result.hitTokens, 2);
  assert.equal(result.totalTokens, 2);
  assert.equal(result.hitRate, 1);
});

test("theoretical prefill throughput is derived from miss fraction and clamped", () => {
  assert.equal(throughputFromHitRate(0), 1);
  assert.equal(Math.round(throughputFromHitRate(0.9)), 10);
  assert.equal(Math.round(throughputFromHitRate(0.95)), 20);
  assert.equal(throughputFromHitRate(1), 1000);
});

test("useful cache occupancy samples future-useful cached blocks after measured requests", () => {
  const trace = {
    blockSize: 1,
    requests: ["A", "B", "A"].map((id) => ({
      inputBlocks: [{ id, tokens: 1 }],
      appendBlocks: [],
    })),
  };

  const result = simulatePolicy(trace, 2, "lru", { warmupRequests: 0 });

  assert.equal(result.hitTokens, 1);
  assert.equal(result.usefulCacheBlockSamples, 2);
  assert.equal(result.usefulCacheSamples, 3);
  assert.equal(result.usefulCacheRate, 1 / 3);
});

test("useful cache occupancy excludes warmup samples but warmup can create useful cache state", () => {
  const trace = {
    blockSize: 1,
    requests: ["A", "A", "A"].map((id) => ({
      inputBlocks: [{ id, tokens: 1 }],
      appendBlocks: [],
    })),
  };

  const result = simulatePolicy(trace, 1, "lru", { warmupRequests: 1 });

  assert.equal(result.hitRate, 1);
  assert.equal(result.usefulCacheBlockSamples, 1);
  assert.equal(result.usefulCacheSamples, 2);
  assert.equal(result.usefulCacheRate, 0.5);
});

test("generated output blocks occupy cache for later prefill hits", () => {
  const trace = {
    blockSize: 1,
    requests: [
      {
        inputBlocks: [{ id: "A", tokens: 1 }],
        appendBlocks: [{ id: "B", tokens: 1 }],
      },
      {
        inputBlocks: [{ id: "B", tokens: 1 }],
        appendBlocks: [],
      },
    ],
  };

  const result = simulatePolicy(trace, 1, "lru", { warmupRequests: 0 });

  assert.equal(result.hitTokens, 1);
  assert.equal(result.totalTokens, 2);
});

test("deterministic trace generation is stable for the same seed", () => {
  const params = { requests: 12, blockSize: 8 };

  const first = generateTrace(tinyPreset, params, 1234);
  const second = generateTrace(tinyPreset, params, 1234);

  assert.deepEqual(first, second);
  assert.equal(first.requests.length, 12);
  assert.ok(first.summary.totalInputTokens > 0);
});

test("capacity accounting converts model bytes per token into cache blocks", () => {
  const bytesPerToken = estimateBytesPerToken(tinyModel, { precision: "bf16_fp16" });
  const bytesPerBlock = bytesPerToken * 64;

  assert.equal(bytesPerToken, 4);
  assert.equal(bytesPerBlock, 256);
  assert.equal(cacheBlocksForGiB(512 / BYTES_PER_GIB, bytesPerBlock), 2);
});

test("model byte estimate honors indexer precision and draft KV settings", () => {
  const fp4Indexer = estimateBytesPerToken(tinyDsaModel, {
    precision: "bf16_fp16",
    indexerPrecision: "fp4_int4",
    includeDraftKvCache: false,
  });
  const bf16Indexer = estimateBytesPerToken(tinyDsaModel, {
    precision: "bf16_fp16",
    indexerPrecision: "bf16_fp16",
    includeDraftKvCache: false,
  });
  const withDraft = estimateBytesPerToken(tinyDsaModel, {
    precision: "bf16_fp16",
    indexerPrecision: "fp4_int4",
    includeDraftKvCache: true,
  });

  assert.equal(fp4Indexer, 8);
  assert.equal(bf16Indexer, 14);
  assert.equal(withDraft, 16);
});

test("precomputed lookup uses model and precision key exactly", () => {
  const preset = { id: "real_trace", label: "Real Trace" };
  const precomputed = {
    traces: {
      real_trace: {
        nativeBlockSize: 64,
        summary: { requests: 2, averageInputTokens: 64, uniqueBlocks: 1, infiniteHitRate: 0.5 },
        modelSweeps: {
          "tiny-standard|precision=bf16_fp16|indexer=|draft=0": {
            blockSize: 64,
            points: [],
            policies: ["lru"],
            reuseCeiling: 0.5,
          },
        },
      },
    },
  };

  assert.equal(
    modelSweepKey(tinyModel, { precision: "bf16_fp16", includeDraftKvCache: false }),
    "tiny-standard|precision=bf16_fp16|indexer=|draft=0",
  );
  assert.ok(precomputedResultFor(precomputed, preset, tinyModel, { precision: "bf16_fp16", includeDraftKvCache: false }));
  assert.equal(
    precomputedResultFor(precomputed, preset, tinyModel, { precision: "bf16_fp16", indexerPrecision: "fp4_int4", includeDraftKvCache: false }),
    null,
  );
});

test("token-dependent cache layouts use estimate tokens for capacity accounting", () => {
  const shortContext = estimateBytesPerToken(tinySlidingModel, { precision: "bf16_fp16", estimateTokens: 2 });
  const longContext = estimateBytesPerToken(tinySlidingModel, { precision: "bf16_fp16", estimateTokens: 8 });

  assert.equal(shortContext, 8);
  assert.equal(longContext, 6);
});

test("capacity sweep derives token-dependent accounting from the trace", () => {
  const trace = {
    blockSize: 1,
    summary: { averageInputTokens: 8 },
    requests: ["A", "A"].map((id) => ({
      inputBlocks: [{ id, tokens: 1 }],
      appendBlocks: [],
    })),
  };

  const sweep = sweepCapacity(trace, tinySlidingModel, {
    precision: "bf16_fp16",
    blockSize: 1,
    minGiB: 6 / BYTES_PER_GIB,
    maxGiB: 6 / BYTES_PER_GIB,
    steps: 2,
    warmupFraction: 0,
  });

  assert.equal(sweep.bytesPerToken, 6);
  assert.equal(sweep.points[0].cacheBlocks, 1);
});

test("capacity sweep truncates budgets above the unique working set", () => {
  const trace = {
    blockSize: 1,
    summary: { averageInputTokens: 1, uniqueBlocks: 1 },
    requests: ["A", "A"].map((id) => ({
      inputBlocks: [{ id, tokens: 1 }],
      appendBlocks: [],
    })),
  };

  const sweep = sweepCapacity(trace, tinyModel, {
    precision: "bf16_fp16",
    blockSize: 1,
    minGiB: 16 / BYTES_PER_GIB,
    maxGiB: 16 / BYTES_PER_GIB,
    steps: 2,
    warmupFraction: 0,
    computeCeiling: true,
  });

  assert.equal(sweep.points.length, 0);
  assert.equal(sweep.reuseCeiling, 0.5);
});

test("capacity sweep returns one result set per GiB point and policy", () => {
  const trace = {
    blockSize: 1,
    requests: ["A", "B", "A", "C"].map((id) => ({
      inputBlocks: [{ id, tokens: 1 }],
      appendBlocks: [],
    })),
  };

  const sweep = sweepCapacity(trace, tinyModel, {
    precision: "bf16_fp16",
    blockSize: 1,
    minGiB: 4 / BYTES_PER_GIB,
    maxGiB: 8 / BYTES_PER_GIB,
    steps: 2,
    warmupFraction: 0,
  });

  assert.equal(sweep.points.length, 2);
  assert.equal(sweep.points[0].cacheBlocks, 1);
  assert.equal(sweep.points[1].cacheBlocks, 2);
  assert.ok(sweep.points[1].results.lru.hitRate >= sweep.points[0].results.lru.hitRate);
  assert.ok(sweep.points[1].results.optimal.hitRate >= sweep.points[1].results.lru.hitRate);
});

test("capacity sweep defaults to fixed GiB points", () => {
  const hugeBlockModel = {
    ...tinyModel,
    fields: {
      ...tinyModel.fields,
      head_dim: 4294967296,
    },
  };
  const trace = {
    blockSize: 1,
    requests: Array.from({ length: 1100 }, (_, index) => ({
      inputBlocks: [{ id: `B${index}`, tokens: 1 }],
      appendBlocks: [],
    })),
  };
  trace.requests.push({
    inputBlocks: [{ id: "B0", tokens: 1 }],
    appendBlocks: [],
  });

  const sweep = sweepCapacity(trace, hugeBlockModel, {
    precision: "bf16_fp16",
    blockSize: 1,
    warmupFraction: 0,
    computeCeiling: true,
  });

  assert.deepEqual(
    sweep.points.map((point) => point.gib),
    DEFAULT_CAPACITY_GIB_VALUES,
  );
});

test("capacity sweep omits no-pressure ceiling policy points", () => {
  const trace = {
    blockSize: 1,
    requests: ["A", "A"].map((id) => ({
      inputBlocks: [{ id, tokens: 1 }],
      appendBlocks: [],
    })),
  };

  const sweep = sweepCapacity(trace, tinyModel, {
    precision: "bf16_fp16",
    blockSize: 1,
    warmupFraction: 0,
    computeCeiling: true,
  });

  assert.equal(sweep.points.length, 0);
  assert.equal(sweep.reuseCeiling, 0.5);
});

test("cache keys are stable for identical settings and change for capacity inputs", () => {
  const first = {
    preset: tinyPreset,
    model: tinyModel,
    params: { requests: 12, blockSize: 8 },
    settings: { precision: "bf16_fp16", blockSize: 8, minGiB: 1, maxGiB: 2, steps: 2 },
    seed: 1234,
  };
  const second = {
    seed: 1234,
    settings: { steps: 2, maxGiB: 2, minGiB: 1, blockSize: 8, precision: "bf16_fp16" },
    params: { blockSize: 8, requests: 12 },
    model: tinyModel,
    preset: tinyPreset,
  };
  const changed = {
    preset: tinyPreset,
    model: tinyModel,
    params: { requests: 12, blockSize: 16 },
    settings: { precision: "bf16_fp16", blockSize: 16, minGiB: 1, maxGiB: 2, steps: 2 },
    seed: 1234,
  };

  assert.equal(createCacheKey(first), createCacheKey(second));
  assert.notEqual(createCacheKey(first), createCacheKey(changed));
});

test("stale worker jobs do not pass the latest-job guard", () => {
  assert.equal(shouldApplyJobResult(2, { jobId: 1 }), false);
  assert.equal(shouldApplyJobResult(2, { jobId: 2 }), true);
  assert.equal(shouldApplyJobResult(2, null), false);
});

test("uploaded trace parser uses provided blockSize when records omit block_size", () => {
  const jsonl = [
    JSON.stringify({ timestamp: 1, hash_ids: [1, 2], input_length: 128 }),
    JSON.stringify({ timestamp: 2, hash_ids: [1, 3], input_length: 128 }),
  ].join("\n");

  const trace = parseUploadedTrace(jsonl, { label: "custom", blockSize: 64 });

  assert.equal(trace.blockSize, 64);
  assert.equal(trace.summary.requests, 2);
  assert.deepEqual(Array.from(trace.__flat.eventTokens), [64, 64, 64, 64]);
});

test("uploaded trace parser requires declared input_length", () => {
  const jsonl = [
    JSON.stringify({ timestamp: 1, block_size: 64, hash_ids: [1, 2], input_length: 128 }),
    JSON.stringify({ timestamp: 2, block_size: 64, hash_ids: [1, 3] }),
  ].join("\n");

  assert.throws(
    () => parseUploadedTrace(jsonl, { label: "missing input" }),
    /input_length/,
  );
});

test("uploaded trace streaming parser requires declared input_length", async () => {
  const jsonl = [
    JSON.stringify({ timestamp: 1, block_size: 64, hash_ids: [1, 2], input_length: 128 }),
    JSON.stringify({ timestamp: 2, block_size: 64, hash_ids: [1, 3] }),
  ].join("\n");
  async function* chunks() {
    yield jsonl;
  }

  await assert.rejects(
    () => parseUploadedTraceStreaming(chunks(), { label: "missing input" }),
    /input_length/,
  );
});

test("uploaded trace parser rejects inconsistent block_size", () => {
  const jsonl = [
    JSON.stringify({ timestamp: 1, block_size: 64, hash_ids: [1, 2], input_length: 128 }),
    JSON.stringify({ timestamp: 2, block_size: 32, hash_ids: [1, 3], input_length: 96 }),
  ].join("\n");

  assert.throws(
    () => parseUploadedTrace(jsonl, { label: "mixed" }),
    /block_size must be consistent/,
  );
});

test("uploaded trace parser keeps bad lines but accepts valid declared-block records", () => {
  const jsonl = [
    "not json",
    JSON.stringify({ timestamp: 1, block_size: 64, hash_ids: ["9007199254740993", 2], input_length: 96 }),
    JSON.stringify({ timestamp: 2, block_size: 64, hash_ids: ["9007199254740993", 3], input_length: 96 }),
  ].join("\n");

  const trace = parseUploadedTrace(jsonl, { label: "valid" });

  assert.equal(trace.blockSize, 64);
  assert.equal(trace.summary.requests, 2);
  assert.equal(trace.summary.parseErrors, 1);
  assert.equal(trace.summary.averageInputTokens, 96);
});

test("uploaded trace parser preserves string hash ids beyond JS safe integers", () => {
  const jsonl = [
    '{"timestamp":1,"block_size":64,"hash_ids":["9007199254740993"],"input_length":64}',
    '{"timestamp":2,"block_size":64,"hash_ids":["9007199254740994"],"input_length":64}',
    '{"timestamp":3,"block_size":64,"hash_ids":["9007199254740993"],"input_length":64}',
  ].join("\n");

  const trace = parseUploadedTrace(jsonl, { label: "large string ids" });

  assert.equal(trace.summary.requests, 3);
  assert.equal(trace.summary.uniqueBlocks, 2);
  assert.deepEqual(Array.from(trace.__flat.eventIds), [0, 1, 0]);
});

test("uploaded trace parser preserves unsafe JSON number hash ids from raw lines", () => {
  const jsonl = [
    '{"timestamp":1,"block_size":64,"hash_ids":[9007199254740993],"input_length":64}',
    '{"timestamp":2,"block_size":64,"hash_ids":[9007199254740994],"input_length":64}',
    '{"timestamp":3,"block_size":64,"hash_ids":[9007199254740993],"input_length":64}',
  ].join("\n");

  const trace = parseUploadedTrace(jsonl, { label: "unsafe number id" });

  assert.equal(trace.summary.requests, 3);
  assert.equal(trace.summary.uniqueBlocks, 2);
  assert.deepEqual(Array.from(trace.__flat.eventIds), [0, 1, 0]);
});

test("uploaded trace head inspection reports schema problems early", () => {
  assert.deepEqual(
    inspectUploadedTraceHeadText(JSON.stringify({ timestamp: 1, block_size: 64, hash_ids: [1], input_length: 64 })),
    { valid: true, blockSize: 64, validRecords: 1, parseErrors: 0 },
  );

  assert.match(
    inspectUploadedTraceHeadText(JSON.stringify({ timestamp: 1, block_size: 64, input_length: 64 })).error,
    /hash_ids/,
  );
  assert.match(
    inspectUploadedTraceHeadText(JSON.stringify({ timestamp: 1, block_size: 64, hash_ids: [1] })).error,
    /input_length/,
  );
  assert.deepEqual(
    inspectUploadedTraceHeadText(JSON.stringify({ timestamp: 1, hash_ids: [1], input_length: 64 })),
    { valid: true, blockSize: 0, validRecords: 1, parseErrors: 0 },
  );
});

test("uploaded trace head inspection rejects schema errors even with valid records", () => {
  const inspected = inspectUploadedTraceHeadText([
    JSON.stringify({ timestamp: 1, block_size: 64, hash_ids: [1], input_length: 64 }),
    JSON.stringify({ timestamp: 2, block_size: 64, hash_ids: [2] }),
  ].join("\n"));

  assert.equal(inspected.valid, false);
  assert.match(inspected.error, /input_length/);
});

test("uploaded trace head inspection tolerates bad lines before a valid record", () => {
  const inspected = inspectUploadedTraceHeadText([
    "not json",
    JSON.stringify({ timestamp: 1, block_size: 32, hash_ids: [1, 2], input_length: 64 }),
  ].join("\n"));

  assert.equal(inspected.valid, true);
  assert.equal(inspected.blockSize, 32);
  assert.equal(inspected.parseErrors, 1);
});

test("worker job returns the same sweep as direct computation", async () => {
  const baseInput = {
    preset: tinyPreset,
    model: tinyModel,
    params: { requests: 12, blockSize: 8 },
    settings: {
      precision: "bf16_fp16",
      blockSize: 8,
      minGiB: 4 / BYTES_PER_GIB,
      maxGiB: 8 / BYTES_PER_GIB,
      steps: 2,
      warmupFraction: 0,
    },
    seed: 1234,
  };
  const input = Object.assign({ type: "run", jobId: 7, cacheKey: createCacheKey(baseInput) }, baseInput);
  const direct = runLabComputation(input);
  const workerResult = await new Promise((resolve, reject) => {
    const worker = new Worker(new URL("../assets/js/kv-cache-lab-worker.js", import.meta.url));
    worker.on("message", (message) => {
      if (message && message.type === "progress") return;
      worker.terminate();
      if (message.error) reject(new Error(message.error));
      else resolve(message.result || message);
    });
    worker.on("error", reject);
    worker.postMessage(input);
  });

  assert.deepEqual(workerResult.sweep, direct.sweep);
  assert.deepEqual(workerResult.trace.summary, direct.trace.summary);
  assert.equal(workerResult.jobId, 7);
});
