import assert from "node:assert/strict";
import test from "node:test";
import zlib from "node:zlib";
import { Worker } from "node:worker_threads";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const lab = require("../assets/js/kv-cache-lab.js");

const BS = 512;

function makePool() {
  const workers = [];
  const counts = { analyze: 0, simulate: 0 };
  const createWorker = () => {
    const w = new Worker(new URL("../assets/js/kv-cache-lab-worker.js", import.meta.url));
    workers.push(w);
    const adapter = {
      onmessage: null,
      onerror: null,
      postMessage(message) {
        if (message && counts[message.type] !== undefined) counts[message.type] += 1;
        w.postMessage(message);
      },
    };
    w.on("message", (data) => adapter.onmessage && adapter.onmessage({ data }));
    w.on("error", (err) => adapter.onerror && adapter.onerror(err));
    return adapter;
  };
  let engine = null;
  function createEngineRun(input, onProgress) {
    if (!engine) engine = lab.createLabEngine({ createWorker, poolSize: 3 });
    return engine.run(input, onProgress).promise;
  }
  return {
    workers,
    counts,
    createWorker,
    createEngineRun,
    terminate: () => Promise.all(workers.map((w) => w.terminate())),
  };
}

function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function makeJsonl(seed, n, pool, skew) {
  const rng = lcg(seed);
  const lines = [];
  let ts = 0;
  for (let i = 0; i < n; i += 1) {
    const count = 2 + Math.floor(rng() * 30);
    const ids = [];
    for (let b = 0; b < count; b += 1) ids.push(Math.floor(pool * Math.pow(rng(), skew)));
    const inputLength = (count - 1) * BS + (1 + Math.floor(rng() * BS));
    ts += Math.floor(rng() * 25);
    lines.push(JSON.stringify({ hash_ids: ids, input_length: inputLength, output_length: 0, block_size: BS, timestamp: ts }));
  }
  return lines.join("\n");
}

const JSONL = makeJsonl(20260605, 1500, 5000, 1.8);

const modelA = {
  id: "model-a",
  formula: "standard_gqa",
  default_tokens: 4096,
  fields: { num_hidden_layers: 32, num_key_value_heads: 8, head_dim: 128 },
};
// Different per-token byte cost -> different GiB->cacheBlocks mapping.
const modelB = {
  id: "model-b",
  formula: "standard_gqa",
  default_tokens: 4096,
  fields: { num_hidden_layers: 48, num_key_value_heads: 4, head_dim: 128 },
};
const settings = { precision: "bf16_fp16", warmupFraction: 0.3, computeCeiling: true };

test("decomposition (analyze + planTasks + planFromBuffers + assemble) equals sweepCapacity", () => {
  const trace = lab.parseUploadedTrace(JSONL, { blockSize: BS });
  const analysis = lab.analyzeTrace(trace, { warmupFraction: settings.warmupFraction });
  // Simulate from the round-tripped buffers, exactly like a pool worker would.
  const rebuiltPlan = lab.planFromBuffers(analysis.planBuffers);

  [modelA, modelB].forEach((model) => {
    const planned = lab.planSweepTasks(analysis.meta, model, settings);
    const memo = new Map();
    planned.tasks.forEach((task) => {
      memo.set(`${task.policy}|${task.cacheBlocks}`, lab.simulatePlanPolicy(rebuiltPlan, task.cacheBlocks, task.policy));
    });
    const sweep = lab.assembleSweep(
      planned,
      analysis.meta,
      settings,
      (policy, cacheBlocks) => memo.get(`${policy}|${cacheBlocks}`),
      analysis.ceiling,
    );
    assert.deepEqual(sweep, lab.sweepCapacity(trace, model, settings));
  });
});

test("pool engine matches sweepCapacity and caches analysis + sims across models", async () => {
  const workers = [];
  const counts = { analyze: 0, simulate: 0 };
  const createWorker = () => {
    const w = new Worker(new URL("../assets/js/kv-cache-lab-worker.js", import.meta.url));
    workers.push(w);
    const adapter = {
      onmessage: null,
      onerror: null,
      postMessage(message) {
        if (message && counts[message.type] !== undefined) counts[message.type] += 1;
        w.postMessage(message);
      },
    };
    w.on("message", (data) => {
      if (adapter.onmessage) adapter.onmessage({ data });
    });
    w.on("error", (err) => {
      if (adapter.onerror) adapter.onerror(err);
    });
    return adapter;
  };

  const engine = lab.createLabEngine({ createWorker, poolSize: 4 });
  const baseInput = {
    preset: { id: lab.UPLOAD_PRESET_ID, label: "u" },
    settings,
    uploadText: JSONL,
    uploadOptions: { blockSize: BS, label: "u" },
  };
  const trace = lab.parseUploadedTrace(JSONL, { blockSize: BS });
  const directA = lab.sweepCapacity(trace, modelA, settings);
  const directB = lab.sweepCapacity(trace, modelB, settings);

  try {
    const r1 = await engine.run(Object.assign({ jobId: 1, cacheKey: "a", model: modelA }, baseInput)).promise;
    assert.deepEqual(r1.sweep, directA);
    assert.equal(r1.jobId, 1);
    assert.ok(r1.timeStats && r1.timeStats.timeBuckets.length === 48);
    assert.equal(r1.trace.summary.requests, trace.summary.requests);
    const analyzeAfter1 = counts.analyze;
    const simulateAfter1 = counts.simulate;
    assert.equal(analyzeAfter1, 1, "exactly one analyze for the first sweep");
    assert.ok(simulateAfter1 > 0, "the first sweep dispatches sims");

    // Second model, same trace: must reuse the cached analysis (no re-parse).
    const r2 = await engine.run(Object.assign({ jobId: 2, cacheKey: "b", model: modelB }, baseInput)).promise;
    assert.deepEqual(r2.sweep, directB);
    assert.equal(counts.analyze, 1, "no re-analyze for a new model on the same trace");
    assert.ok(counts.simulate > simulateAfter1, "the new model needs sims at its own cacheBlocks");
    const simulateAfter2 = counts.simulate;

    // Back to the first model: fully cached -> zero new analyze, zero new sims.
    const r3 = await engine.run(Object.assign({ jobId: 3, cacheKey: "a", model: modelA }, baseInput)).promise;
    assert.deepEqual(r3.sweep, directA);
    assert.equal(counts.analyze, 1, "still no re-analyze");
    assert.equal(counts.simulate, simulateAfter2, "every sim served from cache the second time around");
  } finally {
    await Promise.all(workers.map((w) => w.terminate()));
  }
});

test("pool engine stream-parses a gzip File and matches sweepCapacity", async () => {
  const gz = zlib.gzipSync(Buffer.from(JSONL + "\n", "utf8"));
  const file = new File([gz], "trace.jsonl.gz", { lastModified: 7 });
  const direct = lab.sweepCapacity(lab.parseUploadedTrace(JSONL + "\n", { blockSize: BS }), modelA, settings);
  const pool = makePool();
  let progressTicks = 0;
  try {
    const r = await pool
      .createEngineRun({
        jobId: 1,
        cacheKey: "g",
        preset: { id: lab.UPLOAD_PRESET_ID, label: "u" },
        model: modelA,
        settings,
        uploadFile: file,
        gzip: true,
        uploadOptions: { blockSize: BS, label: "trace.jsonl.gz" },
      }, () => { progressTicks += 1; });
    assert.deepEqual(r.sweep, direct);
    assert.ok(r.trace.summary.requests > 0);
    assert.ok(r.timeStats && r.timeStats.timeBuckets.length === 48);
    assert.ok(progressTicks > 0, "parse progress forwarded to onProgress");
  } finally {
    await pool.terminate();
  }
});

test("pool engine caps a huge upload and reports it, matching the capped sweep", async () => {
  const gz = zlib.gzipSync(Buffer.from(JSONL + "\n", "utf8"));
  const file = new File([gz], "big.jsonl.gz", { lastModified: 9 });
  const maxEvents = 3000;
  const cappedTrace = lab.parseUploadedTrace(JSONL + "\n", { blockSize: BS, maxEvents });
  assert.equal(cappedTrace.summary.capped, true, "test data is large enough to cap");
  const direct = lab.sweepCapacity(cappedTrace, modelA, settings);
  const pool = makePool();
  try {
    const r = await pool.createEngineRun({
      jobId: 1,
      cacheKey: "c",
      preset: { id: lab.UPLOAD_PRESET_ID, label: "u" },
      model: modelA,
      settings,
      uploadFile: file,
      gzip: true,
      uploadOptions: { blockSize: BS, label: "big.jsonl.gz", maxEvents },
    });
    assert.equal(r.trace.summary.capped, true, "engine surfaces the truncation flag");
    assert.equal(r.trace.summary.requests, cappedTrace.summary.requests);
    assert.deepEqual(r.sweep, direct);
  } finally {
    await pool.terminate();
  }
});
