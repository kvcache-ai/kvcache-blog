import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import zlib from "node:zlib";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const lab = require("../assets/js/kv-cache-lab.js");

const here = path.dirname(fileURLToPath(import.meta.url));
const WASM_PATH = path.resolve(here, "../assets/wasm/kvcache-sim.wasm");
const WORKER_URL = new URL("../assets/js/kv-cache-lab-worker.js", import.meta.url);
const hasWasm = fs.existsSync(WASM_PATH);

const model = {
  id: "m",
  formula: "standard_gqa",
  default_tokens: 4096,
  fields: { num_hidden_layers: 32, num_key_value_heads: 8, head_dim: 128 },
};
const settings = { precision: "bf16_fp16", warmupFraction: 0.5, computeCeiling: true };
const POLICY = { fifo: 0, lru: 1, optimal: 2 };

function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
// Small ids (< 2^53) so the wasm's exact-u64 interning groups identically to the
// JS String(JSON-parsed-number) interning; bad/empty lines exercise the counters.
function makeJsonl(seed, n, pool, blockSize, skew) {
  const rng = lcg(seed);
  const lines = [];
  let ts = 0;
  for (let i = 0; i < n; i += 1) {
    if (i % 400 === 3) {
      lines.push("not json {");
      continue;
    }
    if (i % 400 === 11) {
      lines.push(JSON.stringify({ input_length: 5 }));
      continue;
    }
    const count = 2 + Math.floor(rng() * 30);
    const ids = [];
    for (let b = 0; b < count; b += 1) ids.push(Math.floor(pool * Math.pow(rng(), skew)));
    ts += Math.floor(rng() * 15);
    lines.push(JSON.stringify({ hash_ids: ids, input_length: (count - 1) * blockSize + (1 + Math.floor(rng() * blockSize)), block_size: blockSize, timestamp: ts }));
  }
  return lines.join("\n") + "\n";
}

function driveWasm(ex, text, { blockSize = 0, maxEvents = 0, warmupFraction = lab.DEFAULT_WARMUP_FRACTION }) {
  const bytes = new TextEncoder().encode(text);
  ex.reset(blockSize, maxEvents, warmupFraction);
  const CHUNK = 1 << 16;
  for (let off = 0; off < bytes.length; off += CHUNK) {
    const slice = bytes.subarray(off, Math.min(off + CHUNK, bytes.length));
    const ptr = ex.chunk_ptr(slice.length);
    new Uint8Array(ex.memory.buffer, ptr, slice.length).set(slice);
    if (ex.ingest(slice.length)) break;
  }
  ex.finalize();
  return ex;
}

test("wasm sim matches JS on summary + every policy/capacity hit count", { skip: hasWasm ? false : "assets/wasm/kvcache-sim.wasm not built" }, async () => {
  const { instance } = await WebAssembly.instantiate(fs.readFileSync(WASM_PATH), {});
  const ex = instance.exports;
  for (const [seed, n, pool, bs, skew, wf] of [
    [1, 4000, 8000, 512, 2.0, 0.3],
    [2, 3000, 500, 64, 1.3, 0.3],
    [3, 5000, 200000, 16, 2.6, 0],
    [4, 2000, 40, 512, 1.0, 0.5],
  ]) {
    const jsonl = makeJsonl(seed, n, pool, bs, skew);
    driveWasm(ex, jsonl, { warmupFraction: wf });
    const trace = lab.parseUploadedTrace(jsonl, { blockSize: 0 });
    const plan = lab.buildExecutionPlan(trace, { warmupFraction: wf });

    assert.equal(ex.requests(), trace.summary.requests, "requests");
    assert.equal(ex.unique_blocks(), trace.summary.uniqueBlocks, "unique");
    assert.equal(ex.total_input_tokens(), trace.summary.totalInputTokens, "totalInputTokens");
    assert.equal(ex.total_measured(), plan.totalMeasuredTokens, "totalMeasured");
    assert.equal(ex.warmup_requests(), plan.warmupRequests, "warmupRequests");
    assert.equal(ex.ceiling_hit(), lab.simulatePlanPolicy(plan, Math.max(ex.unique_blocks(), 1), "lru").hitTokens, "ceiling");

    const unique = ex.unique_blocks();
    const caps = [1, 2, 4, 16, 64, 256, 1024, 4096, Math.max(1, Math.floor(unique / 2)), unique, unique + 5];
    for (const cap of caps) {
      for (const policy of ["fifo", "lru", "optimal"]) {
        assert.equal(ex.sweep(cap, POLICY[policy]), lab.simulatePlanPolicy(plan, cap, policy).hitTokens, `sweep cap=${cap} ${policy}`);
      }
    }
  }
});

test("wasm upload parser preserves u64 hash id identity beyond JS safe integers", { skip: hasWasm ? false : "assets/wasm/kvcache-sim.wasm not built" }, async () => {
  const { instance } = await WebAssembly.instantiate(fs.readFileSync(WASM_PATH), {});
  const jsonl = [
    '{"timestamp":1,"block_size":64,"hash_ids":[9007199254740992],"input_length":64}',
    '{"timestamp":2,"block_size":64,"hash_ids":[9007199254740993],"input_length":64}',
    '{"timestamp":3,"block_size":64,"hash_ids":[9007199254740992],"input_length":64}',
  ].join("\n");

  driveWasm(instance.exports, jsonl, { warmupFraction: 0 });

  assert.equal(instance.exports.requests(), 3);
  assert.equal(instance.exports.unique_blocks(), 2);
  assert.equal(instance.exports.ceiling_hit(), 64);
});

test("wasm upload parser reports missing block_size, inconsistent block_size, and missing input_length", { skip: hasWasm ? false : "assets/wasm/kvcache-sim.wasm not built" }, async () => {
  const { instance } = await WebAssembly.instantiate(fs.readFileSync(WASM_PATH), {});
  driveWasm(instance.exports, [
    '{"timestamp":1,"hash_ids":[1],"input_length":64}',
    '{"timestamp":2,"block_size":64,"hash_ids":[1],"input_length":64}',
    '{"timestamp":3,"block_size":32,"hash_ids":[2],"input_length":32}',
    '{"timestamp":4,"block_size":64,"hash_ids":[3]}',
  ].join("\n"), { warmupFraction: 0 });

  assert.equal(instance.exports.missing_block_size(), 1);
  assert.equal(instance.exports.inconsistent_block_size(), 1);
  assert.equal(instance.exports.missing_input_length(), 1);
  assert.equal(instance.exports.requests(), 1);
});

test("wasm engine path (worker) equals sweepCapacity for plain + gzip uploads", { skip: hasWasm ? false : "assets/wasm/kvcache-sim.wasm not built" }, async () => {
  const workers = [];
  const createWorker = () => {
    const w = new Worker(WORKER_URL);
    workers.push(w);
    const adapter = { onmessage: null, onerror: null, postMessage: (m) => w.postMessage(m) };
    w.on("message", (data) => adapter.onmessage && adapter.onmessage({ data }));
    w.on("error", (err) => adapter.onerror && adapter.onerror(err));
    return adapter;
  };
  const engine = lab.createLabEngine({ createWorker, poolSize: 1 });
  const jsonl = makeJsonl(42, 4000, 8000, 512, 2.0);
  const jsTrace = lab.parseUploadedTrace(jsonl, { blockSize: 0 });
  const jsSweep = lab.sweepCapacity(jsTrace, model, settings);
  const jsTimeStats = lab.computeTimeSeries(jsTrace);

  try {
    for (const gzip of [false, true]) {
      const bytes = gzip ? zlib.gzipSync(Buffer.from(jsonl, "utf8")) : Buffer.from(jsonl, "utf8");
      const file = new File([bytes], gzip ? "t.jsonl.gz" : "t.jsonl", { lastModified: 1 });
      const result = await engine.run(
        {
          jobId: 1,
          cacheKey: `k${gzip}`,
          preset: { id: lab.UPLOAD_PRESET_ID, label: "u" },
          model,
          settings,
          uploadFile: file,
          gzip,
          useWasm: true,
          wasmUrl: WASM_PATH,
          uploadOptions: { blockSize: 0, label: file.name, maxEvents: 40000000 },
        },
        () => {},
      ).promise;
      assert.deepEqual(result.sweep, jsSweep);
      assert.deepEqual(result.timeStats, jsTimeStats, "temporal statistics match the JS path");
      assert.ok(result.trace.summary.timeSpanSeconds > 0, "time span surfaced");
      assert.ok(!result.trace.summary.capped, "full load (not capped)");
    }
  } finally {
    await Promise.all(workers.map((w) => w.terminate()));
  }
});
