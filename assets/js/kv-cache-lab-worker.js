(function (root) {
  "use strict";

  function labModule() {
    if (root.KVCacheLab) return root.KVCacheLab;
    if (typeof require === "function") return require("./kv-cache-lab.js");
    throw new Error("KVCacheLab is not loaded");
  }

  function ensureBrowserScripts(message) {
    if (root.KVCacheLab) return;
    if (typeof importScripts !== "function") return;
    if (!message.calculatorScriptUrl || !message.labScriptUrl) {
      throw new Error("Worker script URLs are missing");
    }
    importScripts(message.calculatorScriptUrl, message.labScriptUrl);
  }

  // ---- WASM trace processor (full-trace loads via wasm/kvcache-sim) ----------
  // One compiled module per worker; one live instance per trace (its linear
  // memory holds the whole plan). A model/precision change on the same trace
  // reuses the instance and just re-runs the sweeps.
  let wasmModulePromise = null;
  let wasmExports = null;
  let wasmTraceKey = null;
  let wasmSimMemo = null;
  let wasmCeilingStats = null;
  let wasmTimeStats = null;
  const POLICY_CODE = { fifo: 0, lru: 1, optimal: 2 };

  // Read the WASM temporal-statistics buffer and hand its accumulators to the
  // shared JS finalizer, so the panel is identical to the JS-path computeTimeSeries.
  function wasmComputeTimeStats(ex, lab) {
    const ptr = ex.compute_time_series();
    if (!ptr) return null;
    const len = ex.time_series_len();
    const buf = new Float64Array(ex.memory.buffer, ptr, len);
    const TB = 48;
    const tMin = buf[0];
    const tMax = buf[1];
    const span = buf[2];
    const dt = buf[3];
    const totalTokens = buf[4];
    const reuseTokens = buf[5];
    const bucketTotal = buf.slice(6, 6 + TB);
    const bucketHit = buf.slice(6 + TB, 6 + 2 * TB);
    const gapTokens = buf.slice(6 + 2 * TB, 6 + 2 * TB + 10);
    const gapCount = buf.slice(6 + 2 * TB + 10, 6 + 2 * TB + 20);
    return lab.buildTimeSeriesResult(tMin, tMax, span, dt, bucketTotal, bucketHit, gapTokens, gapCount, totalTokens, reuseTokens);
  }

  function ensureWasmModule(url) {
    if (!wasmModulePromise) {
      wasmModulePromise = (async () => {
        try {
          return await WebAssembly.compileStreaming(fetch(url));
        } catch (error) {
          // Node worker_threads: read the file from disk; browsers without
          // compileStreaming: plain fetch + compile.
          if (typeof importScripts === "undefined" && typeof require === "function") {
            const fs = require("node:fs");
            return WebAssembly.compile(fs.readFileSync(url.replace(/^file:\/\//, "")));
          }
          const response = await fetch(url);
          return WebAssembly.compile(await response.arrayBuffer());
        }
      })();
    }
    return wasmModulePromise;
  }

  function clearWasm() {
    wasmExports = null;
    wasmTraceKey = null;
    wasmSimMemo = null;
    wasmCeilingStats = null;
    wasmTimeStats = null;
  }

  function readWasmStats(ex, cacheBlocks, policyCode) {
    if (typeof ex.sweep_stats === "function") {
      const ptr = ex.sweep_stats(cacheBlocks, policyCode);
      const stats = new Float64Array(ex.memory.buffer, ptr, 3);
      return {
        hitTokens: stats[0],
        usefulCacheBlockSamples: stats[1],
        usefulCacheSamples: stats[2],
      };
    }
    return {
      hitTokens: ex.sweep(cacheBlocks, policyCode),
      usefulCacheBlockSamples: 0,
      usefulCacheSamples: 0,
    };
  }

  async function streamFileIntoWasm(ex, file, gzip, fileSize, onParseProgress) {
    let raw = file.stream();
    let seen = 0;
    if (onParseProgress && typeof TransformStream === "function") {
      raw = raw.pipeThrough(
        new TransformStream({
          transform(chunk, controller) {
            seen += chunk.byteLength || chunk.length || 0;
            onParseProgress(seen, fileSize || 0);
            controller.enqueue(chunk);
          },
        }),
      );
    }
    const byteStream = gzip ? raw.pipeThrough(new DecompressionStream("gzip")) : raw;
    const reader = byteStream.getReader();
    try {
      for (;;) {
        const next = await reader.read();
        if (next.done) break;
        const bytes = next.value;
        if (!bytes || bytes.length === 0) continue;
        let off = 0;
        while (off < bytes.length) {
          const len = Math.min(bytes.length - off, 1 << 20);
          const ptr = ex.chunk_ptr(len);
          // Re-acquire the memory view: chunk_ptr/ingest may have grown (and thus
          // detached) the buffer.
          new Uint8Array(ex.memory.buffer, ptr, len).set(bytes.subarray(off, off + len));
          const capped = ex.ingest(len);
          off += len;
          if (capped) {
            try {
              await reader.cancel();
            } catch (error) {
              /* already closing */
            }
            return;
          }
        }
      }
    } finally {
      try {
        await reader.cancel();
      } catch (error) {
        /* already closed */
      }
    }
  }

  function buildWasmMeta(ex, lab, label) {
    const requests = ex.requests();
    const unique = ex.unique_blocks();
    const totalInputTokens = ex.total_input_tokens();
    const tMin = ex.t_min();
    const tMax = ex.t_max();
    const summary = {
      requests,
      totalInputTokens,
      averageInputTokens: requests ? totalInputTokens / requests : 0,
      uniqueBlocks: unique,
      parseErrors: ex.parse_errors(),
      skipped: ex.skipped(),
    };
    if (ex.was_capped()) summary.capped = true;
    if (tMax > tMin) {
      summary.tStart = tMin;
      summary.tEnd = tMax;
      summary.timeSpanSeconds = tMax - tMin;
    }
    const blockSize = ex.block_size();
    return {
      requestCount: requests,
      eventCount: ex.events(),
      uniqueBlocks: unique,
      warmupRequests: ex.warmup_requests(),
      totalMeasuredTokens: ex.total_measured(),
      blockSize,
      summary,
      slimTrace: {
        presetId: lab.UPLOAD_PRESET_ID,
        presetLabel: label,
        sourceKind: "hash",
        blockSize,
        summary,
        requestCount: requests,
      },
    };
  }

  async function runWasmJob(message, onProgress) {
    ensureBrowserScripts(message);
    const lab = labModule();
    const mod = await ensureWasmModule(message.wasmUrl);
    if (wasmTraceKey !== message.traceKey || !wasmExports) {
      // (Re)build the instance and stream the file into its linear memory.
      clearWasm();
      const instance = await WebAssembly.instantiate(mod, {});
      wasmExports = instance.exports;
      wasmSimMemo = new Map();
      const blockSize = Math.max(0, Math.floor(Number(message.blockSizeOverride) || 0));
      const maxEvents = Math.max(0, Math.floor(Number(message.maxEvents) || 0));
      wasmExports.reset(blockSize, maxEvents, Number(message.warmupFraction) || 0);
      await streamFileIntoWasm(wasmExports, message.uploadFile, !!message.gzip, message.fileSize, (bytes, total) =>
        onProgress({ phase: "parse", bytes, total }),
      );
      wasmExports.finalize();
      const missingBlockSize =
        typeof wasmExports.missing_block_size === "function" ? wasmExports.missing_block_size() : 0;
      const inconsistentBlockSize =
        typeof wasmExports.inconsistent_block_size === "function" ? wasmExports.inconsistent_block_size() : 0;
      const missingInputLength =
        typeof wasmExports.missing_input_length === "function" ? wasmExports.missing_input_length() : 0;
      if (missingBlockSize > 0) {
        throw new Error(`Uploaded trace records without block_size need a positive Block size value. Found ${missingBlockSize} valid hash record(s) without one.`);
      }
      if (inconsistentBlockSize > 0) {
        throw new Error(`Uploaded trace block_size must be consistent. Found ${inconsistentBlockSize} record(s) with a different block_size.`);
      }
      if (missingInputLength > 0) {
        throw new Error(`Uploaded trace records must include a positive "input_length". Found ${missingInputLength} valid hash record(s) without it.`);
      }
      if (wasmExports.requests() <= 0) {
        throw new Error('No valid uploaded trace records found. Each line must be JSON with a non-empty "hash_ids" array and a positive "input_length". Include "block_size" in the trace or set Block size before running.');
      }
      wasmTimeStats = wasmComputeTimeStats(wasmExports, lab);
      wasmTraceKey = message.traceKey;
    }
    const ex = wasmExports;
    const meta = buildWasmMeta(ex, lab, (message.uploadOptions && message.uploadOptions.label) || "Uploaded trace");
    const planned = lab.planSweepTasks(meta, message.model, message.settings || {});
    const total = planned.tasks.length;
    let done = 0;
    onProgress({ phase: "sweep", completed: 0, total: Math.max(total, 1) });
    for (const task of planned.tasks) {
      const key = `${task.policy}|${task.cacheBlocks}`;
      if (!wasmSimMemo.has(key)) {
        wasmSimMemo.set(key, readWasmStats(ex, task.cacheBlocks, POLICY_CODE[task.policy]));
      }
      done += 1;
      onProgress({ phase: "sweep", completed: done, total: Math.max(total, 1) });
    }
    const totalMeasured = meta.totalMeasuredTokens;
    if (!wasmCeilingStats) {
      wasmCeilingStats = readWasmStats(ex, Math.max(ex.unique_blocks(), 1), POLICY_CODE.lru);
    }
    const ceilingHit = ex.ceiling_hit();
    const ceiling = {
      hitTokens: ceilingHit,
      totalTokens: totalMeasured,
      hitRate: totalMeasured ? ceilingHit / totalMeasured : 0,
      warmupRequests: meta.warmupRequests,
      usefulCacheBlockSamples: wasmCeilingStats.usefulCacheBlockSamples,
      usefulCacheSamples: wasmCeilingStats.usefulCacheSamples,
      usefulCacheRate: wasmCeilingStats.usefulCacheSamples
        ? wasmCeilingStats.usefulCacheBlockSamples / (wasmCeilingStats.usefulCacheSamples * Math.max(ex.unique_blocks(), 1))
        : 0,
    };
    const simLookup = (policy, cacheBlocks) => {
      const stats = wasmSimMemo.get(`${policy}|${cacheBlocks}`) || { hitTokens: 0, usefulCacheBlockSamples: 0, usefulCacheSamples: 0 };
      return {
        policy,
        cacheBlocks,
        warmupRequests: meta.warmupRequests,
        hitTokens: stats.hitTokens,
        totalTokens: totalMeasured,
        hitRate: totalMeasured ? stats.hitTokens / totalMeasured : 0,
        usefulCacheBlockSamples: stats.usefulCacheBlockSamples,
        usefulCacheSamples: stats.usefulCacheSamples,
        usefulCacheRate: cacheBlocks > 0 && stats.usefulCacheSamples
          ? stats.usefulCacheBlockSamples / (stats.usefulCacheSamples * cacheBlocks)
          : 0,
      };
    };
    const sweep = lab.assembleSweep(planned, meta, message.settings || {}, simLookup, ceiling);
    return {
      jobId: message.jobId,
      cacheKey: message.cacheKey,
      preset: message.preset || { id: meta.slimTrace.presetId, label: meta.slimTrace.presetLabel },
      trace: meta.slimTrace,
      sweep,
      timeStats: wasmTimeStats,
    };
  }

  function slimTrace(trace) {
    // The UI only needs summary-level fields; shipping the full requests array
    // (hundreds of thousands of block objects for a real trace) back across the
    // worker boundary is needless structured-clone and memory cost.
    return {
      presetId: trace.presetId,
      presetLabel: trace.presetLabel,
      sourceKind: trace.sourceKind,
      blockSize: trace.blockSize,
      sourceBlockSizeNote: trace.sourceBlockSizeNote,
      summary: trace.summary || {},
      requestCount: Array.isArray(trace.requests) ? trace.requests.length : (trace.summary && trace.summary.requests) || 0,
    };
  }

  // Turn a job message into a trace. A File is stream-parsed (gzip-decompressed
  // on the fly) so multi-GB / .gz uploads never materialize as one string;
  // legacy uploadText and synthetic presets stay supported.
  async function parseTraceFromMessage(lab, message, onParseProgress) {
    if (message.uploadFile) {
      let onBytes = null;
      if (onParseProgress) {
        let seen = 0;
        onBytes = (n) => {
          seen += n;
          onParseProgress(seen, message.fileSize || 0);
        };
      }
      const stream = lab.createTraceTextStream(message.uploadFile, !!message.gzip, onBytes);
      const opts = Object.assign({}, message.uploadOptions || {});
      if (message.maxEvents) opts.maxEvents = message.maxEvents;
      return lab.parseUploadedTraceStreaming(stream, opts);
    }
    if (message.uploadText != null) {
      return lab.parseUploadedTrace(message.uploadText, message.uploadOptions || {});
    }
    return lab.generateTrace(message.preset, message.params, message.seed);
  }

  async function runWorkerJob(message, onProgress) {
    ensureBrowserScripts(message || {});
    const lab = labModule();
    let trace;
    try {
      trace = await parseTraceFromMessage(lab, message, (bytes, total) =>
        onProgress({ phase: "parse", bytes, total }),
      );
    } catch (error) {
      throw new Error(`parse: ${(error && error.message) || error}`);
    }
    let sweep;
    try {
      sweep = lab.sweepCapacity(trace, message.model, message.settings || {}, (completed, total) =>
        onProgress({ phase: "sweep", completed, total }),
      );
    } catch (error) {
      throw new Error(`sweep: ${(error && error.message) || error}`);
    }
    let timeStats = null;
    try {
      timeStats = typeof lab.computeTimeSeries === "function" ? lab.computeTimeSeries(trace) : null;
    } catch (error) {
      timeStats = null;
    }
    return {
      jobId: message.jobId,
      cacheKey: message.cacheKey,
      preset: message.preset || { id: trace.presetId, label: trace.presetLabel },
      trace: slimTrace(trace),
      sweep,
      timeStats,
    };
  }

  // Per-worker plan cache for the pool protocol: the plan is shipped once per
  // trace key, then reused across every simulate task that worker is given.
  let cachedPlan = null;
  let cachedTraceKey = null;

  async function handleMessage(message, postResult) {
    if (!message) return;
    try {
      ensureBrowserScripts(message);
      const lab = labModule();
      if (message.type === "runWasm") {
        const onProgress = (info) => postResult(Object.assign({ type: "progress", jobId: message.jobId }, info));
        postResult({ type: "result", jobId: message.jobId, result: await runWasmJob(message, onProgress) });
        return;
      }
      if (message.type === "run") {
        const onProgress = (info) => postResult(Object.assign({ type: "progress", jobId: message.jobId }, info));
        postResult({ type: "result", jobId: message.jobId, result: await runWorkerJob(message, onProgress) });
        return;
      }
      if (message.type === "analyze") {
        clearWasm(); // free any large wasm instance before the JS-path plan
        const onParseProgress = (bytes, total) =>
          postResult({ type: "progress", jobId: message.jobId, phase: "parse", bytes, total });
        const trace = await parseTraceFromMessage(lab, message, onParseProgress);
        const analysis = lab.analyzeTrace(trace, { warmupFraction: message.warmupFraction });
        cachedPlan = analysis.plan;
        cachedTraceKey = message.traceKey;
        postResult({
          type: "analyzed",
          jobId: message.jobId,
          traceKey: message.traceKey,
          planBuffers: analysis.planBuffers,
          meta: analysis.meta,
          ceiling: analysis.ceiling,
          timeStats: analysis.timeStats,
        });
        return;
      }
      if (message.type === "simulate") {
        if (message.plan) {
          cachedPlan = lab.planFromBuffers(message.plan);
          cachedTraceKey = message.traceKey;
        }
        if (!cachedPlan || cachedTraceKey !== message.traceKey) {
          postResult({ type: "error", jobId: message.jobId, error: "plan-miss" });
          return;
        }
        postResult({
          type: "simulated",
          jobId: message.jobId,
          result: lab.simulatePlanPolicy(cachedPlan, message.cacheBlocks, message.policy),
        });
        return;
      }
    } catch (error) {
      postResult({
        type: "error",
        jobId: message.jobId,
        error: (error && error.message) || String(error),
        stack: (error && error.stack) || null,
      });
    }
  }

  if (typeof self !== "undefined" && typeof self.addEventListener === "function") {
    self.addEventListener("message", (event) => {
      handleMessage(event.data, (payload) => self.postMessage(payload));
    });
  }

  if (typeof module === "object" && module.exports) {
    module.exports = { runWorkerJob };
    try {
      const workerThreads = require("node:worker_threads");
      if (!workerThreads.isMainThread && workerThreads.parentPort) {
        workerThreads.parentPort.on("message", (message) => {
          handleMessage(message, (payload) => workerThreads.parentPort.postMessage(payload));
        });
      }
    } catch (error) {
      // Browser builds and plain CommonJS imports do not need worker_threads.
    }
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
