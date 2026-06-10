(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(require("./kv-cache-calculator.js"));
  } else {
    root.KVCacheLab = factory(root.KVCacheCalculator);
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function (calculator) {
  "use strict";

  if (!calculator || typeof calculator.calculate !== "function") {
    throw new Error("KVCacheLab requires KVCacheCalculator");
  }

  const BYTES_PER_GIB = calculator.BYTES_PER_GIB || 1024 ** 3;
  const DEFAULT_SEED = 20260528;
  const DEFAULT_WARMUP_FRACTION = 0.3;
  const DEFAULT_BLOCK_SIZE = 64;
  // Huge-upload guards. The JS fallback keeps event arrays compact while using
  // string hash keys for safe identity; for huge uploads the sweep plan copies
  // are still the limiter. We cap at 40M events and bound how many workers get
  // a plan copy during the sweep, so huge plans use fewer workers.  const UPLOAD_MAX_EVENTS = 40000000;
  const PLAN_BYTES_PER_EVENT = 17;
  const PLAN_COPY_BUDGET_BYTES = 1600000000;
  const UPLOAD_PRESET_ID = "__upload__";
  const DEFAULT_REQUESTS = 4000;
  const DEFAULT_CAPACITY_GIB_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
  const INPUT_DEBOUNCE_MS = 250;
  const FALLBACK_STEP_DELAY_MS = 16;
  const MAX_CACHE_ENTRIES = 12;
  const UPLOAD_HEAD_MAX_CHARS = 256 * 1024;
  const UPLOAD_HEAD_MAX_LINES = 64;
  const TIME_BUCKETS = 48;
  const THROUGHPUT_EPSILON = 0.001;
  const THROUGHPUT_DISPLAY_CAP = 1000;
  const REUSE_GAP_BINS = [
    { label: "<1s", max: 1 },
    { label: "1-3s", max: 3 },
    { label: "3-10s", max: 10 },
    { label: "10-30s", max: 30 },
    { label: "30-60s", max: 60 },
    { label: "1-3m", max: 180 },
    { label: "3-10m", max: 600 },
    { label: "10-30m", max: 1800 },
    { label: "30-60m", max: 3600 },
    { label: "1h+", max: Infinity },
  ];
  const POLICIES = ["fifo", "lru", "optimal"];
  const POLICY_LABELS = { fifo: "FIFO", lru: "LRU", optimal: "Optimal" };
  const POLICY_COLORS = { fifo: "#2563eb", lru: "#059669", optimal: "#d97706" };
  const POLICY_HELP = {
    fifo: "Evicts the oldest cached block first.",
    lru: "Evicts the least recently used cached block first.",
    optimal: "Belady/MIN computed with full knowledge of the future trace. Evicts the block whose next use is farthest away, or a block that will never be used again. Theoretical upper bound, not an online policy.",
  };
  const POLICY_TOOLTIP_LINES = {
    fifo: ["Evicts the oldest cached", "block first."],
    lru: ["Evicts the least recently used", "cached block first."],
    optimal: ["Belady/MIN: evict farthest future/unused block; theoretical upper bound."],
  };
  const SOURCE_LABELS = {
    mooncake_fast25: "Mooncake FAST25 Tool&Agent",
    nvidia_aiperf_mooncake: "NVIDIA AIPerf Mooncake traces",
    qwen_bailian_trace: "Qwen Bailian traces",
    ragpulse: "RAGPulse",
    lmcache_agentic_traces: "LMCache agentic traces",
    semianalysis_weka_no_subagents: "SemiAnalysis Weka Claude Code traces",
    semianalysis_weka_with_subagents_256k: "SemiAnalysis Weka Claude Code sub-agent traces",
    kv_cache_tester: "kv-cache-tester",
    exgentic_agent_traces: "Exgentic agent traces",
    burstgpt: "BurstGPT",
    swissai_serving_trace: "SwissAI serving trace",
    sglang_hicache: "SGLang HiCache docs",
  };
  const METRIC_HELP = {
    "Trace requests": "Number of normalized requests in the real trace used to precompute this curve. Each request contributes prefill input blocks.",
    "Warmup skipped": "Requests used only to warm the cache before measuring hit rate. They still populate and evict cache blocks, but their hits and misses are excluded.",
    "Avg input tokens": "Average prefill input tokens per request in the normalized trace. Output tokens are counted only when they appear in later request input/history.",
    "Unique blocks": "Distinct block identities in the normalized trace. This is the trace working-set size before converting the selected model budget into cache-block capacity.",
    "Hit rate ceiling": "Infinite-cache prefill token hit rate after warmup. A finite FIFO/LRU/Optimal cache cannot exceed this trace-level reuse upper bound.",
    "Native block size": "Source-native or declared block granularity used by the normalized trace. Real trace mode keeps this fixed instead of reinterpreting the trace at another block size.",
    "Max cache blocks": "Number of cache blocks that fit at the largest GiB budget on the x axis, after converting model precision and block size into bytes per block.",
    "Loaded time span": "Wall-clock duration covered by the parsed requests (first to last timestamp). For a truncated upload this is the span of the loaded prefix, not the whole file.",
  };

  function toNumber(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function toPositiveNumber(value, fallback) {
    const parsed = toNumber(value, fallback);
    return parsed > 0 ? parsed : fallback;
  }

  function toInteger(value, fallback) {
    return Math.max(0, Math.floor(toNumber(value, fallback)));
  }

  function toPositiveInteger(value, fallback) {
    return Math.max(1, Math.floor(toPositiveNumber(value, fallback)));
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function createRng(seed) {
    let state = toInteger(seed, DEFAULT_SEED) >>> 0;
    return function rng() {
      state += 0x6d2b79f5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function jitter(base, spread, rng) {
    const scale = 1 + (rng() * 2 - 1) * spread;
    return Math.max(1, Math.round(base * scale));
  }

  function buildWeightedSampler(count, skew) {
    const weights = [];
    let total = 0;
    for (let index = 0; index < count; index += 1) {
      const weight = 1 / Math.pow(index + 1, Math.max(0, skew));
      weights.push(weight);
      total += weight;
    }
    let cumulative = 0;
    const cdf = weights.map((weight) => {
      cumulative += weight / total;
      return cumulative;
    });
    return function sample(rng) {
      const value = rng();
      return cdf.findIndex((entry) => value <= entry);
    };
  }

  function makeBlocks(prefix, tokens, blockSize) {
    const safeTokens = toInteger(tokens, 0);
    if (safeTokens <= 0) return [];
    const size = toPositiveInteger(blockSize, DEFAULT_BLOCK_SIZE);
    const count = Math.ceil(safeTokens / size);
    const blocks = [];
    for (let index = 0; index < count; index += 1) {
      const remaining = safeTokens - index * size;
      blocks.push({ id: `${prefix}:${index}`, tokens: Math.min(size, remaining) });
    }
    return blocks;
  }

  function cloneBlocks(blocks) {
    return blocks.map((block) => ({ id: block.id, tokens: block.tokens }));
  }

  // Shared trace primitives — also re-exported by scripts/lib/kv-cache-lab-traces.mjs
  // so offline precompute and the browser/upload path stay byte-for-byte identical.
  function blockTokens(inputLength, blockSize, index, count) {
    if (count <= 0) return 0;
    if (!Number.isFinite(inputLength) || inputLength <= 0) return blockSize;
    if (!Number.isFinite(blockSize) || blockSize <= 0) return Math.max(1, Math.round(inputLength / count));
    const remaining = inputLength - index * blockSize;
    if (remaining <= 0) return 1;
    return Math.max(1, Math.min(blockSize, remaining));
  }

  function namespacedBlocks(ids, namespace, inputLength, blockSize) {
    const safeIds = Array.isArray(ids) ? ids : [];
    return safeIds.map((id, index) => ({
      id: `${namespace}:${String(id)}`,
      tokens: blockTokens(inputLength, blockSize, index, safeIds.length),
    }));
  }

  function inspectUploadedTraceRecord(record) {
    if (!record || typeof record !== "object" || Array.isArray(record)) {
      return { valid: false, error: "Uploaded trace lines must be JSON objects." };
    }
    if (!Array.isArray(record.hash_ids) || !record.hash_ids.length) {
      return { valid: false, error: 'Uploaded trace records must include a non-empty "hash_ids" array.' };
    }
    if (!Number.isFinite(Number(record.input_length)) || Number(record.input_length) <= 0) {
      return { valid: false, error: 'Uploaded trace records must include a positive "input_length".' };
    }
    const blockSize = Math.floor(Number(record.block_size));
    if (!Number.isFinite(Number(record.block_size)) || blockSize <= 0) {
      return { valid: false, error: 'Uploaded trace records must include a positive "block_size".' };
    }
    return { valid: true, blockSize };
  }

  function inspectUploadedTraceHeadText(text, options) {
    const opts = options || {};
    const maxLines = toPositiveInteger(opts.maxLines, UPLOAD_HEAD_MAX_LINES);
    const source = String(text || "");
    const len = Math.min(source.length, toPositiveInteger(opts.maxChars, UPLOAD_HEAD_MAX_CHARS));
    let pos = 0;
    let inspectedLines = 0;
    let nonEmptyLines = 0;
    let parseErrors = 0;
    let firstSchemaError = "";
    let blockSize = 0;
    let validRecords = 0;
    while (pos < len && inspectedLines < maxLines) {
      let newline = source.indexOf("\n", pos);
      if (newline < 0 || newline > len) newline = len;
      const line = source.slice(pos, newline).trim();
      pos = newline + 1;
      if (!line) continue;
      inspectedLines += 1;
      nonEmptyLines += 1;
      let record;
      try {
        record = JSON.parse(line);
      } catch (error) {
        parseErrors += 1;
        continue;
      }
      const checked = inspectUploadedTraceRecord(record);
      if (!checked.valid) {
        if (!firstSchemaError) firstSchemaError = checked.error;
        continue;
      }
      validRecords += 1;
      if (!blockSize) blockSize = checked.blockSize;
      else if (checked.blockSize !== blockSize) {
        return {
          valid: false,
          blockSize,
          error: `Uploaded trace block_size must be consistent. Saw ${blockSize} and ${checked.blockSize} in the file head.`,
        };
      }
    }
    if (validRecords > 0) return { valid: true, blockSize, validRecords, parseErrors };
    if (firstSchemaError) return { valid: false, error: firstSchemaError, parseErrors };
    if (parseErrors > 0) {
      return {
        valid: false,
        error: 'Uploaded trace must be JSONL: each non-empty line must be a JSON object with "hash_ids", "input_length", and "block_size".',
        parseErrors,
      };
    }
    if (nonEmptyLines === 0) {
      return { valid: false, error: "Uploaded trace appears empty; expected JSONL records." };
    }
    return { valid: false, error: 'No valid uploaded trace records found in the file head. Expected JSONL records with "hash_ids", "input_length", and "block_size".' };
  }

  function normalizeMooncakeRecord(record, source) {
    const src = source || {};
    const blockSize = toPositiveInteger(src.nativeBlockSize, 512);
    const inputLength = toPositiveInteger(record.input_length, 1);
    return {
      id: record.id,
      timestamp: toNumber(record.timestamp, 0),
      inputTokens: inputLength,
      outputTokens: Math.max(0, Math.floor(toNumber(record.output_length, 0))),
      inputBlocks: namespacedBlocks(record.hash_ids, src.id || "mooncake", inputLength, blockSize),
      appendBlocks: [],
    };
  }

  // Infinite-cache reuse over the interned event stream: integer ids index a
  // Uint8 "seen" table instead of hashing block-id strings into a Set. Produces
  // the same hitTokens/totalTokens as the requests-based path below.
  function infiniteCacheReuseFlat(flat, opts) {
    const requestCount = flat.requestCount;
    const warmupRequests = Math.min(
      requestCount,
      Math.max(0, Math.floor(toNumber(opts.warmupRequests, requestCount * toNumber(opts.warmupFraction, DEFAULT_WARMUP_FRACTION)))),
    );
    const ids = flat.eventIds;
    const tokens = flat.eventTokens;
    const requestOf = flat.eventRequest;
    const isInput = flat.eventIsInput;
    const seen = new Uint8Array(flat.uniqueBlocks);
    let hitTokens = 0;
    let totalTokens = 0;
    for (let index = 0; index < ids.length; index += 1) {
      const id = ids[index];
      if (isInput[index] && requestOf[index] >= warmupRequests) {
        const tok = tokens[index];
        totalTokens += tok;
        if (seen[id]) hitTokens += tok;
      }
      seen[id] = 1;
    }
    return { warmupRequests, hitTokens, totalTokens, hitRate: totalTokens ? hitTokens / totalTokens : 0 };
  }

  function infiniteCacheReuse(trace, options) {
    const opts = options || {};
    if (trace && trace.__flat) return infiniteCacheReuseFlat(trace.__flat, opts);
    const requests = Array.isArray(trace.requests) ? trace.requests : [];
    const warmupRequests = Math.min(
      requests.length,
      Math.max(0, Math.floor(toNumber(opts.warmupRequests, requests.length * toNumber(opts.warmupFraction, DEFAULT_WARMUP_FRACTION)))),
    );
    const cache = new Set();
    let hitTokens = 0;
    let totalTokens = 0;
    requests.forEach((request, requestIndex) => {
      const measured = requestIndex >= warmupRequests;
      request.inputBlocks.forEach((block) => {
        const tokens = toNumber(block.tokens, 0);
        const hit = cache.has(block.id);
        if (measured) {
          totalTokens += tokens;
          if (hit) hitTokens += tokens;
        }
        cache.add(block.id);
      });
      (request.appendBlocks || []).forEach((block) => cache.add(block.id));
    });
    return { warmupRequests, hitTokens, totalTokens, hitRate: totalTokens ? hitTokens / totalTokens : 0 };
  }

  // Growable typed-array event stream. The upload parser appends interned events
  // here in a single pass, so we never allocate per-block {id, tokens} objects
  // nor build a boxed [] that flattenTrace would later copy into a typed array.
  // finalize() trims to the exact length so a cached result does not retain the
  // doubled backing buffer.
  function createEventStream(initialCapacity) {
    let capacity = Math.max(16, Math.floor(initialCapacity) || 0);
    let ids = new Int32Array(capacity);
    let tokens = new Int32Array(capacity);
    let request = new Int32Array(capacity);
    let isInput = new Uint8Array(capacity);
    let length = 0;
    function grow() {
      const next = capacity * 2;
      const grownIds = new Int32Array(next); grownIds.set(ids); ids = grownIds;
      const grownTokens = new Int32Array(next); grownTokens.set(tokens); tokens = grownTokens;
      const grownRequest = new Int32Array(next); grownRequest.set(request); request = grownRequest;
      const grownIsInput = new Uint8Array(next); grownIsInput.set(isInput); isInput = grownIsInput;
      capacity = next;
    }
    return {
      push(id, tok, requestIndex, input) {
        if (length >= capacity) grow();
        ids[length] = id;
        tokens[length] = tok;
        request[length] = requestIndex;
        isInput[length] = input;
        length += 1;
      },
      finalize(uniqueBlocks, requestCount) {
        // If the arrays are nearly full (a pre-sized capped parse) hand back
        // views to skip a transient full-size copy; otherwise trim the doubling
        // slack with slice. Either way the worker→main clone is ~length-sized.
        const trim = (array) => (length >= capacity * 0.875 ? array.subarray(0, length) : array.slice(0, length));
        return {
          eventIds: trim(ids),
          eventTokens: trim(tokens),
          eventRequest: trim(request),
          eventIsInput: trim(isInput),
          requestCount,
          uniqueBlocks,
        };
      },
    };
  }

  function canonicalHashId(value) {
    if (typeof value === "number") {
      if (!Number.isSafeInteger(value)) {
        throw new Error(
          'Uploaded trace hash_ids include an unsafe JSON number. Store large hash ids as strings, or use the WASM exact-u64 upload path.',
        );
      }
      return String(value);
    }
    return String(value);
  }

  // String-keyed fallback interner: preserves string hash ids exactly and keeps
  // safe JSON numbers stable by canonicalizing both forms through String(id).
  function createHashInterner() {
    const ids = new Map();
    function intern(value) {
      const key = canonicalHashId(value);
      const existing = ids.get(key);
      if (existing !== undefined) return existing;
      const assigned = ids.size;
      ids.set(key, assigned);
      return assigned;
    }
    return { intern, size: () => ids.size };
  }

  // Single source of truth for turning JSONL record lines into interned events
  // (one Mooncake-schema record per line: {"hash_ids":[...], "input_length":N}).
  // Both the whole-string parser and the streaming parser feed lines here, so
  // they stay byte-for-byte consistent. Honors maxRecords/maxEvents caps
  // (checked after each complete request) so huge uploads stay memory-bounded.
  // Uploaded traces must declare one source-native block_size consistently in
  // every valid hash record. The page displays it read-only; it is not a user
  // setting because it defines the trace identity granularity.
  function createTraceIngester(options) {
    const opts = options || {};
    let blockSize = toInteger(opts.blockSize, 0);
    const maxRecords = toInteger(opts.maxRecords, 0);
    const maxEvents = toInteger(opts.maxEvents, 0);
    // When capping, pre-size near the cap so the event arrays don't grow by
    // doubling (which transiently doubles memory) during a multi-GB parse.
    const stream = createEventStream(maxEvents ? maxEvents + 1024 : opts.estimatedEvents || 1024);
    const interner = createHashInterner();
    const timestamps = [];
    const requestStarts = [];
    let totalInputTokens = 0;
    let parseErrors = 0;
    let skipped = 0;
    let eventCount = 0;
    let requestCount = 0;
    let capped = false;
    let missingBlockSize = 0;
    let inconsistentBlockSize = 0;
    let missingInputLength = 0;
    let tMin = Infinity;
    let tMax = -Infinity;

    // Returns false once a cap is reached (the caller should stop feeding lines).
    function ingestLine(line) {
      if (capped) return false;
      let record;
      try {
        record = JSON.parse(line);
      } catch (error) {
        parseErrors += 1;
        return true;
      }
      if (!record || !Array.isArray(record.hash_ids) || !record.hash_ids.length) {
        skipped += 1;
        return true;
      }
      const recordBlockSize = toInteger(record.block_size, 0);
      if (recordBlockSize <= 0) {
        missingBlockSize += 1;
        return true;
      }
      if (!blockSize) blockSize = recordBlockSize;
      if (recordBlockSize !== blockSize) {
        inconsistentBlockSize += 1;
        return true;
      }
      const inputLength = toInteger(record.input_length, 0);
      if (inputLength <= 0) {
        missingInputLength += 1;
        return true;
      }
      const hashIds = record.hash_ids;
      const count = hashIds.length;
      requestStarts.push(eventCount);
      const timestamp = toNumber(record.timestamp, 0);
      timestamps.push(timestamp);
      if (timestamp < tMin) tMin = timestamp;
      if (timestamp > tMax) tMax = timestamp;
      for (let blockIndex = 0; blockIndex < count; blockIndex += 1) {
        const intId = interner.intern(hashIds[blockIndex]);
        const tok = blockTokens(inputLength, blockSize, blockIndex, count);
        stream.push(intId, tok, requestCount, 1);
        totalInputTokens += tok;
        eventCount += 1;
      }
      requestCount += 1;
      if ((maxRecords && requestCount >= maxRecords) || (maxEvents && eventCount >= maxEvents)) {
        capped = true;
        return false;
      }
      return true;
    }

    function finish() {
      if (missingBlockSize > 0) {
        throw new Error(`Uploaded trace records must include block_size. Found ${missingBlockSize} valid hash record(s) without it.`);
      }
      if (inconsistentBlockSize > 0) {
        throw new Error(`Uploaded trace block_size must be consistent. Found ${inconsistentBlockSize} record(s) with a different block_size.`);
      }
      if (missingInputLength > 0) {
        throw new Error(`Uploaded trace records must include a positive "input_length". Found ${missingInputLength} valid hash record(s) without it.`);
      }
      if (!requestCount) {
        throw new Error(
          'No valid uploaded trace records found. Each line must be JSON with a non-empty "hash_ids" array, a positive "block_size", and a positive "input_length".',
        );
      }
      requestStarts.push(eventCount);
      const uniqueBlocks = interner.size();
      const summary = {
        requests: requestCount,
        totalInputTokens,
        averageInputTokens: requestCount ? totalInputTokens / requestCount : 0,
        uniqueBlocks,
        parseErrors,
        skipped,
      };
      // Only present when truncated, so non-capped output is unchanged.
      if (capped) summary.capped = true;
      // Wall-clock span the parsed (possibly truncated) requests cover.
      if (tMax > tMin) {
        summary.tStart = tMin;
        summary.tEnd = tMax;
        summary.timeSpanSeconds = tMax - tMin;
      }
      return {
        presetId: UPLOAD_PRESET_ID,
        presetLabel: opts.label || "Uploaded trace",
        sourceKind: "hash",
        blockSize,
        __flat: stream.finalize(uniqueBlocks, requestCount),
        __timestamps: Float64Array.from(timestamps),
        __requestStarts: Int32Array.from(requestStarts),
        summary,
      };
    }

    return { ingestLine, finish, isCapped: () => capped };
  }

  function parseUploadedTrace(text, options) {
    const source = String(text || "");
    const ingester = createTraceIngester(Object.assign({ estimatedEvents: (source.length / 32) | 0 }, options || {}));
    const len = source.length;
    let pos = 0;
    while (pos < len) {
      let newline = source.indexOf("\n", pos);
      if (newline < 0) newline = len;
      const line = source.slice(pos, newline).trim();
      pos = newline + 1;
      if (!line) continue;
      if (!ingester.ingestLine(line)) break;
    }
    return ingester.finish();
  }

  // Streaming parser: consumes an async iterable of text chunks (e.g. a
  // gzip-decompressed File stream) and builds the trace incrementally, never
  // holding the whole text — the only way a multi-GB / .gz upload fits.
  // onProgress(processedChars) is optional; maxEvents/maxRecords bound memory.
  async function parseUploadedTraceStreaming(chunkIterable, options) {
    const opts = options || {};
    const onProgress = typeof opts.onProgress === "function" ? opts.onProgress : null;
    const ingester = createTraceIngester(opts);
    let buffer = "";
    let processed = 0;
    let stop = false;
    for await (const chunk of chunkIterable) {
      if (stop) break;
      const text = typeof chunk === "string" ? chunk : String(chunk);
      processed += text.length;
      buffer += text;
      let start = 0;
      let newline = buffer.indexOf("\n", start);
      while (newline >= 0) {
        const line = buffer.slice(start, newline).trim();
        start = newline + 1;
        if (line && !ingester.ingestLine(line)) {
          stop = true;
          break;
        }
        newline = buffer.indexOf("\n", start);
      }
      // Keep only the unfinished tail; never accumulate the whole file.
      if (start > 0) buffer = buffer.slice(start);
      if (onProgress) onProgress(processed);
    }
    if (!stop) {
      const tail = buffer.trim();
      if (tail) ingester.ingestLine(tail);
    }
    return ingester.finish();
  }

  // Browser/Worker only: File -> async iterable of decoded text chunks,
  // decompressing gzip on the fly. onBytes(compressedByteLength) reports progress
  // vs file.size. Returns an async generator (driven by getReader) rather than
  // the raw stream, so callers don't depend on ReadableStream async-iteration
  // (only shipped in newer Chrome).
  function createTraceTextStream(file, gzip, onBytes) {
    let raw = file.stream();
    if (typeof onBytes === "function" && typeof TransformStream === "function") {
      raw = raw.pipeThrough(
        new TransformStream({
          transform(chunk, controller) {
            onBytes(chunk.byteLength || chunk.length || 0);
            controller.enqueue(chunk);
          },
        }),
      );
    }
    const bytes = gzip ? raw.pipeThrough(new DecompressionStream("gzip")) : raw;
    const textStream = bytes.pipeThrough(new TextDecoderStream());
    return (async function* () {
      const reader = textStream.getReader();
      try {
        for (;;) {
          const next = await reader.read();
          if (next.done) break;
          if (next.value) yield next.value;
        }
      } finally {
        // cancel() (not just releaseLock) propagates upstream through the gzip +
        // file streams, so hitting the event cap stops decompressing/reading the
        // rest of the file instead of draining it in the background.
        try {
          await reader.cancel();
        } catch (error) {
          /* stream already closed/errored */
        }
      }
    })();
  }

  // Trace-level temporal statistics (independent of cache policy/capacity):
  // walk events in timestamp order with an infinite "seen" set and bucket by
  // wall-clock time. Per bucket we get arriving / reused / newly-created input
  // tokens, which yields hit-rate-over-time and KV-cache production rate; the
  // gap between consecutive uses of a block feeds a reuse-time histogram.
  // Returns null when the trace has no usable timestamps (e.g. synthetic).
  function computeTimeSeries(trace) {
    if (trace && trace.__flat && trace.__timestamps && trace.__requestStarts) {
      return timeSeriesFromFlat(trace);
    }
    return timeSeriesFromRequests(trace);
  }

  // Shared tail: turn the per-bucket accumulators into the panel-ready result.
  // Both the flat and requests walkers fill identical accumulators, so the
  // numbers (and object shape) are the same regardless of which path ran.
  function buildTimeSeriesResult(tMin, tMax, span, dt, bucketTotal, bucketHit, gapTokens, gapCount, totalTokens, reuseTokens) {
    const timeBuckets = [];
    for (let i = 0; i < TIME_BUCKETS; i += 1) {
      const total = bucketTotal[i];
      const hit = bucketHit[i];
      timeBuckets.push({
        tStart: tMin + i * dt,
        tEnd: tMin + (i + 1) * dt,
        offset: i * dt,
        totalTokens: total,
        hitTokens: hit,
        newTokens: total - hit,
        hitRate: total ? hit / total : 0,
        tokensPerSec: dt > 0 ? total / dt : 0,
        newTokensPerSec: dt > 0 ? (total - hit) / dt : 0,
      });
    }
    const reuseHistogram = REUSE_GAP_BINS.map((bin, i) => ({
      label: bin.label,
      tokens: gapTokens[i],
      count: gapCount[i],
      share: reuseTokens ? gapTokens[i] / reuseTokens : 0,
    }));
    return {
      tMin,
      tMax,
      span,
      bucketSeconds: dt,
      timeBuckets,
      reuseHistogram,
      totals: { totalTokens, reuseTokens, newTokens: totalTokens - reuseTokens },
    };
  }

  // Fast path for uploaded traces: walk the interned event stream in timestamp
  // order using integer ids into a typed last-seen table, instead of a string
  // Map over per-block objects.
  function timeSeriesFromFlat(trace) {
    const flat = trace.__flat;
    const ts = trace.__timestamps;
    const starts = trace.__requestStarts;
    const requestCount = flat.requestCount;
    if (requestCount < 2) return null;
    let tMin = Infinity;
    let tMax = -Infinity;
    for (let r = 0; r < requestCount; r += 1) {
      const t = ts[r];
      if (t < tMin) tMin = t;
      if (t > tMax) tMax = t;
    }
    if (!(tMax > tMin)) return null;
    const order = new Array(requestCount);
    for (let r = 0; r < requestCount; r += 1) order[r] = r;
    order.sort((a, b) => ts[a] - ts[b]);

    const span = tMax - tMin;
    const dt = span / TIME_BUCKETS;
    const bucketTotal = new Float64Array(TIME_BUCKETS);
    const bucketHit = new Float64Array(TIME_BUCKETS);
    const gapTokens = new Float64Array(REUSE_GAP_BINS.length);
    const gapCount = new Float64Array(REUSE_GAP_BINS.length);
    const lastTs = new Float64Array(flat.uniqueBlocks);
    const hasLast = new Uint8Array(flat.uniqueBlocks);
    const ids = flat.eventIds;
    const tokens = flat.eventTokens;
    const isInput = flat.eventIsInput;
    let totalTokens = 0;
    let reuseTokens = 0;

    for (let oi = 0; oi < requestCount; oi += 1) {
      const r = order[oi];
      const t = ts[r];
      let bucket = Math.floor(((t - tMin) / span) * TIME_BUCKETS);
      if (bucket >= TIME_BUCKETS) bucket = TIME_BUCKETS - 1;
      if (bucket < 0) bucket = 0;
      const end = starts[r + 1];
      for (let k = starts[r]; k < end; k += 1) {
        if (!isInput[k]) continue;
        const id = ids[k];
        const tok = tokens[k];
        bucketTotal[bucket] += tok;
        totalTokens += tok;
        if (hasLast[id]) {
          bucketHit[bucket] += tok;
          reuseTokens += tok;
          const gap = t - lastTs[id];
          let bin = 0;
          while (bin < REUSE_GAP_BINS.length - 1 && gap >= REUSE_GAP_BINS[bin].max) bin += 1;
          gapTokens[bin] += tok;
          gapCount[bin] += 1;
        }
        lastTs[id] = t;
        hasLast[id] = 1;
      }
    }
    return buildTimeSeriesResult(tMin, tMax, span, dt, bucketTotal, bucketHit, gapTokens, gapCount, totalTokens, reuseTokens);
  }

  function timeSeriesFromRequests(trace) {
    const requests = Array.isArray(trace && trace.requests) ? trace.requests : [];
    if (!requests.length) return null;
    const order = [];
    let tMin = Infinity;
    let tMax = -Infinity;
    for (let i = 0; i < requests.length; i += 1) {
      const t = Number(requests[i].timestamp);
      if (!Number.isFinite(t)) continue;
      order.push(i);
      if (t < tMin) tMin = t;
      if (t > tMax) tMax = t;
    }
    if (order.length < 2 || !(tMax > tMin)) return null;
    order.sort((a, b) => Number(requests[a].timestamp) - Number(requests[b].timestamp));

    const span = tMax - tMin;
    const dt = span / TIME_BUCKETS;
    const bucketTotal = new Float64Array(TIME_BUCKETS);
    const bucketHit = new Float64Array(TIME_BUCKETS);
    const gapTokens = new Float64Array(REUSE_GAP_BINS.length);
    const gapCount = new Float64Array(REUSE_GAP_BINS.length);
    const lastTs = new Map();
    let totalTokens = 0;
    let reuseTokens = 0;

    for (let k = 0; k < order.length; k += 1) {
      const request = requests[order[k]];
      const t = Number(request.timestamp);
      let bucket = Math.floor(((t - tMin) / span) * TIME_BUCKETS);
      if (bucket >= TIME_BUCKETS) bucket = TIME_BUCKETS - 1;
      if (bucket < 0) bucket = 0;
      const blocks = request.inputBlocks || [];
      for (let b = 0; b < blocks.length; b += 1) {
        const id = blocks[b].id;
        const tok = toNumber(blocks[b].tokens, 0);
        bucketTotal[bucket] += tok;
        totalTokens += tok;
        const prev = lastTs.get(id);
        if (prev !== undefined) {
          bucketHit[bucket] += tok;
          reuseTokens += tok;
          const gap = t - prev;
          let bin = 0;
          while (bin < REUSE_GAP_BINS.length - 1 && gap >= REUSE_GAP_BINS[bin].max) bin += 1;
          gapTokens[bin] += tok;
          gapCount[bin] += 1;
        }
        lastTs.set(id, t);
      }
    }
    return buildTimeSeriesResult(tMin, tMax, span, dt, bucketTotal, bucketHit, gapTokens, gapCount, totalTokens, reuseTokens);
  }

  function normalizePreset(preset) {
    if (!preset || typeof preset !== "object") {
      return { id: "custom", label: "Custom", defaults: {} };
    }
    return {
      id: preset.id || "custom",
      label: preset.label || preset.id || "Custom",
      summary: preset.summary || "",
      sources: Array.isArray(preset.sources) ? preset.sources : [],
      defaults: preset.defaults || {},
    };
  }

  function sessionTurnLimit(averageTurns, rng) {
    const spread = 0.65 + rng() * 0.9;
    return Math.max(1, Math.round(averageTurns * spread));
  }

  function createSession(index, presetId, params, topicCount, rng) {
    return {
      id: index,
      epoch: 0,
      turn: 0,
      maxTurns: sessionTurnLimit(params.average_turns, rng),
      topic: Math.floor(rng() * topicCount),
      history: [],
      prefix: `trace:${presetId}:session:${index}`,
    };
  }

  function resetSession(session, params, topicCount, rng) {
    session.epoch += 1;
    session.turn = 0;
    session.maxTurns = sessionTurnLimit(params.average_turns, rng);
    session.topic = Math.floor(rng() * topicCount);
    session.history = [];
  }

  function generateTrace(presetInput, inputParams, seed) {
    const preset = normalizePreset(presetInput);
    const defaults = preset.defaults || {};
    const params = {
      requests: toPositiveInteger(inputParams && inputParams.requests, DEFAULT_REQUESTS),
      blockSize: toPositiveInteger(inputParams && inputParams.blockSize, DEFAULT_BLOCK_SIZE),
      sessions: toPositiveInteger(inputParams && inputParams.sessions, defaults.sessions || 120),
      average_turns: toPositiveInteger(inputParams && inputParams.average_turns, defaults.average_turns || 8),
      shared_prefix_tokens: toInteger(inputParams && inputParams.shared_prefix_tokens, defaults.shared_prefix_tokens || 0),
      document_tokens: toInteger(inputParams && inputParams.document_tokens, defaults.document_tokens || 0),
      per_turn_input_tokens: toPositiveInteger(inputParams && inputParams.per_turn_input_tokens, defaults.per_turn_input_tokens || 128),
      output_tokens: toInteger(inputParams && inputParams.output_tokens, defaults.output_tokens || 256),
      reuse_skew: toNumber(inputParams && inputParams.reuse_skew, defaults.reuse_skew || 1),
      burstiness: clamp(toNumber(inputParams && inputParams.burstiness, defaults.burstiness || 0), 0, 1),
    };
    const rng = createRng(seed || DEFAULT_SEED);
    const topicCount = Math.max(4, Math.round(Math.sqrt(params.sessions)));
    const sampleSession = buildWeightedSampler(params.sessions, params.reuse_skew);
    const sharedBlocks = makeBlocks(`trace:${preset.id}:shared`, params.shared_prefix_tokens, params.blockSize);
    const topicBlocks = Array.from({ length: topicCount }, (_, index) =>
      makeBlocks(`trace:${preset.id}:topic:${index}`, params.document_tokens, params.blockSize)
    );
    const sessions = Array.from({ length: params.sessions }, (_, index) =>
      createSession(index, preset.id, params, topicCount, rng)
    );
    const requests = [];
    let lastSession = null;
    let totalInputTokens = 0;

    for (let requestIndex = 0; requestIndex < params.requests; requestIndex += 1) {
      let session = null;
      if (lastSession && lastSession.turn < lastSession.maxTurns && rng() < params.burstiness) {
        session = lastSession;
      } else {
        session = sessions[sampleSession(rng)];
      }
      if (session.turn >= session.maxTurns) resetSession(session, params, topicCount, rng);

      const turnInputTokens = jitter(params.per_turn_input_tokens, 0.35, rng);
      const outputTokens = params.output_tokens > 0 ? jitter(params.output_tokens, 0.3, rng) : 0;
      const userBlocks = makeBlocks(
        `${session.prefix}:epoch:${session.epoch}:turn:${session.turn}:user`,
        turnInputTokens,
        params.blockSize,
      );
      const outputBlocks = makeBlocks(
        `${session.prefix}:epoch:${session.epoch}:turn:${session.turn}:output`,
        outputTokens,
        params.blockSize,
      );
      const inputBlocks = []
        .concat(cloneBlocks(sharedBlocks))
        .concat(cloneBlocks(topicBlocks[session.topic] || []))
        .concat(cloneBlocks(session.history))
        .concat(cloneBlocks(userBlocks));
      const inputTokens = inputBlocks.reduce((total, block) => total + block.tokens, 0);
      totalInputTokens += inputTokens;

      requests.push({
        id: requestIndex,
        sessionId: session.id,
        turn: session.turn,
        inputBlocks,
        appendBlocks: cloneBlocks(outputBlocks),
        inputTokens,
        outputTokens,
      });

      session.history = session.history.concat(cloneBlocks(userBlocks), cloneBlocks(outputBlocks));
      session.turn += 1;
      lastSession = session;
    }

    return {
      presetId: preset.id,
      presetLabel: preset.label,
      blockSize: params.blockSize,
      params,
      requests,
      summary: {
        requests: requests.length,
        sessions: params.sessions,
        totalInputTokens,
        averageInputTokens: requests.length ? totalInputTokens / requests.length : 0,
      },
    };
  }

  function normalizeBlock(block, fallbackTokens) {
    if (typeof block === "string") return { id: block, tokens: fallbackTokens || 1 };
    return { id: String(block.id), tokens: toPositiveInteger(block.tokens, fallbackTokens || 1) };
  }

  function normalizeRequests(trace) {
    if (trace && Array.isArray(trace.normalizedRequests)) return trace.normalizedRequests;
    const rawRequests = Array.isArray(trace) ? trace : trace && trace.requests;
    if (!Array.isArray(rawRequests)) return [];
    const fallbackTokens = trace && trace.blockSize ? trace.blockSize : DEFAULT_BLOCK_SIZE;
    return rawRequests.map((request, index) => ({
      id: request.id || index,
      inputBlocks: (request.inputBlocks || request.input || []).map((block) => normalizeBlock(block, fallbackTokens)),
      appendBlocks: (request.appendBlocks || request.append || []).map((block) => normalizeBlock(block, fallbackTokens)),
    }));
  }

  // Flatten a trace into a typed-array event stream once, interning block ids
  // to integers. Input and output (append) blocks both populate the cache; only
  // input blocks count toward hit statistics. Reused across every capacity point
  // and policy in a sweep so the heavy O(events) work happens a single time.
  function flattenTrace(trace) {
    const requests = normalizeRequests(trace);
    const idMap = new Map();
    const eventIds = [];
    const eventTokens = [];
    const eventRequest = [];
    const eventIsInput = [];
    function pushEvent(block, requestIndex, isInput) {
      let intId = idMap.get(block.id);
      if (intId === undefined) {
        intId = idMap.size;
        idMap.set(block.id, intId);
      }
      eventIds.push(intId);
      eventTokens.push(toNumber(block.tokens, 0));
      eventRequest.push(requestIndex);
      eventIsInput.push(isInput);
    }
    requests.forEach((request, requestIndex) => {
      request.inputBlocks.forEach((block) => pushEvent(block, requestIndex, 1));
      (request.appendBlocks || []).forEach((block) => pushEvent(block, requestIndex, 0));
    });
    return {
      eventIds: Int32Array.from(eventIds),
      eventTokens: Int32Array.from(eventTokens),
      eventRequest: Int32Array.from(eventRequest),
      eventIsInput: Uint8Array.from(eventIsInput),
      requestCount: requests.length,
      uniqueBlocks: idMap.size,
    };
  }

  function planWarmupRequests(requestCount, options) {
    const opts = options || {};
    const fraction = toNumber(opts.warmupFraction, DEFAULT_WARMUP_FRACTION);
    const raw =
      Number.isFinite(Number(opts.warmupRequests)) ? Number(opts.warmupRequests) : requestCount * fraction;
    return clamp(toInteger(raw, 0), 0, requestCount);
  }

  function buildExecutionPlan(trace, options) {
    const flat = trace && trace.__flat ? trace.__flat : flattenTrace(trace);
    const warmupRequests = planWarmupRequests(flat.requestCount, options);
    const length = flat.eventIds.length;
    const nextInput = new Int32Array(length);
    const never = length + 1;
    const lastInput = new Int32Array(flat.uniqueBlocks).fill(-1);
    let totalMeasuredTokens = 0;
    for (let index = length - 1; index >= 0; index -= 1) {
      const id = flat.eventIds[index];
      const seen = lastInput[id];
      nextInput[index] = seen >= 0 ? seen : never;
      if (flat.eventIsInput[index]) {
        lastInput[id] = index;
        if (flat.eventRequest[index] >= warmupRequests) totalMeasuredTokens += flat.eventTokens[index];
      }
    }
    return Object.assign({}, flat, { nextInput, warmupRequests, totalMeasuredTokens });
  }

  function emptyPlanResult(policy, capacity, plan) {
    return {
      policy,
      cacheBlocks: capacity,
      warmupRequests: plan.warmupRequests,
      hitTokens: 0,
      totalTokens: plan.totalMeasuredTokens,
      hitRate: 0,
      usefulCacheBlockSamples: 0,
      usefulCacheSamples: 0,
      usefulCacheRate: 0,
    };
  }

  function finishPlanResult(policy, capacity, plan, hitTokens, usefulCacheBlockSamples, usefulCacheSamples) {
    return {
      policy,
      cacheBlocks: capacity,
      warmupRequests: plan.warmupRequests,
      hitTokens,
      totalTokens: plan.totalMeasuredTokens,
      hitRate: plan.totalMeasuredTokens ? hitTokens / plan.totalMeasuredTokens : 0,
      usefulCacheBlockSamples,
      usefulCacheSamples,
      usefulCacheRate: capacity > 0 && usefulCacheSamples > 0
        ? usefulCacheBlockSamples / (usefulCacheSamples * capacity)
        : 0,
    };
  }

  function cloneCeilingResult(ceiling, policy, cacheBlocks) {
    const usefulCacheBlockSamples = toNumber(ceiling && ceiling.usefulCacheBlockSamples, 0);
    const usefulCacheSamples = toNumber(ceiling && ceiling.usefulCacheSamples, 0);
    return {
      policy,
      cacheBlocks,
      warmupRequests: ceiling.warmupRequests,
      hitTokens: ceiling.hitTokens,
      totalTokens: ceiling.totalTokens,
      hitRate: ceiling.hitRate,
      usefulCacheBlockSamples,
      usefulCacheSamples,
      usefulCacheRate: cacheBlocks > 0 && usefulCacheSamples > 0
        ? usefulCacheBlockSamples / (usefulCacheSamples * cacheBlocks)
        : 0,
    };
  }

  function isMeasuredRequestEnd(plan, eventIndex) {
    const requestIndex = plan.eventRequest[eventIndex];
    if (requestIndex < plan.warmupRequests) return false;
    return eventIndex === plan.eventIds.length - 1 || plan.eventRequest[eventIndex + 1] !== requestIndex;
  }

  function throughputFromHitRate(hitRate) {
    const missFraction = Math.max(1 - clamp(toNumber(hitRate, 0), 0, 1), THROUGHPUT_EPSILON);
    return Math.min(THROUGHPUT_DISPLAY_CAP, 1 / missFraction);
  }

  // Belady/optimal with bypass over interned ids. Block membership and the
  // per-block {nextUse, version} ride on typed arrays (no Map), and the eviction
  // heap is a struct-of-arrays binary max-heap (no per-push object). The heap
  // pushes, pops, comparisons and lazy-versioned staleness checks are the exact
  // same operations the object version ran, so the pop order — and therefore
  // every hit count — is identical; only the allocation churn is gone.
  //
  // Bypass: when the cache is full, only admit the incoming block if it will be
  // reused strictly sooner than the farthest-future cached block; otherwise leave
  // the cache untouched (no point caching a block needed later than one we hold).
  function simulateOptimalPlan(plan, capacity) {
    if (!plan.eventIds.length || capacity <= 0) return emptyPlanResult("optimal", capacity, plan);
    const eventIds = plan.eventIds;
    const eventTokens = plan.eventTokens;
    const eventIsInput = plan.eventIsInput;
    const eventRequest = plan.eventRequest;
    const nextInput = plan.nextInput;
    const warmupRequests = plan.warmupRequests;
    const n = plan.uniqueBlocks;

    const present = new Uint8Array(n);
    const stateNextUse = new Float64Array(n);
    const stateVersion = new Int32Array(n);
    let cacheSize = 0;
    const never = eventIds.length + 1;
    let usefulCount = 0;
    let usefulCacheBlockSamples = 0;
    let usefulCacheSamples = 0;

    function sampleUseful(index) {
      if (isMeasuredRequestEnd(plan, index)) {
        usefulCacheBlockSamples += usefulCount;
        usefulCacheSamples += 1;
      }
    }

    function addUseful(nextUse) {
      if (nextUse < never) usefulCount += 1;
    }

    function removeUseful(nextUse) {
      if (nextUse < never) usefulCount -= 1;
    }

    let heapCap = 1 << 12;
    let heapId = new Int32Array(heapCap);
    let heapNextUse = new Float64Array(heapCap);
    let heapVersion = new Int32Array(heapCap);
    let heapLen = 0;

    function heapPush(id, nextUse, version) {
      if (heapLen >= heapCap) {
        const cap = heapCap * 2;
        const grownId = new Int32Array(cap); grownId.set(heapId); heapId = grownId;
        const grownNext = new Float64Array(cap); grownNext.set(heapNextUse); heapNextUse = grownNext;
        const grownVersion = new Int32Array(cap); grownVersion.set(heapVersion); heapVersion = grownVersion;
        heapCap = cap;
      }
      let index = heapLen;
      heapId[index] = id;
      heapNextUse[index] = nextUse;
      heapVersion[index] = version;
      heapLen += 1;
      while (index > 0) {
        const parent = (index - 1) >> 1;
        if (heapNextUse[parent] >= heapNextUse[index]) break;
        const ti = heapId[parent]; heapId[parent] = heapId[index]; heapId[index] = ti;
        const tn = heapNextUse[parent]; heapNextUse[parent] = heapNextUse[index]; heapNextUse[index] = tn;
        const tv = heapVersion[parent]; heapVersion[parent] = heapVersion[index]; heapVersion[index] = tv;
        index = parent;
      }
    }

    // Pop writes the former root into these before re-heapifying.
    let topId = 0;
    let topNextUse = 0;
    let topVersion = 0;
    function heapPop() {
      topId = heapId[0];
      topNextUse = heapNextUse[0];
      topVersion = heapVersion[0];
      heapLen -= 1;
      if (heapLen > 0) {
        heapId[0] = heapId[heapLen];
        heapNextUse[0] = heapNextUse[heapLen];
        heapVersion[0] = heapVersion[heapLen];
        let index = 0;
        for (;;) {
          const left = index * 2 + 1;
          const right = left + 1;
          let largest = index;
          if (left < heapLen && heapNextUse[left] > heapNextUse[largest]) largest = left;
          if (right < heapLen && heapNextUse[right] > heapNextUse[largest]) largest = right;
          if (largest === index) break;
          const ti = heapId[largest]; heapId[largest] = heapId[index]; heapId[index] = ti;
          const tn = heapNextUse[largest]; heapNextUse[largest] = heapNextUse[index]; heapNextUse[index] = tn;
          const tv = heapVersion[largest]; heapVersion[largest] = heapVersion[index]; heapVersion[index] = tv;
          index = largest;
        }
      }
    }

    function pushState(id, nextUse) {
      const version = present[id] ? stateVersion[id] + 1 : 1;
      if (!present[id]) {
        present[id] = 1;
        cacheSize += 1;
        addUseful(nextUse);
      } else {
        removeUseful(stateNextUse[id]);
        addUseful(nextUse);
      }
      stateNextUse[id] = nextUse;
      stateVersion[id] = version;
      heapPush(id, nextUse, version);
    }

    function evictForCandidate(candidateNext) {
      for (;;) {
        if (heapLen === 0) return true;
        heapPop();
        if (!present[topId] || stateVersion[topId] !== topVersion || stateNextUse[topId] !== topNextUse) continue;
        if (topNextUse > candidateNext) {
          removeUseful(stateNextUse[topId]);
          present[topId] = 0;
          cacheSize -= 1;
          return true;
        }
        heapPush(topId, topNextUse, topVersion);
        return false;
      }
    }

    let hitTokens = 0;
    for (let index = 0; index < eventIds.length; index += 1) {
      const id = eventIds[index];
      const hit = present[id] === 1;
      if (hit && eventIsInput[index] && eventRequest[index] >= warmupRequests) hitTokens += eventTokens[index];
      if (hit) {
        pushState(id, nextInput[index]);
      } else if (cacheSize < capacity) {
        pushState(id, nextInput[index]);
      } else if (evictForCandidate(nextInput[index])) {
        pushState(id, nextInput[index]);
      }
      sampleUseful(index);
    }

    return finishPlanResult("optimal", capacity, plan, hitTokens, usefulCacheBlockSamples, usefulCacheSamples);
  }

  function simulatePlanPolicy(plan, cacheBlocks, policy) {
    const normalizedPolicy = policy || "lru";
    const capacity = Math.max(0, Math.floor(toNumber(cacheBlocks, 0)));
    if (!plan.eventIds.length || capacity <= 0) return emptyPlanResult(normalizedPolicy, capacity, plan);
    if (normalizedPolicy === "optimal") {
      const result = simulateOptimalPlan(plan, capacity);
      result.policy = normalizedPolicy;
      return result;
    }

    // Block ids are interned to 0..uniqueBlocks-1, so membership and LRU order
    // ride on typed arrays — no Map churn (which V8 compacts on repeated
    // delete+set) and no per-eviction iterator allocation.
    const n = plan.uniqueBlocks;
    const inCache = new Uint8Array(n);
    const stateNextUse = new Int32Array(n).fill(plan.eventIds.length + 1);
    const never = plan.eventIds.length + 1;
    let size = 0;
    let hitTokens = 0;
    let usefulCount = 0;
    let usefulCacheBlockSamples = 0;
    let usefulCacheSamples = 0;

    function setCached(id, nextUse) {
      if (inCache[id] === 1 && stateNextUse[id] < never) usefulCount -= 1;
      if (nextUse < never) usefulCount += 1;
      stateNextUse[id] = nextUse;
    }

    function evictCached(id) {
      if (inCache[id] === 1 && stateNextUse[id] < never) usefulCount -= 1;
      inCache[id] = 0;
      stateNextUse[id] = never;
      size -= 1;
    }

    function sampleUseful(index) {
      if (isMeasuredRequestEnd(plan, index)) {
        usefulCacheBlockSamples += usefulCount;
        usefulCacheSamples += 1;
      }
    }

    if (normalizedPolicy === "fifo") {
      const queue = new Int32Array(plan.eventIds.length);
      let qHead = 0;
      let qTail = 0;
      for (let index = 0; index < plan.eventIds.length; index += 1) {
        const id = plan.eventIds[index];
        const measured = plan.eventIsInput[index] && plan.eventRequest[index] >= plan.warmupRequests;
        const hit = inCache[id] === 1;
        if (measured && hit) hitTokens += plan.eventTokens[index];
        if (!hit) {
          if (size >= capacity) {
            while (qHead < qTail) {
              const victim = queue[qHead];
              qHead += 1;
              if (inCache[victim] === 1) {
                evictCached(victim);
                break;
              }
            }
          }
          if (size < capacity) {
            inCache[id] = 1;
            size += 1;
            setCached(id, plan.nextInput[index]);
            queue[qTail] = id;
            qTail += 1;
          }
        } else {
          setCached(id, plan.nextInput[index]);
        }
        sampleUseful(index);
      }
    } else {
      // LRU as a doubly linked list over integer ids (head = oldest, tail = newest).
      const prev = new Int32Array(n).fill(-1);
      const next = new Int32Array(n).fill(-1);
      let lruHead = -1;
      let lruTail = -1;
      for (let index = 0; index < plan.eventIds.length; index += 1) {
        const id = plan.eventIds[index];
        const measured = plan.eventIsInput[index] && plan.eventRequest[index] >= plan.warmupRequests;
        const hit = inCache[id] === 1;
        if (measured && hit) hitTokens += plan.eventTokens[index];
        if (hit) {
          setCached(id, plan.nextInput[index]);
          if (id !== lruTail) {
            const p = prev[id];
            const nx = next[id];
            if (p !== -1) next[p] = nx;
            else lruHead = nx;
            if (nx !== -1) prev[nx] = p;
            prev[id] = lruTail;
            next[id] = -1;
            if (lruTail !== -1) next[lruTail] = id;
            lruTail = id;
          }
        } else {
          if (size >= capacity) {
            const victim = lruHead;
            const nx = next[victim];
            lruHead = nx;
            if (nx !== -1) prev[nx] = -1;
            else lruTail = -1;
            prev[victim] = -1;
            next[victim] = -1;
            evictCached(victim);
          }
          inCache[id] = 1;
          size += 1;
          setCached(id, plan.nextInput[index]);
          prev[id] = lruTail;
          next[id] = -1;
          if (lruTail !== -1) next[lruTail] = id;
          lruTail = id;
          if (lruHead === -1) lruHead = id;
        }
        sampleUseful(index);
      }
    }

    return finishPlanResult(normalizedPolicy, capacity, plan, hitTokens, usefulCacheBlockSamples, usefulCacheSamples);
  }

  function simulatePolicy(trace, cacheBlocks, policy, options) {
    const plan = buildExecutionPlan(trace, options || {});
    return simulatePlanPolicy(plan, cacheBlocks, policy || "lru");
  }

  function estimateBytesPerToken(model, settings) {
    const estimateTokens = toPositiveInteger(settings && settings.estimateTokens, model.default_tokens || 4096);
    const result = calculator.calculate(
      model,
      {
        tokens: estimateTokens,
        sequences: 1,
        precision: settings && settings.precision,
        indexerPrecision: settings && settings.indexerPrecision,
        includeDraftKvCache: settings && settings.includeDraftKvCache,
        includeLinearAttentionState: false,
      },
      {
        precisionOptions: settings && settings.precisionOptions,
        indexerPrecisionOptions: settings && settings.indexerPrecisionOptions,
      },
    );
    return result.bytesPerToken;
  }

  function traceEstimateTokens(trace, model) {
    if (trace && trace.summary && Number.isFinite(Number(trace.summary.averageInputTokens))) {
      return toPositiveInteger(trace.summary.averageInputTokens, model.default_tokens || 4096);
    }
    const requests = trace && Array.isArray(trace.requests) ? trace.requests : [];
    if (requests.length) {
      const total = requests.reduce((sum, request) => {
        if (Number.isFinite(Number(request.inputTokens))) return sum + Number(request.inputTokens);
        const blocks = Array.isArray(request.inputBlocks) ? request.inputBlocks : [];
        return sum + blocks.reduce((blockSum, block) => blockSum + toNumber(block.tokens, 0), 0);
      }, 0);
      return toPositiveInteger(total / requests.length, model.default_tokens || 4096);
    }
    return toPositiveInteger(model.default_tokens || 4096, 4096);
  }

  function cacheBlocksForGiB(gib, bytesPerBlock) {
    if (!Number.isFinite(bytesPerBlock) || bytesPerBlock <= 0) return 0;
    return Math.max(0, Math.floor((gib * BYTES_PER_GIB) / bytesPerBlock));
  }

  function capacityValues(settings) {
    if (settings && Array.isArray(settings.capacityGiBValues) && settings.capacityGiBValues.length) {
      return settings.capacityGiBValues.map((value) => Math.max(0, toNumber(value, 0)));
    }
    const minGiB = Math.max(0, toNumber(settings && settings.minGiB, 1));
    const maxGiB = Math.max(minGiB, toNumber(settings && settings.maxGiB, 64));
    const steps = clamp(toPositiveInteger(settings && settings.steps, 24), 2, 80);
    if (settings && (Object.prototype.hasOwnProperty.call(settings, "minGiB") || Object.prototype.hasOwnProperty.call(settings, "maxGiB") || Object.prototype.hasOwnProperty.call(settings, "steps"))) {
      return Array.from({ length: steps }, (_, index) => minGiB + ((maxGiB - minGiB) * index) / (steps - 1));
    }
    return DEFAULT_CAPACITY_GIB_VALUES.slice();
  }

  function sweepCapacity(trace, model, settings, onProgress) {
    const blockSize = toPositiveInteger(settings && settings.blockSize, trace && trace.blockSize ? trace.blockSize : DEFAULT_BLOCK_SIZE);
    const accountingSettings = Object.assign({}, settings || {}, { estimateTokens: traceEstimateTokens(trace, model) });
    const bytesPerToken = estimateBytesPerToken(model, accountingSettings);
    const bytesPerBlock = bytesPerToken * blockSize;
    const policies = (settings && settings.policies) || POLICIES;
    const warmupFraction = settings && settings.warmupFraction;
    // Build the flattened event stream + next-use plan once, then reuse it for
    // every capacity point and policy. Capacities that can hold the whole
    // working set never evict, so hit tokens equal the infinite-cache ceiling.
    // Useful occupancy still needs the full-working-set sample sum once; larger
    // budgets then only change the denominator.
    const plan = buildExecutionPlan(trace, { warmupFraction });
    const uniqueBlocks = plan.uniqueBlocks;
    // The infinite-cache result equals any policy with capacity >= uniqueBlocks
    // (no evictions ever happen), so compute it once with a cheap LRU pass.
    let ceilingResult = null;
    function ceilingFor(cacheBlocks) {
      if (!ceilingResult) ceilingResult = simulatePlanPolicy(plan, Math.max(uniqueBlocks, 1), "lru");
      return cloneCeilingResult(ceilingResult, "lru", cacheBlocks);
    }
    const capValues = capacityValues(settings || {});
    const totalPoints = capValues.length;
    const points = capValues.map((gib, pointIndex) => {
      const cacheBlocks = cacheBlocksForGiB(gib, bytesPerBlock);
      const results = {};
      policies.forEach((policy) => {
        if (uniqueBlocks > 0 && cacheBlocks >= uniqueBlocks) {
          results[policy] = cloneCeilingResult(ceilingFor(cacheBlocks), policy, cacheBlocks);
        } else {
          results[policy] = simulatePlanPolicy(plan, cacheBlocks, policy);
        }
      });
      if (typeof onProgress === "function") onProgress(pointIndex + 1, totalPoints);
      return { gib, cacheBlocks, results };
    });
    const sweep = { blockSize, bytesPerToken, bytesPerBlock, points, policies };
    if (settings && settings.computeCeiling) {
      const c = ceilingFor(0);
      sweep.reuseCeiling = c.hitRate;
      sweep.warmupRequests = c.warmupRequests;
    }
    return sweep;
  }

  // --- Parallel sweep decomposition ------------------------------------------
  // sweepCapacity does three separable things: (1) parse+plan the trace, which
  // is model-independent; (2) simulate each (policy, cacheBlocks), which is also
  // model-independent — the model only picks which cacheBlocks each GiB maps to;
  // (3) assemble the curve, which is cheap. Splitting them lets a worker pool run
  // the (policy, cacheBlocks) sims in parallel and lets the main thread cache the
  // plan + per-(policy, cacheBlocks) results so changing model/precision reuses
  // everything except the cheap GiB→cacheBlocks remap. Each piece calls the exact
  // same simulatePlanPolicy / buildExecutionPlan as sweepCapacity, so the curve
  // is byte-identical to the single-threaded path.

  function extractPlanBuffers(plan) {
    return {
      buffers: {
        eventIds: plan.eventIds,
        eventTokens: plan.eventTokens,
        eventRequest: plan.eventRequest,
        eventIsInput: plan.eventIsInput,
        nextInput: plan.nextInput,
      },
      scalars: {
        requestCount: plan.requestCount,
        uniqueBlocks: plan.uniqueBlocks,
        warmupRequests: plan.warmupRequests,
        totalMeasuredTokens: plan.totalMeasuredTokens,
      },
    };
  }

  function planFromBuffers(payload) {
    return Object.assign({}, payload.buffers, payload.scalars);
  }

  // Model-independent analysis of a trace: the execution plan (as transferable
  // typed-array buffers), the infinite-cache ceiling (one LRU pass at full
  // capacity, exactly as sweepCapacity computes it), the temporal stats, and a
  // slim trace for the UI. Cached per trace so model/precision changes skip it.
  function analyzeTrace(trace, options) {
    const opts = options || {};
    const plan = buildExecutionPlan(trace, { warmupFraction: opts.warmupFraction });
    const ceilingRaw = simulatePlanPolicy(plan, Math.max(plan.uniqueBlocks, 1), "lru");
    const ceiling = {
      hitTokens: ceilingRaw.hitTokens,
      totalTokens: ceilingRaw.totalTokens,
      hitRate: ceilingRaw.hitRate,
      warmupRequests: ceilingRaw.warmupRequests,
      usefulCacheBlockSamples: ceilingRaw.usefulCacheBlockSamples,
      usefulCacheSamples: ceilingRaw.usefulCacheSamples,
      usefulCacheRate: ceilingRaw.usefulCacheRate,
    };
    let timeStats = null;
    try {
      timeStats = computeTimeSeries(trace);
    } catch (error) {
      timeStats = null;
    }
    const summary = trace.summary || {};
    const meta = {
      requestCount: plan.requestCount,
      eventCount: plan.eventIds.length,
      uniqueBlocks: plan.uniqueBlocks,
      warmupRequests: plan.warmupRequests,
      totalMeasuredTokens: plan.totalMeasuredTokens,
      blockSize: trace.blockSize,
      summary,
      slimTrace: {
        presetId: trace.presetId,
        presetLabel: trace.presetLabel,
        sourceKind: trace.sourceKind,
        blockSize: trace.blockSize,
        sourceBlockSizeNote: trace.sourceBlockSizeNote,
        summary,
        requestCount: Array.isArray(trace.requests) ? trace.requests.length : summary.requests || plan.requestCount,
      },
    };
    return { plan, planBuffers: extractPlanBuffers(plan), meta, ceiling, timeStats };
  }

  function estimateTokensFromMeta(meta, model) {
    const summary = (meta && meta.summary) || {};
    if (Number.isFinite(Number(summary.averageInputTokens))) {
      return toPositiveInteger(summary.averageInputTokens, model.default_tokens || 4096);
    }
    return toPositiveInteger(model.default_tokens || 4096, 4096);
  }

  // The model/precision-dependent step: map every GiB point to a cache-block
  // capacity, then list the distinct (policy, cacheBlocks) sims actually needed
  // (skipping cacheBlocks<=0 and cacheBlocks>=uniqueBlocks, which the ceiling
  // covers). Mirrors sweepCapacity's bytesPerBlock + capacity math exactly.
  function planSweepTasks(meta, model, settings) {
    const opts = settings || {};
    const blockSize = toPositiveInteger(opts.blockSize, meta.blockSize || DEFAULT_BLOCK_SIZE);
    const estimateTokens = estimateTokensFromMeta(meta, model);
    const bytesPerToken = estimateBytesPerToken(model, Object.assign({}, opts, { estimateTokens }));
    const bytesPerBlock = bytesPerToken * blockSize;
    const policies = opts.policies || POLICIES;
    const capValues = capacityValues(opts);
    const uniqueBlocks = meta.uniqueBlocks;
    const points = capValues.map((gib) => ({ gib, cacheBlocks: cacheBlocksForGiB(gib, bytesPerBlock) }));
    const seen = new Set();
    const tasks = [];
    points.forEach(({ cacheBlocks }) => {
      if (cacheBlocks <= 0) return;
      if (uniqueBlocks > 0 && cacheBlocks >= uniqueBlocks) return;
      policies.forEach((policy) => {
        const key = `${policy}|${cacheBlocks}`;
        if (seen.has(key)) return;
        seen.add(key);
        tasks.push({ policy, cacheBlocks });
      });
    });
    return { blockSize, bytesPerToken, bytesPerBlock, policies, points, tasks };
  }

  // Build the final sweep object from a planSweepTasks() result, the per-policy
  // ceiling, and a lookup into already-computed sim results. Field-for-field
  // identical to sweepCapacity's output.
  function assembleSweep(planned, meta, settings, simLookup, ceiling) {
    const uniqueBlocks = meta.uniqueBlocks;
    const points = planned.points.map(({ gib, cacheBlocks }) => {
      const results = {};
      planned.policies.forEach((policy) => {
        if (uniqueBlocks > 0 && cacheBlocks >= uniqueBlocks) {
          results[policy] = cloneCeilingResult(ceiling, policy, cacheBlocks);
        } else if (cacheBlocks <= 0) {
          results[policy] = {
            policy,
            cacheBlocks,
            warmupRequests: meta.warmupRequests,
            hitTokens: 0,
            totalTokens: meta.totalMeasuredTokens,
            hitRate: 0,
            usefulCacheBlockSamples: 0,
            usefulCacheSamples: 0,
            usefulCacheRate: 0,
          };
        } else {
          results[policy] = simLookup(policy, cacheBlocks);
        }
      });
      return { gib, cacheBlocks, results };
    });
    const sweep = {
      blockSize: planned.blockSize,
      bytesPerToken: planned.bytesPerToken,
      bytesPerBlock: planned.bytesPerBlock,
      points,
      policies: planned.policies,
    };
    if (settings && settings.computeCeiling) {
      sweep.reuseCeiling = ceiling.hitRate;
      sweep.warmupRequests = ceiling.warmupRequests;
    }
    return sweep;
  }

  // Worker-pool engine. One pool of persistent workers per lab mount. A job:
  //   1. analyze the trace once (cached per trace key) -> plan buffers + ceiling,
  //   2. fan the not-yet-cached (policy, cacheBlocks) sims across the pool,
  //   3. assemble the curve on the main thread.
  // Jobs are serialized through `tail` so a superseding job waits for the prior
  // one to drain (cancellation makes that wait ~one task), which keeps each
  // worker's single in-flight request unambiguous. The plan is copied to a worker
  // only the first time that worker sees a given trace key.
  function createLabEngine(options) {
    const opts = options || {};
    const createWorker = opts.createWorker || function () { return new Worker(opts.workerUrl); };
    const navConcurrency =
      typeof navigator !== "undefined" && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 4;
    // Default to half the logical cores (capped at 32) — leaves headroom for the
    // main thread / other tabs while using real parallelism on big machines.
    const poolSize = Math.max(1, Math.min(32, opts.poolSize || Math.floor(navConcurrency / 2) || 1));
    const urls = { calculatorScriptUrl: opts.calculatorUrl, labScriptUrl: opts.labUrl };
    const analysisCache = new Map();
    const simMemo = new Map();
    const MAX_TRACES = 4;
    let pool = null;
    let tail = Promise.resolve();
    // Set for the duration of a job so worker "progress" messages (parse phase)
    // can be forwarded to the current job's onProgress.
    let activeOnProgress = null;

    function ensurePool() {
      if (pool) return pool;
      pool = [];
      for (let i = 0; i < poolSize; i += 1) {
        const entry = { worker: createWorker(), busy: false, traceKey: null, resolve: null, reject: null };
        entry.worker.onmessage = function (event) {
          const data = (event && event.data) || {};
          if (data.type === "progress") {
            if (activeOnProgress) activeOnProgress(data);
            return;
          }
          const resolve = entry.resolve;
          const reject = entry.reject;
          entry.resolve = null;
          entry.reject = null;
          entry.busy = false;
          if (data.type === "error") {
            if (reject) reject(new Error(data.error || "worker error"));
          } else if (resolve) {
            resolve(data);
          }
        };
        entry.worker.onerror = function (event) {
          const reject = entry.reject;
          entry.resolve = null;
          entry.reject = null;
          entry.busy = false;
          entry.traceKey = null;
          if (reject) reject(new Error((event && event.message) || "worker error"));
        };
        pool.push(entry);
      }
      return pool;
    }

    function request(entry, message) {
      return new Promise(function (resolve, reject) {
        entry.busy = true;
        entry.resolve = resolve;
        entry.reject = reject;
        entry.worker.postMessage(message);
      });
    }

    function rememberAnalysis(key, analysis) {
      if (analysisCache.has(key)) analysisCache.delete(key);
      analysisCache.set(key, analysis);
      if (!simMemo.has(key)) simMemo.set(key, new Map());
      while (analysisCache.size > MAX_TRACES) {
        const oldest = analysisCache.keys().next().value;
        analysisCache.delete(oldest);
        simMemo.delete(oldest);
      }
    }

    function traceKeyFor(input) {
      const settings = input.settings || {};
      const wf = settings.warmupFraction;
      const cap = (input.uploadOptions && input.uploadOptions.maxEvents) || "";
      if (input.uploadFile) {
        const file = input.uploadFile;
        const uploadOptions = input.uploadOptions || {};
        return `upf|${file.name}|${file.size}|${file.lastModified}|bs=${uploadOptions.blockSize || ""}|cap=${cap}|wf=${wf}`;
      }
      if (input.uploadText != null) {
        const uploadOptions = input.uploadOptions || {};
        return `up|${uploadOptions.label || ""}|${input.uploadText.length}|bs=${uploadOptions.blockSize || ""}|cap=${cap}|wf=${wf}`;
      }
      return `gen|${input.preset && input.preset.id}|${createCacheKey(input.params || {})}|seed=${input.seed || DEFAULT_SEED}|wf=${wf}`;
    }

    function run(input, onProgress) {
      let cancelled = false;
      const previous = tail;
      const job = (async function () {
        await previous.catch(function () {});
        if (cancelled) throw new Error("cancelled");
        ensurePool();
        const key = traceKeyFor(input);

        // WASM path: one worker streams the whole (gzip) file into the Rust
        // trace processor's linear memory and runs every sweep there, so a
        // multi-GB trace loads fully (no truncation). The plan lives in that
        // worker's memory, so there is no pool fan-out; model/precision changes
        // reuse the instance and re-sweep. Failure (e.g. >4 GB) throws and
        // startJob falls back to the JS path.
        if (input.useWasm && input.wasmUrl) {
          activeOnProgress = (data) => {
            if (typeof onProgress !== "function" || !data) return;
            if (data.phase === "parse") onProgress(data.bytes || 0, data.total || 1);
            else if (data.phase === "sweep") onProgress(data.completed || 0, data.total || 1);
          };
          try {
            const resp = await request(
              pool[0],
              Object.assign(
                {
                  type: "runWasm",
                  traceKey: key,
                  jobId: input.jobId,
                  cacheKey: input.cacheKey,
                  preset: input.preset,
                  uploadFile: input.uploadFile,
                  gzip: input.gzip,
                  fileSize: input.uploadFile ? input.uploadFile.size : 0,
                  blockSizeOverride: input.uploadOptions && input.uploadOptions.blockSize,
                  maxEvents: 0, // full load in wasm; JS fallback keeps the cap via uploadOptions
                  warmupFraction: (input.settings || {}).warmupFraction,
                  model: input.model,
                  settings: input.settings,
                  uploadOptions: input.uploadOptions,
                  wasmUrl: input.wasmUrl,
                },
                urls,
              ),
            );
            pool[0].traceKey = key;
            if (cancelled) throw new Error("cancelled");
            return resp.result;
          } finally {
            activeOnProgress = null;
          }
        }

        let analysis = analysisCache.get(key);
        if (!analysis) {
          activeOnProgress = (data) => {
            if (data && data.phase === "parse" && typeof onProgress === "function") {
              onProgress(data.bytes || 0, data.total || 1);
            }
          };
          let resp;
          try {
            resp = await request(
              pool[0],
              Object.assign(
                {
                  type: "analyze",
                  traceKey: key,
                  uploadText: input.uploadText,
                  uploadFile: input.uploadFile,
                  gzip: input.gzip,
                  fileSize: input.uploadFile ? input.uploadFile.size : undefined,
                  maxEvents: input.uploadOptions && input.uploadOptions.maxEvents,
                  uploadOptions: input.uploadOptions,
                  preset: input.preset,
                  params: input.params,
                  seed: input.seed,
                  warmupFraction: (input.settings || {}).warmupFraction,
                },
                urls,
              ),
            );
          } finally {
            activeOnProgress = null;
          }
          pool[0].traceKey = key;
          analysis = { planBuffers: resp.planBuffers, meta: resp.meta, ceiling: resp.ceiling, timeStats: resp.timeStats };
          rememberAnalysis(key, analysis);
        }
        if (cancelled) throw new Error("cancelled");
        const planned = planSweepTasks(analysis.meta, input.model, input.settings || {});
        let memo = simMemo.get(key);
        if (!memo) {
          memo = new Map();
          simMemo.set(key, memo);
        }
        const todo = planned.tasks.filter((task) => !memo.has(`${task.policy}|${task.cacheBlocks}`));
        const total = planned.tasks.length;
        let completed = total - todo.length;
        if (typeof onProgress === "function") onProgress(completed, Math.max(total, 1));
        // Fan out across as many workers as a plan-copy memory budget allows.
        // pool[0] already holds the plan (from analyze, no copy); each extra
        // worker costs ~17 B/event, so a huge plan collapses toward one worker
        // while a small plan uses the whole pool for the speedup.
        const planCopyBytes = analysis.meta.eventCount * PLAN_BYTES_PER_EVENT;
        const maxWorkers = Math.max(
          1,
          Math.min(pool.length, Math.floor(PLAN_COPY_BUDGET_BYTES / Math.max(planCopyBytes, 1)) + 1),
        );
        const simWorkers = pool.slice(0, maxWorkers);
        let next = 0;
        let aborted = false;
        async function workerLoop(entry) {
          for (;;) {
            if (cancelled || aborted) return;
            const taskIndex = next;
            next += 1;
            if (taskIndex >= todo.length) return;
            const task = todo[taskIndex];
            const message = Object.assign(
              { type: "simulate", traceKey: key, policy: task.policy, cacheBlocks: task.cacheBlocks },
              urls,
            );
            if (entry.traceKey !== key) {
              message.plan = analysis.planBuffers;
              entry.traceKey = key;
            }
            let resp;
            try {
              resp = await request(entry, message);
            } catch (error) {
              aborted = true;
              throw error;
            }
            if (!cancelled && resp && resp.result) {
              memo.set(`${resp.result.policy}|${resp.result.cacheBlocks}`, resp.result);
              completed += 1;
              if (typeof onProgress === "function") onProgress(completed, Math.max(total, 1));
            }
          }
        }
        await Promise.all(simWorkers.map(workerLoop));
        if (cancelled) throw new Error("cancelled");
        const sweep = assembleSweep(
          planned,
          analysis.meta,
          input.settings || {},
          (policy, cacheBlocks) => memo.get(`${policy}|${cacheBlocks}`),
          analysis.ceiling,
        );
        return {
          jobId: input.jobId,
          cacheKey: input.cacheKey,
          preset: input.preset || { id: analysis.meta.slimTrace.presetId, label: analysis.meta.slimTrace.presetLabel },
          trace: analysis.meta.slimTrace,
          sweep,
          timeStats: analysis.timeStats,
        };
      })();
      tail = job.catch(function () {});
      return {
        promise: job,
        cancel() {
          cancelled = true;
        },
      };
    }

    return { run };
  }

  function stableValue(value) {
    if (Array.isArray(value)) return value.map(stableValue);
    if (value && typeof value === "object") {
      const result = {};
      Object.keys(value)
        .sort()
        .forEach((key) => {
          result[key] = stableValue(value[key]);
        });
      return result;
    }
    return value;
  }

  function createCacheKey(input) {
    return JSON.stringify(stableValue(input));
  }

  function shouldApplyJobResult(latestJobId, result) {
    return !!result && Number(result.jobId) === Number(latestJobId);
  }

  function modelSweepKey(model, settings) {
    return [
      model && model.id,
      `precision=${settings && settings.precision ? settings.precision : ""}`,
      `indexer=${settings && settings.indexerPrecision ? settings.indexerPrecision : ""}`,
      `draft=${settings && settings.includeDraftKvCache ? "1" : "0"}`,
    ].join("|");
  }

  function precomputedTrace(precomputed, preset) {
    if (!precomputed || !precomputed.traces || !preset) return null;
    return precomputed.traces[preset.id] || null;
  }

  function parseSweepKey(key) {
    const parts = String(key || "").split("|");
    const setting = { modelId: parts[0] || "" };
    parts.slice(1).forEach((part) => {
      const equalIndex = part.indexOf("=");
      if (equalIndex < 0) return;
      const name = part.slice(0, equalIndex);
      const value = part.slice(equalIndex + 1);
      if (name === "precision") setting.precision = value || undefined;
      if (name === "indexer") setting.indexerPrecision = value || undefined;
      if (name === "draft") setting.includeDraftKvCache = value === "1";
    });
    return setting;
  }

  function availableSettingsFor(precomputed, preset, model) {
    const trace = precomputedTrace(precomputed, preset);
    if (!trace || !trace.modelSweeps || !model) return [];
    return Object.keys(trace.modelSweeps)
      .map(parseSweepKey)
      .filter((setting) => setting.modelId === model.id);
  }

  function availableModelIdsFor(precomputed, preset) {
    const trace = precomputedTrace(precomputed, preset);
    if (!trace || !trace.modelSweeps) return null;
    return new Set(Object.keys(trace.modelSweeps).map((key) => parseSweepKey(key).modelId));
  }

  function precomputedResultFor(precomputed, preset, model, settings) {
    const trace = precomputedTrace(precomputed, preset);
    if (!trace || !trace.modelSweeps) return null;
    const key = modelSweepKey(model, settings || {});
    const sweep = trace.modelSweeps[key];
    if (!sweep) return null;
    return {
      preset,
      trace: {
        presetId: preset.id,
        presetLabel: preset.label,
        blockSize: trace.nativeBlockSize || sweep.blockSize,
        sourceKind: trace.sourceKind,
        sourceBlockSizeNote: trace.sourceBlockSizeNote,
        requests: null,
        summary: trace.summary || {},
      },
      sweep: Object.assign({}, sweep, {
        precomputed: true,
        reuseCeiling:
          Number.isFinite(Number(sweep.reuseCeiling))
            ? Number(sweep.reuseCeiling)
            : trace.summary && Number.isFinite(Number(trace.summary.infiniteHitRate))
              ? Number(trace.summary.infiniteHitRate)
              : undefined,
      }),
    };
  }

  function runLabComputation(input) {
    const trace = generateTrace(input.preset, input.params, input.seed || DEFAULT_SEED);
    const sweep = sweepCapacity(trace, input.model, input.settings || {});
    return {
      jobId: input.jobId,
      cacheKey: input.cacheKey,
      preset: input.preset,
      trace,
      sweep,
    };
  }

  function rememberCachedResult(cache, key, result) {
    if (cache.has(key)) cache.delete(key);
    cache.set(key, result);
    while (cache.size > MAX_CACHE_ENTRIES) cache.delete(cache.keys().next().value);
  }

  function modelFamily(model) {
    return calculator.modelFamily ? calculator.modelFamily(model) : model.family || "Other";
  }

  function sortedModelFamilies(models) {
    return Array.from(new Set(models.map(modelFamily))).sort();
  }

  function modelsForFamily(models, family) {
    return models.filter((model) => modelFamily(model) === family);
  }

  function modelById(models, id) {
    return models.find((model) => model.id === id) || models[0];
  }

  function modelFieldNumber(model, name, fallback) {
    return toNumber(model && model.fields ? model.fields[name] : undefined, fallback);
  }

  function isDeepSeekV4(model) {
    return Boolean(model && model.formula === "deepseek_v4_hybrid");
  }

  function hasIndexerCache(model) {
    return Number.isFinite(modelFieldNumber(model, "index_head_dim", NaN));
  }

  function draftLayerCount(model) {
    const nextnLayers = modelFieldNumber(model, "num_nextn_predict_layers", 0);
    if (nextnLayers > 0) return nextnLayers;
    if (model && model.fields && model.fields.use_mtp === true) {
      return modelFieldNumber(model, "num_mtp_modules", 0) * modelFieldNumber(model, "mtp_transformer_layers", 0);
    }
    return 0;
  }

  function hasDraftKvCache(model) {
    if (!model || !model.fields) return false;
    if (isDeepSeekV4(model)) {
      const layers = modelFieldNumber(model, "num_hidden_layers", 0);
      return Array.isArray(model.fields.compress_ratios) && model.fields.compress_ratios.length > layers;
    }
    return draftLayerCount(model) > 0;
  }

  function optionIds(options) {
    return (options || []).map((option) => option.id || option);
  }

  function defaultPrecisionId(model, options) {
    const ids = optionIds(options);
    if (isDeepSeekV4(model) && ids.includes("fp8_int8")) return "fp8_int8";
    if (ids.includes("bf16_fp16")) return "bf16_fp16";
    return ids[0];
  }

  function defaultIndexerPrecisionId(model, indexerOptions, fallbackPrecisionId) {
    const ids = optionIds(indexerOptions);
    if (isDeepSeekV4(model) && ids.includes("fp4_int4")) return "fp4_int4";
    if (fallbackPrecisionId && ids.includes(fallbackPrecisionId)) return fallbackPrecisionId;
    if (ids.includes("bf16_fp16")) return "bf16_fp16";
    return ids.includes("fp4_int4") ? "fp4_int4" : ids[0];
  }

  function presetById(presets, id) {
    return presets.find((preset) => preset.id === id) || presets[0];
  }

  function sourceLabel(sourceId) {
    return SOURCE_LABELS[sourceId] || String(sourceId).replace(/_/g, "-");
  }

  function formatPresetShape(defaults) {
    if (!defaults) return "";
    const parts = [];
    if (defaults.sessions) parts.push(`${formatInteger(defaults.sessions)} sessions`);
    if (defaults.average_turns) parts.push(`${formatInteger(defaults.average_turns)} avg turns`);
    if (defaults.shared_prefix_tokens) parts.push(`${formatInteger(defaults.shared_prefix_tokens)} shared-prefix tokens`);
    if (defaults.document_tokens) parts.push(`${formatInteger(defaults.document_tokens)} document-prefix tokens`);
    if (defaults.per_turn_input_tokens) parts.push(`${formatInteger(defaults.per_turn_input_tokens)} per-turn input tokens`);
    if (defaults.output_tokens) parts.push(`${formatInteger(defaults.output_tokens)} output tokens added to later history`);
    return parts.join("; ");
  }

  function tracePresetHelpText(preset) {
    if (!preset) return "Select a public real trace with block or message identity.";
    const sources = (preset.sources || []).map(sourceLabel).join(", ");
    const nativeBlockSize = preset.native_block_size || preset.nativeBlockSize;
    const sourceKind = preset.source_kind || preset.sourceKind;
    const sections = [
      `${preset.label}: ${preset.summary || "Public trace converted into a block-request stream."}`,
    ];
    if (sourceKind === "hash") sections.push("Uses published hash/block identities directly, so cache hits are computed from repeated block ids.");
    if (sourceKind === "agent_text") sections.push("Source does not publish block ids; message/span text is converted offline into approximate hashed token buckets.");
    if (nativeBlockSize) sections.push(`Native block size: ${formatInteger(nativeBlockSize)} tokens.`);
    if (sources) sections.push(`Sources: ${sources}.`);
    return sections.join(" ");
  }

  function formatNumber(value, digits) {
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
  }

  function formatInteger(value) {
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
  }

  function formatPercent(value) {
    return `${formatNumber(value * 100, 1)}%`;
  }

  function formatDuration(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value)) return "—";
    if (value < 1) return `${Math.round(value * 1000)}ms`;
    if (value < 60) return `${value < 10 ? value.toFixed(1) : Math.round(value)}s`;
    if (value < 3600) return `${value < 600 ? (value / 60).toFixed(1) : Math.round(value / 60)}m`;
    return `${(value / 3600).toFixed(1)}h`;
  }

  // Epoch seconds -> human clock; tolerant of ms-scale timestamps and junk.
  function formatClock(epochSeconds) {
    let value = Number(epochSeconds);
    if (!Number.isFinite(value) || value <= 0) return "";
    if (value > 1e12) value /= 1000; // tolerate millisecond timestamps
    try {
      return new Date(value * 1000).toLocaleString();
    } catch (error) {
      return "";
    }
  }

  function formatTokensPerSec(tps) {
    const value = Number(tps);
    if (!Number.isFinite(value)) return "—";
    if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M tok/s`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(1)}k tok/s`;
    return `${Math.round(value)} tok/s`;
  }

  function formatCompactNumber(value) {
    const x = Number(value);
    if (!Number.isFinite(x)) return "—";
    const abs = Math.abs(x);
    if (abs >= 1e6) return `${(x / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `${(x / 1e3).toFixed(1)}k`;
    if (abs >= 10) return `${Math.round(x)}`;
    return `${x.toFixed(1)}`;
  }

  function formatCapacityGiB(gib) {
    const numeric = Number(gib);
    if (Number.isFinite(numeric) && numeric >= 1024 && numeric % 1024 === 0) return `${formatNumber(numeric / 1024, 0)} TiB`;
    return `${formatNumber(numeric, 0)} GiB`;
  }

  function setText(root, selector, text) {
    const node = root.querySelector(selector);
    if (node) node.textContent = text;
  }

  function checkboxValue(input) {
    return Boolean(input && input.checked);
  }

  function setStatus(root, text) {
    setText(root, "[data-lab-status]", text);
  }

  function createHelpButton(text, className) {
    const help = document.createElement("button");
    help.type = "button";
    help.className = className || "kv-lab-help";
    help.textContent = "?";
    help.dataset.labTooltip = text;
    help.setAttribute("aria-label", text);
    return help;
  }

  function startSyncJob(input, onProgress) {
    let timer = null;
    let cancelled = false;
    let preParsedTrace = null;
    const promise = new Promise((resolve, reject) => {
      let trace = null;
      let simulationTrace = null;
      let point = null;
      let pointIndex = 0;
      let policyIndex = 0;
      let sweep = null;

      function step() {
        if (cancelled) return;
        try {
          if (!trace) {
            trace = preParsedTrace
              ? preParsedTrace
              : input.uploadText != null
                ? parseUploadedTrace(input.uploadText, input.uploadOptions || {})
                : generateTrace(input.preset, input.params, input.seed || DEFAULT_SEED);
            // Uploaded traces already carry an interned __flat plan, so
            // buildExecutionPlan reuses it directly — no need to materialize and
            // re-normalize a requests array the simulator would never read.
            simulationTrace = trace.__flat
              ? trace
              : Object.assign({}, trace, { normalizedRequests: normalizeRequests(trace) });
            const blockSize = toPositiveInteger(input.settings && input.settings.blockSize, trace.blockSize || DEFAULT_BLOCK_SIZE);
            const accountingSettings = Object.assign({}, input.settings || {}, { estimateTokens: traceEstimateTokens(trace, input.model) });
            const bytesPerToken = estimateBytesPerToken(input.model, accountingSettings);
            sweep = {
              blockSize,
              bytesPerToken,
              bytesPerBlock: bytesPerToken * blockSize,
              points: [],
              policies: (input.settings && input.settings.policies) || POLICIES,
              values: capacityValues(input.settings || {}),
            };
          }

          if (!point) {
            const gib = sweep.values[pointIndex];
            point = { gib, cacheBlocks: cacheBlocksForGiB(gib, sweep.bytesPerBlock), results: {} };
          }

          const policy = sweep.policies[policyIndex];
          point.results[policy] = simulatePolicy(simulationTrace, point.cacheBlocks, policy, {
            warmupFraction: input.settings && input.settings.warmupFraction,
          });
          policyIndex += 1;

          if (policyIndex >= sweep.policies.length) {
            sweep.points.push(point);
            point = null;
            policyIndex = 0;
            pointIndex += 1;
            if (typeof onProgress === "function") onProgress(pointIndex, sweep.values.length);
          }

          if (pointIndex >= sweep.values.length) {
            delete sweep.values;
            if (input.settings && input.settings.computeCeiling) {
              const ceiling = infiniteCacheReuse(simulationTrace, {
                warmupFraction: input.settings.warmupFraction,
              });
              sweep.reuseCeiling = ceiling.hitRate;
              sweep.warmupRequests = ceiling.warmupRequests;
            }
            let timeStats = null;
            try {
              timeStats = computeTimeSeries(trace);
            } catch (error) {
              timeStats = null;
            }
            resolve({
              jobId: input.jobId,
              cacheKey: input.cacheKey,
              preset: input.preset,
              trace,
              sweep,
              timeStats,
            });
            return;
          }
          timer = setTimeout(step, FALLBACK_STEP_DELAY_MS);
        } catch (error) {
          reject(error);
        }
      }

      // No-Worker fallback for a File upload: stream-parse it (async, on the main
      // thread) before the stepwise sweep begins. Rare — Workers are ~universal.
      if (input.uploadFile && typeof createTraceTextStream === "function") {
        parseUploadedTraceStreaming(
          createTraceTextStream(input.uploadFile, !!input.gzip, null),
          Object.assign({}, input.uploadOptions || {}),
        )
          .then((parsed) => {
            if (cancelled) return;
            preParsedTrace = parsed;
            timer = setTimeout(step, 0);
          })
          .catch(reject);
      } else {
        timer = setTimeout(step, 0);
      }
    });
    return {
      promise,
      cancel() {
        cancelled = true;
        if (timer) clearTimeout(timer);
      },
    };
  }

  function startWorkerJob(input, options, onProgress) {
    const worker = new Worker(options.workerUrl);
    let settled = false;
    const promise = new Promise((resolve, reject) => {
      worker.onmessage = (event) => {
        const message = event.data || {};
        if (message.type === "progress") {
          if (typeof onProgress === "function") onProgress(message.completed, message.total);
          return;
        }
        settled = true;
        worker.terminate();
        if (message.error) {
          if (message.stack && typeof console !== "undefined") console.error("KV Cache Lab worker error:\n" + message.stack);
          reject(new Error(message.error));
        } else {
          resolve(message.result || message);
        }
      };
      worker.onerror = (event) => {
        settled = true;
        worker.terminate();
        const where = event && event.filename ? ` (${event.filename}:${event.lineno || 0}:${event.colno || 0})` : "";
        if (typeof console !== "undefined") console.error("KV Cache Lab worker failed:", event && (event.message || event));
        reject(new Error((event && event.message ? event.message : "Worker failed") + where));
      };
      worker.postMessage(
        Object.assign({}, input, {
          type: "run",
          calculatorScriptUrl: options.calculatorUrl,
          labScriptUrl: options.labUrl,
        }),
      );
    });
    return {
      promise,
      cancel() {
        if (!settled) worker.terminate();
      },
    };
  }

  function setSelectOptions(select, options, preferredValue, labelForOption) {
    if (!select) return;
    select.innerHTML = "";
    options.forEach((option) => {
      const item = document.createElement("option");
      item.value = option.id || option;
      item.textContent = labelForOption ? labelForOption(option) : option.label || String(option);
      select.appendChild(item);
    });
    const values = options.map((option) => option.id || option);
    select.value = values.includes(preferredValue) ? preferredValue : values[0];
  }

  function renderMetric(label, value) {
    const item = document.createElement("div");
    const key = document.createElement("span");
    key.className = "kv-lab-metric-label";
    const text = document.createElement("span");
    text.textContent = label;
    key.appendChild(text);
    if (METRIC_HELP[label]) key.appendChild(createHelpButton(METRIC_HELP[label], "kv-lab-help kv-lab-metric-help"));
    const val = document.createElement("strong");
    val.textContent = value;
    item.append(key, val);
    return item;
  }

  function renderMetrics(root, trace, sweep) {
    const list = root.querySelector("[data-lab-metrics]");
    if (!list) return;
    const maxPoint = sweep.points[sweep.points.length - 1];
    const summary = (trace && trace.summary) || {};
    const requestCount = Array.isArray(trace && trace.requests) ? trace.requests.length : summary.requests || 0;
    const warmupRequests =
      Number.isFinite(Number(sweep.warmupRequests))
        ? Number(sweep.warmupRequests)
        : maxPoint && maxPoint.results && maxPoint.results.lru
          ? maxPoint.results.lru.warmupRequests
          : summary.warmupRequests || 0;
    const averageInputTokens =
      Number.isFinite(Number(summary.averageInputTokens))
        ? Number(summary.averageInputTokens)
        : 0;
    const metrics = [
      ["Trace requests", formatInteger(requestCount)],
      ["Warmup skipped", formatInteger(warmupRequests)],
      ["Avg input tokens", formatNumber(averageInputTokens, 0)],
    ];
    if (Number.isFinite(Number(summary.timeSpanSeconds)) && Number(summary.timeSpanSeconds) > 0) {
      metrics.push(["Loaded time span", formatDuration(Number(summary.timeSpanSeconds))]);
    }
    if (Number.isFinite(Number(summary.uniqueBlocks))) metrics.push(["Unique blocks", formatInteger(summary.uniqueBlocks)]);
    if (Number.isFinite(Number(sweep.reuseCeiling))) metrics.push(["Hit rate ceiling", formatPercent(Number(sweep.reuseCeiling))]);
    if (Number.isFinite(Number(sweep.blockSize || trace.blockSize))) metrics.push(["Native block size", `${formatInteger(sweep.blockSize || trace.blockSize)} tokens`]);
    if (maxPoint) metrics.push(["Max cache blocks", formatInteger(maxPoint.cacheBlocks)]);
    list.innerHTML = "";
    metrics.forEach(([label, value]) => list.appendChild(renderMetric(label, value)));
  }

  function clearSvg(svg) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  }

  function svgNode(name, attrs) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }

  function renderChart(root, sweep) {
    const svg = root.querySelector("[data-lab-chart]");
    if (!svg) return;
    clearSvg(svg);
    const width = 780;
    const height = 380;
    const margin = { top: 42, right: 32, bottom: 54, left: 64 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const xScale = (index) => margin.left + (index / Math.max(1, sweep.points.length - 1)) * plotWidth;
    const yScale = (value) => margin.top + (1 - clamp(value, 0, 1)) * plotHeight;

    svg.appendChild(svgNode("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
    [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
      const y = yScale(tick);
      svg.appendChild(svgNode("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, stroke: "#e2e8f0", "stroke-width": 1 }));
      const label = svgNode("text", { x: margin.left - 12, y: y + 4, "text-anchor": "end", "font-size": 12, fill: "#64748b" });
      label.textContent = formatPercent(tick);
      svg.appendChild(label);
    });

    sweep.points.forEach((point, index) => {
      const x = xScale(index);
      svg.appendChild(svgNode("line", { x1: x, y1: margin.top, x2: x, y2: height - margin.bottom, stroke: "#f1f5f9", "stroke-width": 1 }));
      const label = svgNode("text", { x, y: height - 20, "text-anchor": "middle", "font-size": 12, fill: "#64748b" });
      label.textContent = formatCapacityGiB(point.gib);
      svg.appendChild(label);
    });

    svg.appendChild(svgNode("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, stroke: "#94a3b8", "stroke-width": 1.2 }));
    svg.appendChild(svgNode("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, stroke: "#94a3b8", "stroke-width": 1.2 }));

    if (Number.isFinite(Number(sweep.reuseCeiling))) {
      const ceilingY = yScale(Number(sweep.reuseCeiling));
      svg.appendChild(svgNode("line", {
        x1: margin.left,
        y1: ceilingY,
        x2: width - margin.right,
        y2: ceilingY,
        stroke: "#475569",
        "stroke-width": 1.4,
        "stroke-dasharray": "6 5",
      }));
      const ceilingLabel = svgNode("text", {
        x: width - margin.right - 12,
        y: Math.max(30, ceilingY - 10),
        "text-anchor": "end",
        "font-size": 11,
        fill: "#475569",
        "font-weight": 700,
      });
      ceilingLabel.textContent = `hit rate ceiling ${formatPercent(Number(sweep.reuseCeiling))}`;
      svg.appendChild(ceilingLabel);
    }

    const tooltip = svgNode("g", { visibility: "hidden", display: "none", "pointer-events": "none" });
    const tooltipBox = svgNode("rect", { width: 176, height: 76, rx: 6, fill: "#0f172a", opacity: 0.94 });
    const tooltipTitle = svgNode("text", { fill: "#ffffff", "font-size": 12, "font-weight": 700 });
    const tooltipRate = svgNode("text", { fill: "#dbeafe", "font-size": 12 });
    const tooltipBlocks = svgNode("text", { fill: "#cbd5e1", "font-size": 12 });
    tooltip.append(tooltipBox, tooltipTitle, tooltipRate, tooltipBlocks);

    function showTooltip(point, pointIndex, policy) {
      const x = xScale(pointIndex);
      const y = yScale(point.results[policy].hitRate);
      const boxWidth = 176;
      const boxHeight = 76;
      const boxX = Math.min(width - margin.right - boxWidth, Math.max(margin.left, x + 12));
      const boxY = Math.max(10, y - boxHeight - 12);
      tooltip.removeAttribute("display");
      tooltip.setAttribute("visibility", "visible");
      tooltipBox.setAttribute("width", boxWidth);
      tooltipBox.setAttribute("height", boxHeight);
      tooltipBox.setAttribute("x", boxX);
      tooltipBox.setAttribute("y", boxY);
      tooltipTitle.setAttribute("x", boxX + 12);
      tooltipTitle.setAttribute("y", boxY + 22);
      tooltipRate.setAttribute("x", boxX + 12);
      tooltipRate.setAttribute("y", boxY + 43);
      tooltipBlocks.setAttribute("x", boxX + 12);
      tooltipBlocks.setAttribute("y", boxY + 63);
      tooltipBlocks.setAttribute("visibility", "visible");
      tooltipTitle.textContent = `${POLICY_LABELS[policy]} @ ${formatCapacityGiB(point.gib)}`;
      tooltipRate.textContent = `Hit rate: ${formatPercent(point.results[policy].hitRate)}`;
      tooltipBlocks.textContent = `Cache blocks: ${formatInteger(point.cacheBlocks)}`;
    }

    function hideTooltip() {
      tooltip.setAttribute("visibility", "hidden");
      tooltip.setAttribute("display", "none");
      tooltipTitle.textContent = "";
      tooltipRate.textContent = "";
      tooltipBlocks.textContent = "";
    }

    function isTooltipTarget(node) {
      return Boolean(node && node.getAttribute && node.getAttribute("data-lab-tooltip-target") === "true");
    }

    function showPolicyHelp(policy, x, y) {
      const lines = POLICY_TOOLTIP_LINES[policy] || [POLICY_HELP[policy] || "", ""];
      const hasSecondLine = Boolean(lines[1]);
      const boxWidth = policy === "optimal" ? 430 : 220;
      const boxHeight = hasSecondLine ? 72 : 52;
      const boxX = Math.min(width - margin.right - boxWidth, Math.max(margin.left, x + 10));
      const boxY = Math.max(10, y + 10);
      tooltip.removeAttribute("display");
      tooltip.setAttribute("visibility", "visible");
      tooltipBox.setAttribute("width", boxWidth);
      tooltipBox.setAttribute("height", boxHeight);
      tooltipBox.setAttribute("x", boxX);
      tooltipBox.setAttribute("y", boxY);
      tooltipTitle.setAttribute("x", boxX + 12);
      tooltipTitle.setAttribute("y", boxY + 22);
      tooltipRate.setAttribute("x", boxX + 12);
      tooltipRate.setAttribute("y", boxY + 44);
      tooltipBlocks.setAttribute("x", boxX + 12);
      tooltipBlocks.setAttribute("y", boxY + 64);
      tooltipTitle.textContent = POLICY_LABELS[policy];
      tooltipRate.textContent = lines[0] || "";
      tooltipBlocks.textContent = lines[1] || "";
      tooltipBlocks.setAttribute("visibility", hasSecondLine ? "visible" : "hidden");
    }

    sweep.policies.forEach((policy, index) => {
      const points = sweep.points.map((point, pointIndex) => `${xScale(pointIndex)},${yScale(point.results[policy].hitRate)}`).join(" ");
      svg.appendChild(svgNode("polyline", { points, fill: "none", stroke: POLICY_COLORS[policy], "stroke-width": 3, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      sweep.points.forEach((point, pointIndex) => {
        const tooltipLabel = `${POLICY_LABELS[policy]} @ ${formatCapacityGiB(point.gib)}; Hit rate: ${formatPercent(point.results[policy].hitRate)}; Cache blocks: ${formatInteger(point.cacheBlocks)}`;
        const marker = svgNode("circle", {
          class: "kv-lab-point",
          cx: xScale(pointIndex),
          cy: yScale(point.results[policy].hitRate),
          r: 5,
          fill: POLICY_COLORS[policy],
          stroke: "#ffffff",
          "stroke-width": 2,
          tabindex: 0,
          focusable: "true",
          "aria-label": tooltipLabel,
          "data-lab-tooltip-target": "true",
          cursor: "pointer",
        });
        marker.addEventListener("pointerenter", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("pointermove", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("mouseenter", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("mousemove", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("focus", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("click", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("pointerleave", hideTooltip);
        marker.addEventListener("mouseleave", hideTooltip);
        marker.addEventListener("blur", hideTooltip);
        svg.appendChild(marker);
      });
      const legendX = margin.left + index * 130;
      svg.appendChild(svgNode("line", { x1: legendX, y1: 22, x2: legendX + 28, y2: 22, stroke: POLICY_COLORS[policy], "stroke-width": 4, "stroke-linecap": "round" }));
      const label = svgNode("text", {
        x: legendX + 36,
        y: 26,
        "font-size": 13,
        fill: "#0f172a",
        "font-weight": 700,
        tabindex: 0,
        focusable: "true",
        "aria-label": `${POLICY_LABELS[policy]}: ${POLICY_HELP[policy]}`,
        "data-lab-tooltip-target": "true",
        cursor: "help",
      });
      label.textContent = POLICY_LABELS[policy];
      label.addEventListener("pointerenter", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("pointermove", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("mouseenter", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("mousemove", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("focus", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("click", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("pointerleave", hideTooltip);
      label.addEventListener("mouseleave", hideTooltip);
      label.addEventListener("blur", hideTooltip);
      svg.appendChild(label);
    });

    svg.onpointermove = (event) => {
      if (!isTooltipTarget(event.target)) hideTooltip();
    };
    svg.onmousemove = (event) => {
      if (!isTooltipTarget(event.target)) hideTooltip();
    };
    svg.onpointerleave = hideTooltip;
    svg.onmouseleave = hideTooltip;
    svg.appendChild(tooltip);

    const yLabelX = 13;
    const yLabel = svgNode("text", { x: yLabelX, y: margin.top + plotHeight / 2, transform: `rotate(-90 ${yLabelX} ${margin.top + plotHeight / 2})`, "text-anchor": "middle", "font-size": 12, fill: "#475569", "font-weight": 700 });
    yLabel.textContent = "Prefill token hit rate";
    svg.appendChild(yLabel);

    const xLabel = svgNode("text", { x: margin.left + plotWidth / 2, y: height - 5, "text-anchor": "middle", "font-size": 12, fill: "#475569", "font-weight": 700 });
    xLabel.textContent = "KV cache budget";
    svg.appendChild(xLabel);
  }

  function niceChartMax(value) {
    const safe = Math.max(1, toNumber(value, 1));
    const power = 10 ** Math.floor(Math.log10(safe));
    const scaled = safe / power;
    const nice = scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
    return nice * power;
  }

  function renderPolicyValueChart(svg, sweep, options) {
    if (!svg) return;
    clearSvg(svg);
    if (!sweep || !Array.isArray(sweep.points) || !sweep.points.length) return;
    const opts = options || {};
    const valueFor = opts.valueFor || (() => 0);
    const width = 780;
    const height = 380;
    const margin = { top: 42, right: 32, bottom: 54, left: 72 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const values = [];
    (sweep.policies || POLICIES).forEach((policy) => {
      sweep.points.forEach((point) => {
        const result = point.results && point.results[policy];
        if (!result) return;
        const value = valueFor(result, point, policy);
        if (Number.isFinite(Number(value))) values.push(Number(value));
      });
    });
    const yMax = opts.yMax || niceChartMax(Math.max(1, ...values));
    const xScale = (index) => margin.left + (index / Math.max(1, sweep.points.length - 1)) * plotWidth;
    const yScale = (value) => margin.top + (1 - clamp(value / yMax, 0, 1)) * plotHeight;
    const yFormat = opts.yFormat || ((value) => formatNumber(value, 0));

    svg.appendChild(svgNode("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
    [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
      const value = yMax * tick;
      const y = yScale(value);
      svg.appendChild(svgNode("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, stroke: "#e2e8f0", "stroke-width": 1 }));
      const label = svgNode("text", { x: margin.left - 12, y: y + 4, "text-anchor": "end", "font-size": 12, fill: "#64748b" });
      label.textContent = yFormat(value);
      svg.appendChild(label);
    });

    sweep.points.forEach((point, index) => {
      const x = xScale(index);
      svg.appendChild(svgNode("line", { x1: x, y1: margin.top, x2: x, y2: height - margin.bottom, stroke: "#f1f5f9", "stroke-width": 1 }));
      const label = svgNode("text", { x, y: height - 20, "text-anchor": "middle", "font-size": 12, fill: "#64748b" });
      label.textContent = formatCapacityGiB(point.gib);
      svg.appendChild(label);
    });

    svg.appendChild(svgNode("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, stroke: "#94a3b8", "stroke-width": 1.2 }));
    svg.appendChild(svgNode("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, stroke: "#94a3b8", "stroke-width": 1.2 }));

    const tooltip = svgNode("g", { visibility: "hidden", display: "none", "pointer-events": "none" });
    const tooltipBox = svgNode("rect", { width: 210, height: 92, rx: 6, fill: "#0f172a", opacity: 0.94 });
    const tooltipTitle = svgNode("text", { fill: "#ffffff", "font-size": 12, "font-weight": 700 });
    const tooltipLine1 = svgNode("text", { fill: "#dbeafe", "font-size": 12 });
    const tooltipLine2 = svgNode("text", { fill: "#cbd5e1", "font-size": 12 });
    const tooltipLine3 = svgNode("text", { fill: "#cbd5e1", "font-size": 12 });
    tooltip.append(tooltipBox, tooltipTitle, tooltipLine1, tooltipLine2, tooltipLine3);

    function showTooltip(point, pointIndex, policy) {
      const result = point.results[policy];
      const value = valueFor(result, point, policy);
      const lines = opts.tooltipLines ? opts.tooltipLines(result, point, policy, value) : [`Value: ${yFormat(value)}`];
      const x = xScale(pointIndex);
      const y = yScale(value);
      const boxWidth = 236;
      const boxHeight = 94;
      const boxX = Math.min(width - margin.right - boxWidth, Math.max(margin.left, x + 12));
      const boxY = Math.max(10, y - boxHeight - 12);
      tooltip.removeAttribute("display");
      tooltip.setAttribute("visibility", "visible");
      tooltipBox.setAttribute("width", boxWidth);
      tooltipBox.setAttribute("height", boxHeight);
      tooltipBox.setAttribute("x", boxX);
      tooltipBox.setAttribute("y", boxY);
      tooltipTitle.setAttribute("x", boxX + 12);
      tooltipTitle.setAttribute("y", boxY + 22);
      [tooltipLine1, tooltipLine2, tooltipLine3].forEach((node, lineIndex) => {
        node.setAttribute("x", boxX + 12);
        node.setAttribute("y", boxY + 44 + lineIndex * 18);
        node.textContent = lines[lineIndex] || "";
        node.setAttribute("visibility", lines[lineIndex] ? "visible" : "hidden");
      });
      tooltipTitle.textContent = `${POLICY_LABELS[policy]} @ ${formatCapacityGiB(point.gib)}`;
    }

    function hideTooltip() {
      tooltip.setAttribute("visibility", "hidden");
      tooltip.setAttribute("display", "none");
      tooltipTitle.textContent = "";
      tooltipLine1.textContent = "";
      tooltipLine2.textContent = "";
      tooltipLine3.textContent = "";
    }

    function isTooltipTarget(node) {
      return Boolean(node && node.getAttribute && node.getAttribute("data-lab-tooltip-target") === "true");
    }

    function showPolicyHelp(policy, x, y) {
      const lines = POLICY_TOOLTIP_LINES[policy] || [POLICY_HELP[policy] || "", ""];
      const visibleLines = lines.filter(Boolean);
      const boxWidth = policy === "optimal" ? 430 : 220;
      const boxHeight = visibleLines.length > 1 ? 72 : 52;
      const boxX = Math.min(width - margin.right - boxWidth, Math.max(margin.left, x + 10));
      const boxY = Math.max(10, y + 10);
      tooltip.removeAttribute("display");
      tooltip.setAttribute("visibility", "visible");
      tooltipBox.setAttribute("width", boxWidth);
      tooltipBox.setAttribute("height", boxHeight);
      tooltipBox.setAttribute("x", boxX);
      tooltipBox.setAttribute("y", boxY);
      tooltipTitle.setAttribute("x", boxX + 12);
      tooltipTitle.setAttribute("y", boxY + 22);
      [tooltipLine1, tooltipLine2, tooltipLine3].forEach((node, lineIndex) => {
        node.setAttribute("x", boxX + 12);
        node.setAttribute("y", boxY + 44 + lineIndex * 18);
        node.textContent = lines[lineIndex] || "";
        node.setAttribute("visibility", lines[lineIndex] ? "visible" : "hidden");
      });
      tooltipTitle.textContent = POLICY_LABELS[policy];
    }

    const policies = sweep.policies || POLICIES;
    policies.forEach((policy, index) => {
      const points = sweep.points
        .map((point, pointIndex) => `${xScale(pointIndex)},${yScale(valueFor(point.results[policy], point, policy))}`)
        .join(" ");
      svg.appendChild(svgNode("polyline", { points, fill: "none", stroke: POLICY_COLORS[policy], "stroke-width": 3, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      sweep.points.forEach((point, pointIndex) => {
        const result = point.results[policy];
        const value = valueFor(result, point, policy);
        const marker = svgNode("circle", {
          class: "kv-lab-point",
          cx: xScale(pointIndex),
          cy: yScale(value),
          r: 5,
          fill: POLICY_COLORS[policy],
          stroke: "#ffffff",
          "stroke-width": 2,
          tabindex: 0,
          focusable: "true",
          "aria-label": `${POLICY_LABELS[policy]} @ ${formatCapacityGiB(point.gib)}; ${opts.yTitle || "Value"}: ${yFormat(value)}`,
          "data-lab-tooltip-target": "true",
          cursor: "pointer",
        });
        marker.addEventListener("pointerenter", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("pointermove", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("focus", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("click", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("pointerleave", hideTooltip);
        marker.addEventListener("blur", hideTooltip);
        svg.appendChild(marker);
      });
      const legendX = margin.left + index * 130;
      svg.appendChild(svgNode("line", { x1: legendX, y1: 22, x2: legendX + 28, y2: 22, stroke: POLICY_COLORS[policy], "stroke-width": 4, "stroke-linecap": "round" }));
      const label = svgNode("text", {
        x: legendX + 36,
        y: 26,
        "font-size": 13,
        fill: "#0f172a",
        "font-weight": 700,
        tabindex: 0,
        focusable: "true",
        "aria-label": `${POLICY_LABELS[policy]}: ${POLICY_HELP[policy]}`,
        "data-lab-tooltip-target": "true",
        cursor: "help",
      });
      label.textContent = POLICY_LABELS[policy];
      label.addEventListener("pointerenter", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("pointermove", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("mouseenter", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("mousemove", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("focus", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("click", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("pointerleave", hideTooltip);
      label.addEventListener("mouseleave", hideTooltip);
      label.addEventListener("blur", hideTooltip);
      svg.appendChild(label);
    });

    svg.onpointermove = (event) => {
      if (!isTooltipTarget(event.target)) hideTooltip();
    };
    svg.onmousemove = (event) => {
      if (!isTooltipTarget(event.target)) hideTooltip();
    };
    svg.onpointerleave = hideTooltip;
    svg.onmouseleave = hideTooltip;
    svg.appendChild(tooltip);

    const yLabelX = 15;
    const yLabel = svgNode("text", { x: yLabelX, y: margin.top + plotHeight / 2, transform: `rotate(-90 ${yLabelX} ${margin.top + plotHeight / 2})`, "text-anchor": "middle", "font-size": 12, fill: "#475569", "font-weight": 700 });
    yLabel.textContent = opts.yTitle || "";
    svg.appendChild(yLabel);

    const xLabel = svgNode("text", { x: margin.left + plotWidth / 2, y: height - 5, "text-anchor": "middle", "font-size": 12, fill: "#475569", "font-weight": 700 });
    xLabel.textContent = "KV cache budget";
    svg.appendChild(xLabel);
  }

  function sweepHasUsefulCacheRate(sweep) {
    return Boolean(
      sweep &&
        Array.isArray(sweep.points) &&
        sweep.points.length &&
        sweep.points.every((point) =>
          (sweep.policies || POLICIES).every((policy) =>
            Number.isFinite(Number(point.results && point.results[policy] && point.results[policy].usefulCacheRate)),
          ),
        ),
    );
  }

  function renderDerivedCharts(root, sweep) {
    const section = root.querySelector("[data-lab-derived-charts]");
    if (!section) return;
    if (!sweep || !Array.isArray(sweep.points) || !sweep.points.length) {
      section.hidden = true;
      return;
    }
    section.hidden = false;
    renderPolicyValueChart(root.querySelector("[data-lab-throughput-chart]"), sweep, {
      yTitle: "Ideal prefill speedup",
      yFormat: (value) => `${formatNumber(value, value >= 10 ? 0 : 1)}x`,
      valueFor: (result) => throughputFromHitRate(result.hitRate),
      tooltipLines: (result, point, policy, value) => [
        `Speedup: ${formatNumber(value, value >= 10 ? 0 : 1)}x`,
        `Hit rate: ${formatPercent(result.hitRate)}`,
        `Miss compute: ${formatPercent(1 - clamp(result.hitRate, 0, 1))}`,
      ],
    });

    const usefulPanel = root.querySelector("[data-lab-useful-panel]");
    if (!sweepHasUsefulCacheRate(sweep)) {
      if (usefulPanel) usefulPanel.hidden = true;
      clearSvg(root.querySelector("[data-lab-useful-chart]"));
      return;
    }
    if (usefulPanel) usefulPanel.hidden = false;
    renderPolicyValueChart(root.querySelector("[data-lab-useful-chart]"), sweep, {
      yTitle: "Useful cache occupancy",
      yMax: 1,
      yFormat: (value) => formatPercent(value),
      valueFor: (result) => toNumber(result.usefulCacheRate, 0),
      tooltipLines: (result, point) => [
        `Useful occupancy: ${formatPercent(result.usefulCacheRate)}`,
        `Cache blocks: ${formatInteger(point.cacheBlocks)}`,
        `Samples: ${formatInteger(result.usefulCacheSamples || 0)}`,
      ],
    });
  }

  function renderSources(root, preset, metadata) {
    const node = root.querySelector("[data-lab-sources]");
    if (!node) return;
    const sources = metadata && metadata.sources ? metadata.sources : {};
    node.innerHTML = "";
    const links = (preset.sources || []).map((sourceId) => [sourceId, sources[sourceId]]).filter(([, href]) => href);
    if (!links.length) return;
    node.appendChild(document.createTextNode("Sources: "));
    links.forEach(([sourceId, href], index) => {
      if (index > 0) node.appendChild(document.createTextNode(", "));
      const link = document.createElement("a");
      link.href = href;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = sourceLabel(sourceId);
      node.appendChild(link);
    });
  }

  // Mini chart sharing the main hit-rate chart's exact frame style (viewBox,
  // margins, fonts, axis/grid colors) so the temporal panels look native.
  function drawMiniChart(svg, opts) {
    if (!svg) return;
    clearSvg(svg);
    const width = 780;
    const height = 380;
    const margin = { top: 42, right: 32, bottom: 54, left: 64 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const vals = opts.values || [];
    const n = vals.length;
    const yMax = opts.yMax != null ? opts.yMax : Math.max(1e-9, ...vals);
    const yFmt = opts.yFormat || ((v) => String(v));
    const color = opts.color || POLICY_COLORS.fifo;
    const yScale = (v) => margin.top + (1 - clamp(v / yMax, 0, 1)) * plotHeight;

    svg.appendChild(svgNode("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
    [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
      const y = yScale(yMax * tick);
      svg.appendChild(svgNode("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, stroke: "#e2e8f0", "stroke-width": 1 }));
      const label = svgNode("text", { x: margin.left - 12, y: y + 4, "text-anchor": "end", "font-size": 12, fill: "#64748b" });
      label.textContent = yFmt(yMax * tick);
      svg.appendChild(label);
    });

    if (opts.kind === "bars") {
      const band = plotWidth / Math.max(1, n);
      vals.forEach((v, i) => {
        const bx = margin.left + i * band + band * 0.18;
        const by = yScale(v);
        svg.appendChild(svgNode("rect", { x: bx, y: by, width: band * 0.64, height: height - margin.bottom - by, fill: color, rx: 2 }));
        const cx = margin.left + i * band + band * 0.5;
        const label = svgNode("text", { x: cx, y: height - 20, "text-anchor": "middle", "font-size": 12, fill: "#64748b" });
        label.textContent = (opts.barLabels && opts.barLabels[i]) || "";
        svg.appendChild(label);
      });
    } else {
      const xAt = (i) => (n <= 1 ? margin.left + plotWidth / 2 : margin.left + (i / (n - 1)) * plotWidth);
      (opts.xTicks || []).forEach((t) => {
        const x = margin.left + t.pos * plotWidth;
        svg.appendChild(svgNode("line", { x1: x, y1: margin.top, x2: x, y2: height - margin.bottom, stroke: "#f1f5f9", "stroke-width": 1 }));
        const label = svgNode("text", { x, y: height - 20, "text-anchor": "middle", "font-size": 12, fill: "#64748b" });
        label.textContent = t.label;
        svg.appendChild(label);
      });
      const pts = vals.map((v, i) => `${xAt(i)},${yScale(v)}`).join(" ");
      if (opts.kind === "area") {
        svg.appendChild(svgNode("polygon", { points: `${margin.left},${height - margin.bottom} ${pts} ${width - margin.right},${height - margin.bottom}`, fill: color, "fill-opacity": 0.15 }));
      }
      svg.appendChild(svgNode("polyline", { points: pts, fill: "none", stroke: color, "stroke-width": 3, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    }

    svg.appendChild(svgNode("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, stroke: "#94a3b8", "stroke-width": 1.2 }));
    svg.appendChild(svgNode("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, stroke: "#94a3b8", "stroke-width": 1.2 }));

    const yLabelX = 13;
    const yLabel = svgNode("text", {
      x: yLabelX,
      y: margin.top + plotHeight / 2,
      transform: `rotate(-90 ${yLabelX} ${margin.top + plotHeight / 2})`,
      "text-anchor": "middle",
      "font-size": 12,
      fill: "#475569",
      "font-weight": 700,
    });
    yLabel.textContent = opts.yTitle || "";
    svg.appendChild(yLabel);

    const xLabel = svgNode("text", { x: margin.left + plotWidth / 2, y: height - 5, "text-anchor": "middle", "font-size": 12, fill: "#475569", "font-weight": 700 });
    xLabel.textContent = opts.xTitle || "";
    svg.appendChild(xLabel);
  }

  function timeAxisTicks(timeStats) {
    const buckets = timeStats.timeBuckets;
    return [0, 0.25, 0.5, 0.75, 1].map((pos) => {
      const i = Math.min(buckets.length - 1, Math.round(pos * (buckets.length - 1)));
      return { pos, label: `+${formatDuration(buckets[i].offset)}` };
    });
  }

  function renderTimeStats(root, timeStats) {
    const section = root.querySelector("[data-lab-timeseries]");
    if (!section) return;
    if (!timeStats || !timeStats.timeBuckets || !timeStats.timeBuckets.length) {
      section.hidden = true;
      return;
    }
    section.hidden = false;
    const xTicks = timeAxisTicks(timeStats);

    drawMiniChart(root.querySelector("[data-lab-chart-hitrate]"), {
      kind: "line",
      values: timeStats.timeBuckets.map((b) => b.hitRate),
      yMax: 1,
      yFormat: (v) => formatPercent(v),
      yTitle: "Prefill token hit rate",
      xTitle: "Time since trace start",
      color: POLICY_COLORS.lru,
      xTicks,
    });

    const peakRate = Math.max(1e-9, ...timeStats.timeBuckets.map((b) => b.newTokensPerSec));
    drawMiniChart(root.querySelector("[data-lab-chart-production]"), {
      kind: "area",
      values: timeStats.timeBuckets.map((b) => b.newTokensPerSec),
      yMax: peakRate,
      yFormat: (v) => formatCompactNumber(v),
      yTitle: "New KV (tokens/s)",
      xTitle: "Time since trace start",
      color: POLICY_COLORS.fifo,
      xTicks,
    });

    const peakShare = Math.max(1e-9, ...timeStats.reuseHistogram.map((b) => b.share));
    drawMiniChart(root.querySelector("[data-lab-chart-reuse]"), {
      kind: "bars",
      values: timeStats.reuseHistogram.map((b) => b.share),
      barLabels: timeStats.reuseHistogram.map((b) => b.label),
      yMax: peakShare,
      yFormat: (v) => formatPercent(v),
      yTitle: "Share of reused tokens",
      xTitle: "Time since previous use",
      color: POLICY_COLORS.optimal,
    });
  }

  function renderResults(root, preset, trace, sweep, metadata) {
    const isPrecomputed = sweep && sweep.precomputed;
    const isUpload = trace && trace.presetId === UPLOAD_PRESET_ID;
    const sourceKind = trace && trace.sourceKind;
    const summary = (trace && trace.summary) || {};
    const skippedNote =
      isUpload && (summary.skipped || summary.parseErrors)
        ? ` Skipped ${formatInteger((summary.skipped || 0) + (summary.parseErrors || 0))} unparseable/empty line(s).`
        : "";
    const spanNote =
      isUpload && Number.isFinite(Number(summary.timeSpanSeconds)) && Number(summary.timeSpanSeconds) > 0
        ? ` The loaded requests span ${formatDuration(Number(summary.timeSpanSeconds))}${
            formatClock(summary.tStart) ? ` (${formatClock(summary.tStart)} – ${formatClock(summary.tEnd)})` : ""
          }.`
        : "";
    const cappedNote =
      isUpload && summary.capped
        ? ` Trace too large to fully load in-browser: analyzed the first ${formatInteger(summary.requests || 0)} request(s) (${formatInteger((summary.uniqueBlocks || 0))} unique blocks). For the complete curve, run the offline precompute pipeline.`
        : "";
    const modeNote = isPrecomputed
      ? "This chart uses offline precomputed curves from a normalized real trace; the browser does not replay the full trace."
      : isUpload
        ? `Computed live in your browser from your uploaded trace; the file never leaves your device.${spanNote}${skippedNote}${cappedNote}`
        : "Deterministic simulation converts the selected preset into block requests, skips the warmup window, and reports prefill input-token hits.";
    const identityNote =
      sourceKind === "agent_text"
        ? "This agent trace is approximated from message/span text because the source does not publish block ids."
        : sourceKind === "hash"
          ? "Repeated hash/block ids define cache hits."
          : "";
    setText(
      root,
      "[data-lab-output='note']",
      `${preset.summary || ""} ${modeNote} ${identityNote} ${trace.sourceBlockSizeNote || metadata.block_size_note || ""}`,
    );
    renderMetrics(root, trace, sweep);
    renderChart(root, sweep);
    renderDerivedCharts(root, sweep);
    renderSources(root, preset, metadata);
  }

  function renderUnavailable(root, preset, model, settings) {
    const metrics = root.querySelector("[data-lab-metrics]");
    const svg = root.querySelector("[data-lab-chart]");
    const derived = root.querySelector("[data-lab-derived-charts]");
    const sources = root.querySelector("[data-lab-sources]");
    if (metrics) metrics.innerHTML = "";
    if (svg) clearSvg(svg);
    if (derived) {
      derived.hidden = true;
      clearSvg(root.querySelector("[data-lab-throughput-chart]"));
      clearSvg(root.querySelector("[data-lab-useful-chart]"));
    }
    if (sources) sources.innerHTML = "";
    const precision = settings && settings.precision ? settings.precision : "default";
    const indexer = settings && settings.indexerPrecision ? `, indexer ${settings.indexerPrecision}` : "";
    setText(
      root,
      "[data-lab-output='note']",
      `No precomputed curve is available for ${preset.label} with ${model.label} (${precision}${indexer}). Run scripts/kv-cache-lab-precompute-curves.mjs for this model/precision combination, then rebuild the Hugo data.`,
    );
  }

  function initialize(root, data, options) {
    const models = data.models || [];
    const precomputed = data.precomputed || null;
    const rawPresets = data.lab && data.lab.presets ? data.lab.presets : [];
    const presets = precomputed && precomputed.traces
      ? rawPresets.filter((preset) => precomputedTrace(precomputed, preset))
      : rawPresets;
    if (!root || !models.length || !presets.length) return;
    const runtimeOptions = options || {};
    const defaults = (data.lab && data.lab.simulation_defaults) || {};
    const metadata = (data.lab && data.lab.metadata) || {};
    const resultCache = new Map();
    // Persistent worker pool + plan/result cache. Created once per mount; spawns
    // its workers lazily on the first compute. Null when Workers are unavailable
    // (then startJob falls back to a single worker, then to the main thread).
    const labEngine =
      runtimeOptions.workerUrl && runtimeOptions.calculatorUrl && runtimeOptions.labUrl && typeof Worker === "function"
        ? createLabEngine(runtimeOptions)
        : null;
    let debounceTimer = null;
    let latestJobId = 0;
    let activeJob = null;
    const inputs = {
      modelFamily: root.querySelector("[data-lab-input='modelFamily']"),
      model: root.querySelector("[data-lab-input='model']"),
      precision: root.querySelector("[data-lab-input='precision']"),
      indexerPrecision: root.querySelector("[data-lab-input='indexerPrecision']"),
      includeDraftKvCache: root.querySelector("[data-lab-input='includeDraftKvCache']"),
      preset: root.querySelector("[data-lab-input='preset']"),
    };
    const paramInputs = Array.from(root.querySelectorAll("[data-lab-param]"));
    const uploadInput = root.querySelector("[data-lab-upload-input]");
    const uploadZone = root.querySelector("[data-lab-upload-zone]");
    const uploadClear = root.querySelector("[data-lab-upload-clear]");
    const uploadBlockSizeInput = root.querySelector("[data-lab-upload-blocksize]");
    const progressEl = root.querySelector("[data-lab-progress]");
    const progressFill = root.querySelector("[data-lab-progress-fill]");
    const progressText = root.querySelector("[data-lab-progress-text]");
    let uploadState = null;

    function showProgress() {
      if (!progressEl) return;
      progressEl.hidden = false;
      if (progressFill) progressFill.style.width = "0%";
      if (progressText) progressText.textContent = "0%";
    }

    function setProgress(completed, total) {
      if (!progressEl) return;
      progressEl.hidden = false;
      const fraction = total > 0 ? Math.max(0, Math.min(1, completed / total)) : 0;
      if (progressFill) progressFill.style.width = `${(fraction * 100).toFixed(0)}%`;
      if (progressText) progressText.textContent = `${completed}/${total}`;
    }

    function hideProgress() {
      if (progressEl) progressEl.hidden = true;
    }

    function failJob(phase, error) {
      hideProgress();
      if (typeof console !== "undefined") console.error(`KV Cache Lab ${phase} error:`, error);
      root.dataset.state = "error";
      setStatus(root, phase === "render" ? "Render failed" : "Calculation failed");
      setText(root, "[data-lab-output='note']", (error && error.message) || String(error));
    }

    function currentUploadBlockSize() {
      const typed = uploadBlockSizeInput && Number(uploadBlockSizeInput.value);
      if (typed && typed > 0) return Math.floor(typed);
      if (uploadState && uploadState.detectedBlockSize) return uploadState.detectedBlockSize;
      return 0;
    }

    function selectedModel() {
      return modelById(models, inputs.model.value);
    }

    function selectedPreset() {
      if (uploadState && inputs.preset.value === UPLOAD_PRESET_ID) return uploadState.preset;
      return presetById(presets, inputs.preset.value);
    }

    function setUploadStatus(text) {
      const node = root.querySelector("[data-lab-upload-status]");
      if (node) node.textContent = text || "";
    }

    function failUploadValidation(error) {
      if (activeJob) {
        activeJob.cancel();
        activeJob = null;
      }
      uploadState = null;
      const option = inputs.preset.querySelector(`option[value="${UPLOAD_PRESET_ID}"]`);
      if (option) option.remove();
      if (uploadClear) uploadClear.hidden = true;
      hideProgress();
      root.dataset.state = "error";
      setStatus(root, "Invalid trace");
      const message = (error && error.message) || String(error);
      setUploadStatus("Invalid trace");
      setText(root, "[data-lab-output='note']", message);
    }

    function setUploadOption(label) {
      let option = inputs.preset.querySelector(`option[value="${UPLOAD_PRESET_ID}"]`);
      if (!option) {
        option = document.createElement("option");
        option.value = UPLOAD_PRESET_ID;
        inputs.preset.appendChild(option);
      }
      option.textContent = label;
      inputs.preset.value = UPLOAD_PRESET_ID;
    }

    function formatFileSize(bytes) {
      const value = Number(bytes) || 0;
      if (value >= 1e9) return `${(value / 1e9).toFixed(1)} GB`;
      if (value >= 1e6) return `${(value / 1e6).toFixed(value >= 1e7 ? 0 : 1)} MB`;
      return `${Math.max(1, Math.round(value / 1e3))} KB`;
    }

    // Read just the decoded head to validate JSONL schema and detect the source
    // block size before starting an expensive full simulation.
    async function inspectUploadFileHead(file, gzip) {
      try {
        let buffer = "";
        for await (const chunk of createTraceTextStream(file, gzip, null)) {
          buffer += chunk || "";
          if (buffer.indexOf("\n") >= 0 || buffer.length >= UPLOAD_HEAD_MAX_CHARS) {
            return inspectUploadedTraceHeadText(buffer);
          }
        }
        return inspectUploadedTraceHeadText(buffer);
      } catch (error) {
        return {
          valid: false,
          error: gzip
            ? "Could not decompress uploaded trace. Check that this is a valid JSONL.GZ file."
            : "Could not read uploaded trace. Check that this is a valid JSONL file.",
        };
      }
    }

    // Keep the File itself (never read it into a string on the main thread); the
    // worker streams + gzip-decompresses + parses it incrementally.
    function loadUploadFile(file, isGzip, detected) {
      const label = file.name || "Uploaded trace";
      uploadState = {
        file,
        isGzip,
        fileName: label,
        detectedBlockSize: detected,
        token: `${file.size}:${file.lastModified}:${label}`,
        preset: { id: UPLOAD_PRESET_ID, label, summary: "", sources: [], defaults: {} },
      };
      if (uploadBlockSizeInput) {
        uploadBlockSizeInput.value = "";
        uploadBlockSizeInput.placeholder = detected ? `auto (${detected})` : "declared in trace";
      }
      setUploadOption(`Uploaded: ${label}`);
      if (uploadClear) uploadClear.hidden = false;
      populateFamilies(inputs.modelFamily.value);
      populateModels(inputs.model.value);
      syncModelControls();
      syncTraceHelp();
      setUploadStatus(`Loaded ${label} (${formatFileSize(file.size)}${isGzip ? ", gzip" : ""})`);
      scheduleUpdate(0);
    }

    function clearUpload() {
      uploadState = null;
      const option = inputs.preset.querySelector(`option[value="${UPLOAD_PRESET_ID}"]`);
      if (option) option.remove();
      if (uploadInput) uploadInput.value = "";
      if (uploadClear) uploadClear.hidden = true;
      if (uploadBlockSizeInput) {
        uploadBlockSizeInput.value = "";
        uploadBlockSizeInput.placeholder = "auto";
      }
      inputs.preset.value = presets[0].id;
      setUploadStatus("");
      applyPresetDefaults();
      populateFamilies(inputs.modelFamily.value);
      populateModels(inputs.model.value);
      syncModelControls();
      syncTraceHelp();
      scheduleUpdate(0);
    }

    function readUploadFile(file) {
      if (!file) return;
      const isGzip = /\.gz$/i.test(file.name || "");
      setUploadStatus(`Reading ${file.name} (${formatFileSize(file.size)})…`);
      inspectUploadFileHead(file, isGzip).then((inspection) => {
        if (!inspection || !inspection.valid) {
          failUploadValidation(new Error((inspection && inspection.error) || "Uploaded trace is not valid JSONL."));
          return;
        }
        loadUploadFile(file, isGzip, inspection.blockSize || 0);
      });
    }

    function syncTraceHelp() {
      const help = root.querySelector("[data-lab-trace-help]");
      if (!help) return;
      const text = tracePresetHelpText(selectedPreset());
      help.dataset.labTooltip = text;
      help.setAttribute("aria-label", text);
      help.removeAttribute("title");
    }

    function setupTraceHelp() {
      const help = root.querySelector("[data-lab-trace-help]");
      if (!help) return;
      help.addEventListener("click", (event) => {
        event.preventDefault();
      });
    }

    function populateFamilies(preferredFamily) {
      const availableIds = availableModelIdsFor(precomputed, selectedPreset());
      const familyModels = availableIds ? models.filter((model) => availableIds.has(model.id)) : models;
      setSelectOptions(inputs.modelFamily, sortedModelFamilies(familyModels), preferredFamily, (family) => family);
    }

    function populateModels(preferredModelId) {
      const availableIds = availableModelIdsFor(precomputed, selectedPreset());
      const familyModels = modelsForFamily(models, inputs.modelFamily.value).filter((model) => !availableIds || availableIds.has(model.id));
      setSelectOptions(inputs.model, familyModels, preferredModelId, (model) => model.label);
    }

    function syncPrecisionControls(model) {
      const availableSettings = availableSettingsFor(precomputed, selectedPreset(), model);
      const availablePrecisions = availableSettings.length
        ? Array.from(new Set(availableSettings.map((setting) => setting.precision).filter(Boolean)))
        : [];
      const precisionOptions = availablePrecisions.length
        ? (data.precision_options || []).filter((option) => availablePrecisions.includes(option.id))
        : data.precision_options || [];
      const precisionDefault = availablePrecisions[0] || defaultPrecisionId(model, data.precision_options || []);
      setSelectOptions(inputs.precision, precisionOptions, precisionDefault, (option) => option.label);

      const indexerControl = root.querySelector("[data-lab-indexer-control]");
      const availableIndexerPrecisions = availableSettings.length
        ? Array.from(new Set(availableSettings.map((setting) => setting.indexerPrecision).filter(Boolean)))
        : [];
      const showIndexer = hasIndexerCache(model) && (!availableSettings.length || availableIndexerPrecisions.length > 0);
      if (indexerControl) indexerControl.hidden = !showIndexer;
      if (showIndexer) {
        const indexerOptions = availableIndexerPrecisions.length
          ? (data.indexer_precision_options || []).filter((option) => availableIndexerPrecisions.includes(option.id))
          : data.indexer_precision_options || [];
        const indexerDefault = availableIndexerPrecisions[0] || defaultIndexerPrecisionId(model, data.indexer_precision_options || [], inputs.precision ? inputs.precision.value : undefined);
        setSelectOptions(inputs.indexerPrecision, indexerOptions, indexerDefault, (option) => option.label);
      }
    }

    function syncDraftControl(model) {
      const control = root.querySelector("[data-lab-draft-control]");
      const availableSettings = availableSettingsFor(precomputed, selectedPreset(), model);
      const showDraft = hasDraftKvCache(model) && (!availableSettings.length || availableSettings.some((setting) => setting.includeDraftKvCache));
      if (control) control.hidden = !showDraft;
      if (inputs.includeDraftKvCache) inputs.includeDraftKvCache.checked = false;
    }

    function syncModelControls() {
      const model = selectedModel();
      syncPrecisionControls(model);
      syncDraftControl(model);
    }

    function applyPresetDefaults() {
      const preset = selectedPreset();
      paramInputs.forEach((input) => {
        const key = input.dataset.labParam;
        if (key === "requests") input.value = defaults.requests || DEFAULT_REQUESTS;
        else if (preset.defaults && Object.prototype.hasOwnProperty.call(preset.defaults, key)) input.value = preset.defaults[key];
      });
    }

    function readParams() {
      const params = { requests: defaults.requests || DEFAULT_REQUESTS };
      paramInputs.forEach((input) => {
        params[input.dataset.labParam] = input.value;
      });
      params.blockSize = selectedPreset().native_block_size || selectedPreset().nativeBlockSize || defaults.block_size || DEFAULT_BLOCK_SIZE;
      return params;
    }

    function readSettings(blockSize) {
      const model = selectedModel();
      return {
        precision: inputs.precision.value,
        indexerPrecision: hasIndexerCache(model) && inputs.indexerPrecision ? inputs.indexerPrecision.value : undefined,
        includeDraftKvCache: hasDraftKvCache(model) && checkboxValue(inputs.includeDraftKvCache),
        precisionOptions: data.precision_options,
        indexerPrecisionOptions: data.indexer_precision_options,
        blockSize,
        capacityGiBValues: defaults.capacity_gib_values || DEFAULT_CAPACITY_GIB_VALUES,
        warmupFraction: defaults.warmup_fraction || DEFAULT_WARMUP_FRACTION,
      };
    }

    function computationInput() {
      const preset = selectedPreset();
      const blockSize = toPositiveInteger(preset.native_block_size || preset.nativeBlockSize || defaults.block_size, DEFAULT_BLOCK_SIZE);
      return {
        preset,
        model: selectedModel(),
        params: readParams(),
        settings: readSettings(blockSize),
        seed: defaults.seed || DEFAULT_SEED,
      };
    }

    // Single Web Worker with a transparent main-thread retry. Used as the
    // fallback when the pool is unavailable or fails mid-job.
    function startWorkerJobWithSyncFallback(input, onProgress) {
      root.dataset.computeMode = "worker";
      let workerJob = null;
      try {
        workerJob = startWorkerJob(input, runtimeOptions, onProgress);
      } catch (error) {
        root.dataset.computeMode = "fallback";
        return startSyncJob(input, onProgress);
      }
      // If the Web Worker fails at runtime (e.g. it can't load its scripts in
      // this environment), transparently retry on the main thread instead of
      // surfacing a hard failure. The sync path is identical and self-contained.
      let cancelled = false;
      let fallbackJob = null;
      const promise = workerJob.promise.catch((error) => {
        if (cancelled) throw error;
        if (typeof console !== "undefined") {
          console.warn("KV Cache Lab worker failed; retrying on the main thread.", error && error.message);
        }
        root.dataset.computeMode = "fallback";
        fallbackJob = startSyncJob(input, onProgress);
        return fallbackJob.promise;
      });
      return {
        promise,
        cancel() {
          cancelled = true;
          workerJob.cancel();
          if (fallbackJob) fallbackJob.cancel();
        },
      };
    }

    function startJob(input, onProgress) {
      const canUseWorker =
        runtimeOptions.workerUrl &&
        runtimeOptions.calculatorUrl &&
        runtimeOptions.labUrl &&
        typeof Worker === "function";
      if (!canUseWorker) {
        root.dataset.computeMode = "fallback";
        return startSyncJob(input, onProgress);
      }
      if (!labEngine) {
        return startWorkerJobWithSyncFallback(input, onProgress);
      }
      // Pool path. A cancelled job rejects with "cancelled" and is ignored by the
      // latest-job guard (no fallback). A genuine pool failure falls back to the
      // single-worker path, which itself falls back to the main thread.
      root.dataset.computeMode = "pool";
      let cancelled = false;
      let fallbackJob = null;
      const engineJob = labEngine.run(input, onProgress);
      const promise = engineJob.promise.catch((error) => {
        if (cancelled) throw error;
        if (typeof console !== "undefined") {
          console.warn("KV Cache Lab pool failed; retrying on a single worker.", error && error.message);
        }
        fallbackJob = startWorkerJobWithSyncFallback(input, onProgress);
        return fallbackJob.promise;
      });
      return {
        promise,
        cancel() {
          cancelled = true;
          engineJob.cancel();
          if (fallbackJob) fallbackJob.cancel();
        },
      };
    }

    function applyResult(result, fromCache) {
      hideProgress();
      renderResults(root, result.preset || selectedPreset(), result.trace, result.sweep, metadata);
      renderTimeStats(root, result.timeStats);
      root.dataset.state = "ready";
      setStatus(root, fromCache ? "Cached" : result.sweep && result.sweep.precomputed ? "Precomputed" : "Ready");
    }

    function runUploadJob(preset) {
      if (!uploadState) {
        root.dataset.state = "ready";
        setStatus(root, "Ready");
        setText(root, "[data-lab-output='note']", "Upload a Mooncake-schema JSONL trace to compute its hit-rate curve.");
        return;
      }
      const model = selectedModel();
      const blockSize = currentUploadBlockSize();
      const settings = Object.assign(readSettings(blockSize), { computeCeiling: true });
      const maxEvents = UPLOAD_MAX_EVENTS;
      const useWasm =
        !!runtimeOptions.wasmUrl &&
        typeof WebAssembly === "object" &&
        typeof WebAssembly.instantiate === "function";
      const cacheKey = `upload:${uploadState.token}|bs=${blockSize}|cap=${maxEvents}|wasm=${useWasm ? 1 : 0}|${modelSweepKey(model, settings)}`;
      const jobId = latestJobId + 1;
      latestJobId = jobId;
      if (activeJob) {
        activeJob.cancel();
        activeJob = null;
      }
      if (resultCache.has(cacheKey)) {
        applyResult(resultCache.get(cacheKey), true);
        return;
      }
      root.dataset.state = "calculating";
      setStatus(root, "Calculating...");
      showProgress();
      activeJob = startJob(
        {
          jobId,
          cacheKey,
          preset,
          model,
          settings,
          uploadFile: uploadState.file,
          gzip: uploadState.isGzip,
          useWasm,
          wasmUrl: runtimeOptions.wasmUrl,
          uploadOptions: { blockSize, sourceId: "upload", label: uploadState.fileName, maxEvents },
        },
        (completed, total) => {
          if (jobId === latestJobId) setProgress(completed, total);
        },
      );
      activeJob.promise
        .then((result) => {
          if (!shouldApplyJobResult(latestJobId, result)) return;
          activeJob = null;
          hideProgress();
          rememberCachedResult(resultCache, cacheKey, result);
          try {
            applyResult(result, false);
          } catch (error) {
            failJob("render", error);
          }
        })
        .catch((error) => {
          if (jobId !== latestJobId) return;
          activeJob = null;
          failJob("compute", error);
        });
    }

    function update() {
      try {
        const activePreset = selectedPreset();
        if (activePreset.id === UPLOAD_PRESET_ID) {
          runUploadJob(activePreset);
          return;
        }
        const baseInput = computationInput();
        const precomputedResult = precomputedResultFor(precomputed, baseInput.preset, baseInput.model, baseInput.settings);
        if (precomputed && precomputed.traces) {
          if (precomputedResult) {
            if (activeJob) {
              activeJob.cancel();
              activeJob = null;
            }
            latestJobId += 1;
            applyResult(precomputedResult, false);
          } else {
            if (activeJob) {
              activeJob.cancel();
              activeJob = null;
            }
            latestJobId += 1;
            root.dataset.state = "error";
            setStatus(root, "Unavailable");
            renderUnavailable(root, baseInput.preset, baseInput.model, baseInput.settings);
          }
          return;
        }
        const cacheKey = createCacheKey(baseInput);
        const jobId = latestJobId + 1;
        latestJobId = jobId;
        if (activeJob) {
          activeJob.cancel();
          activeJob = null;
        }
        if (resultCache.has(cacheKey)) {
          applyResult(resultCache.get(cacheKey), true);
          return;
        }

        root.dataset.state = "calculating";
        setStatus(root, "Calculating...");
        showProgress();
        activeJob = startJob(Object.assign({ jobId, cacheKey }, baseInput), (completed, total) => {
          if (jobId === latestJobId) setProgress(completed, total);
        });
        activeJob.promise
          .then((result) => {
            if (!shouldApplyJobResult(latestJobId, result)) return;
            activeJob = null;
            hideProgress();
            rememberCachedResult(resultCache, cacheKey, result);
            try {
              applyResult(result, false);
            } catch (error) {
              failJob("render", error);
            }
          })
          .catch((error) => {
            if (jobId !== latestJobId) return;
            activeJob = null;
            failJob("compute", error);
          });
      } catch (error) {
        failJob("compute", error);
      }
    }

    function scheduleUpdate(delay) {
      if (debounceTimer) clearTimeout(debounceTimer);
      const wait = Math.max(0, delay || 0);
      if (wait === 0) {
        update();
        return;
      }
      debounceTimer = setTimeout(() => {
        debounceTimer = null;
        update();
      }, wait);
    }

    const defaultModel = models[0];
    populateFamilies(modelFamily(defaultModel));
    setSelectOptions(inputs.preset, presets, presets[0].id, (preset) => preset.label);
    populateModels(defaultModel.id);
    syncModelControls();
    applyPresetDefaults();
    setupTraceHelp();
    syncTraceHelp();

    inputs.modelFamily.addEventListener("change", () => {
      populateModels();
      syncModelControls();
      scheduleUpdate(0);
    });
    inputs.model.addEventListener("change", () => {
      syncModelControls();
      scheduleUpdate(0);
    });
    inputs.preset.addEventListener("change", () => {
      applyPresetDefaults();
      populateFamilies(inputs.modelFamily.value);
      populateModels(inputs.model.value);
      syncModelControls();
      syncTraceHelp();
      scheduleUpdate(0);
    });
    Object.values(inputs).forEach((input) => {
      if (!input || input === inputs.modelFamily || input === inputs.model || input === inputs.preset) return;
      input.addEventListener("input", () => scheduleUpdate(INPUT_DEBOUNCE_MS));
      input.addEventListener("change", () => scheduleUpdate(0));
    });
    paramInputs.forEach((input) => {
      input.addEventListener("input", () => scheduleUpdate(INPUT_DEBOUNCE_MS));
      input.addEventListener("change", () => scheduleUpdate(0));
    });

    if (uploadInput) {
      uploadInput.addEventListener("change", () => readUploadFile(uploadInput.files && uploadInput.files[0]));
    }
    if (uploadClear) {
      uploadClear.addEventListener("click", clearUpload);
    }
    if (uploadBlockSizeInput) {
      uploadBlockSizeInput.addEventListener("input", () => {
        if (uploadState && inputs.preset.value === UPLOAD_PRESET_ID) scheduleUpdate(INPUT_DEBOUNCE_MS);
      });
    }
    if (uploadZone) {
      ["dragenter", "dragover"].forEach((eventName) =>
        uploadZone.addEventListener(eventName, (event) => {
          event.preventDefault();
          uploadZone.dataset.dragging = "true";
        }),
      );
      ["dragleave", "drop"].forEach((eventName) =>
        uploadZone.addEventListener(eventName, (event) => {
          event.preventDefault();
          delete uploadZone.dataset.dragging;
        }),
      );
      uploadZone.addEventListener("drop", (event) => {
        const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
        readUploadFile(file);
      });
    }

    scheduleUpdate(0);
  }

  function mount(rootId, data, options) {
    const rootNode = document.getElementById(rootId);
    const runtimeOptions = options || {};
    if (
      runtimeOptions.precomputedUrl &&
      data &&
      !data.precomputed &&
      typeof fetch === "function"
    ) {
      if (rootNode) {
        rootNode.dataset.state = "loading";
        setStatus(rootNode, "Loading trace data...");
      }
      fetch(runtimeOptions.precomputedUrl)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Failed to load precomputed trace data (${response.status})`);
          }
          return response.json();
        })
        .then((precomputed) => {
          initialize(rootNode, Object.assign({}, data, { precomputed }), runtimeOptions);
        })
        .catch((error) => {
          if (!rootNode) return;
          rootNode.dataset.state = "error";
          setStatus(rootNode, "Trace data failed");
          setText(rootNode, "[data-lab-output='note']", error.message);
        });
      return;
    }
    initialize(rootNode, data, runtimeOptions);
  }

  return {
    BYTES_PER_GIB,
    DEFAULT_CAPACITY_GIB_VALUES,
    POLICY_LABELS,
    UPLOAD_PRESET_ID,
    analyzeTrace,
    assembleSweep,
    blockTokens,
    buildExecutionPlan,
    buildTimeSeriesResult,
    cacheBlocksForGiB,
    computeTimeSeries,
    createCacheKey,
    createLabEngine,
    estimateBytesPerToken,
    extractPlanBuffers,
    generateTrace,
    infiniteCacheReuse,
    mount,
    modelSweepKey,
    namespacedBlocks,
    normalizeMooncakeRecord,
    createTraceTextStream,
    inspectUploadedTraceHeadText,
    parseUploadedTrace,
    parseUploadedTraceStreaming,
    planFromBuffers,
    planSweepTasks,
    precomputedResultFor,
    runLabComputation,
    simulatePlanPolicy,
    simulatePolicy,
    shouldApplyJobResult,
    sweepCapacity,
    throughputFromHitRate,
  };
});
