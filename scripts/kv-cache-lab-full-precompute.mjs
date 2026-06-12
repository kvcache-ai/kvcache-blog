#!/usr/bin/env node
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import { once } from "node:events";
import { createRequire } from "node:module";
import { execFile, execFileSync } from "node:child_process";
import { promisify } from "node:util";

import {
  DEFAULT_CAPACITY_GIB_VALUES,
  DEFAULT_OUTPUT_PATH,
  DEFAULT_TRACE_CACHE_DIR,
  DEFAULT_WARMUP_FRACTION,
  POLICIES,
  REFERENCE_SOURCES,
  SOURCE_LINKS,
  TRACE_SOURCES,
  allModelSettings,
  blockCapacityGrid,
  downloadFile,
  filterPrecisionOptions,
  kvArchitectureGroups,
  loadModelsData,
  modelSweepKey,
  parseArgs as parseCommonArgs,
  selectModels,
} from "./lib/kv-cache-lab-traces.mjs";

const require = createRequire(import.meta.url);
const lab = require("../assets/js/kv-cache-lab.js");
const execFileAsync = promisify(execFile);

const FULL_TRACE_IDS = new Set([
  "bailian_qwen_trace_a",
  "semianalysis_weka_no_subagents",
  "semianalysis_weka_with_subagents_256k",
  "kv_cache_tester_claude_code",
]);

const HF_EXPECTED_ROWS = {
  semianalysis_weka_no_subagents: 949,
  semianalysis_weka_with_subagents_256k: 470,
};

const UINT32_MAX = 0xffffffff;
const EVENT_CHUNK = 1_000_000;
const SIMULATION_RESULT_CACHE_DIR = "results-context-window-v1";

function toNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function positiveInteger(value, fallback) {
  const parsed = Math.floor(toNumber(value, fallback));
  return parsed > 0 ? parsed : fallback;
}

function blockTokens(inputLength, blockSize, index, count) {
  if (count <= 0) return 0;
  if (!Number.isFinite(inputLength) || inputLength <= 0) return blockSize;
  const remaining = inputLength - index * blockSize;
  if (remaining <= 0) return 1;
  return Math.max(1, Math.min(blockSize, remaining));
}

function parseArgs(argv) {
  const args = parseCommonArgs(argv);
  args.outputPath = args.outputPath || DEFAULT_OUTPUT_PATH;
  args.fullCacheDir = path.join(args.cacheDir || DEFAULT_TRACE_CACHE_DIR, "kv-cache-lab-full");
  args.capacityGiBValues = DEFAULT_CAPACITY_GIB_VALUES;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--full-cache-dir") args.fullCacheDir = argv[++index];
    else if (arg === "--force-events") args.forceEvents = true;
    else if (arg === "--force-next") args.forceNext = true;
    else if (arg === "--events-only") args.eventsOnly = true;
    else if (arg === "--default-settings") args.defaultSettings = true;
    else if (arg === "--block-curve") args.blockCurve = true;
    else if (arg === "--curve-points") args.curvePoints = Math.max(2, Math.floor(Number(argv[++index])));
    else if (arg === "--native-sim") args.nativeSimPath = argv[++index];
    else if (arg === "--native-jobs") args.nativeJobs = Math.max(1, Math.floor(Number(argv[++index])));
    else if (arg === "--native-serial-capacity-threshold") args.nativeSerialCapacityThreshold = Math.max(0, Math.floor(Number(argv[++index])));
    else if (arg === "--capacity-gib-values") {
      args.capacityGiBValues = argv[++index]
        .split(",")
        .map((value) => Number(value.trim()))
        .filter((value) => Number.isFinite(value) && value > 0);
    }
  }
  return args;
}

function sourceById(id) {
  const source = TRACE_SOURCES.find((candidate) => candidate.id === id);
  if (!source) throw new Error(`Unknown trace id: ${id}`);
  if (!FULL_TRACE_IDS.has(id)) throw new Error(`${id} is not a full-trace target`);
  return { ...source, requestLimit: undefined, rowLimit: undefined, maxRequestsPerSession: undefined };
}

async function writeBuffer(stream, buffer) {
  if (!stream.write(buffer)) await once(stream, "drain");
}

async function countLines(filePath) {
  if (!fs.existsSync(filePath)) return 0;
  let count = 0;
  const rl = readline.createInterface({ input: fs.createReadStream(filePath), crlfDelay: Infinity });
  for await (const line of rl) {
    if (line.trim()) count += 1;
  }
  return count;
}

async function appendFileToStream(filePath, writer) {
  await new Promise((resolve, reject) => {
    const reader = fs.createReadStream(filePath);
    reader.on("error", reject);
    reader.on("data", (chunk) => {
      if (!writer.write(chunk)) {
        reader.pause();
        writer.once("drain", () => reader.resume());
      }
    });
    reader.on("end", resolve);
  });
}

async function fetchJsonWithTimeout(url, timeoutMs = 30_000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  } finally {
    clearTimeout(timeout);
  }
}

function sqlLiteral(value) {
  return `'${String(value).replaceAll("'", "''")}'`;
}

async function fetchHfParquetRowsToCache(source, options) {
  const cacheDir = options.cacheDir || DEFAULT_TRACE_CACHE_DIR;
  await fsp.mkdir(cacheDir, { recursive: true });
  const cachePath = path.join(cacheDir, `${source.id}.hf-rows.jsonl`);
  const expectedRows = HF_EXPECTED_ROWS[source.id] || 0;
  if (!options.force && expectedRows > 0 && await countLines(cachePath) >= expectedRows) {
    return cachePath;
  }

  const parquetApi = new URL("https://datasets-server.huggingface.co/parquet");
  parquetApi.searchParams.set("dataset", source.hfDataset);
  const payload = await fetchJsonWithTimeout(parquetApi);
  const files = (Array.isArray(payload.parquet_files) ? payload.parquet_files : [])
    .filter((file) =>
      file.config === (source.hfConfig || "default") &&
      file.split === (source.hfSplit || "train") &&
      file.url,
    );
  if (!files.length) throw new Error(`No parquet files found for ${source.id}`);

  const parquetDir = path.join(cacheDir, `${source.id}-parquet`);
  const duckTmpDir = path.join(parquetDir, "duckdb-tmp");
  await fsp.mkdir(parquetDir, { recursive: true });
  await fsp.mkdir(duckTmpDir, { recursive: true });
  const writer = fs.createWriteStream(cachePath, { flags: "w" });
  try {
    for (const file of files) {
      const shardName = file.filename || path.basename(new URL(file.url).pathname);
      const parquetPath = path.join(parquetDir, shardName);
      const jsonPath = `${parquetPath}.jsonl`;
      if (!fs.existsSync(parquetPath) || options.force) {
        console.error(`[fetch] ${source.id}: download parquet ${shardName}`);
        await downloadFile(file.url, parquetPath, { force: options.force });
      }
      if (!fs.existsSync(jsonPath) || options.force) {
        const temporaryJsonPath = `${jsonPath}.${process.pid}.tmp`;
        await fsp.rm(temporaryJsonPath, { force: true });
        console.error(`[fetch] ${source.id}: convert parquet ${shardName}`);
        execFileSync(
          "npx",
          [
            "-y",
            "-p",
            "parquetlens",
            "-p",
            "@parquetlens/sql",
            "parquetlens",
            parquetPath,
            "--sql",
            [
              `SET temp_directory=${sqlLiteral(duckTmpDir)}`,
              "SET preserve_insertion_order=false",
              "SET threads=1",
              "SET memory_limit='8GB'",
              `COPY (SELECT * FROM data) TO ${sqlLiteral(temporaryJsonPath)} (FORMAT JSON)`,
            ].join("; "),
          ],
          { stdio: "inherit", maxBuffer: 20 * 1024 * 1024 },
        );
        const actualJsonPath = fs.existsSync(temporaryJsonPath)
          ? temporaryJsonPath
          : temporaryJsonPath.endsWith(".tmp") && fs.existsSync(temporaryJsonPath.slice(0, -4))
            ? temporaryJsonPath.slice(0, -4)
            : temporaryJsonPath;
        await fsp.rename(actualJsonPath, jsonPath);
      }
      await appendFileToStream(jsonPath, writer);
      await writeBuffer(writer, Buffer.from("\n"));
    }
  } finally {
    await new Promise((resolve) => writer.end(resolve));
  }

  const rows = await countLines(cachePath);
  if (expectedRows > 0 && rows < expectedRows) {
    throw new Error(`Parquet conversion for ${source.id} produced ${rows} rows, expected ${expectedRows}`);
  }
  return cachePath;
}

async function fetchHfRowsToCache(source, options) {
  if (source.parser === "weka_session" && !options.forceRowsApi) {
    return fetchHfParquetRowsToCache(source, options);
  }
  const cacheDir = options.cacheDir || DEFAULT_TRACE_CACHE_DIR;
  await fsp.mkdir(cacheDir, { recursive: true });
  const cachePath = path.join(cacheDir, `${source.id}.hf-rows.jsonl`);
  let offset = options.force ? 0 : await countLines(cachePath);
  const expectedRows = HF_EXPECTED_ROWS[source.id] || 0;
  if (!options.force && expectedRows > 0 && offset >= expectedRows) {
    return cachePath;
  }
  const writer = fs.createWriteStream(cachePath, { flags: offset > 0 && !options.force ? "a" : "w" });
  let singleRowMode = false;
  for (;;) {
    if (expectedRows > 0 && offset >= expectedRows) break;
    const url = new URL("https://datasets-server.huggingface.co/rows");
    url.searchParams.set("dataset", source.hfDataset);
    url.searchParams.set("config", source.hfConfig || "default");
    url.searchParams.set("split", source.hfSplit || "train");
    const remainingRows = expectedRows > 0 ? Math.max(1, expectedRows - offset) : 100;
    const pageSize = Math.min(
      remainingRows,
      singleRowMode ? 1 : Math.max(1, Math.floor(Number(source.hfPageSize || 0)) > 1 ? Number(source.hfPageSize) : 25),
    );
    url.searchParams.set("offset", String(offset));
    let response = null;
    let fetchError = null;
    let attemptPageSize = pageSize;
    for (let attempt = 0; attempt < 12; attempt += 1) {
      attemptPageSize = attempt >= 3 ? 1 : pageSize;
      const requestUrl = new URL(url);
      requestUrl.searchParams.set("length", String(attemptPageSize));
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 30_000);
        try {
          response = await fetch(requestUrl, { signal: controller.signal });
        } finally {
          clearTimeout(timeout);
        }
        fetchError = null;
      } catch (error) {
        response = null;
        fetchError = error;
      }
      if (response && response.ok) break;
      const retryAfter = Number(response && response.headers ? response.headers.get("retry-after") : NaN);
      const delayMs = response && response.status === 429
        ? Math.max(60_000, Number.isFinite(retryAfter) ? retryAfter * 1000 : Math.min(120_000, 1000 * 2 ** attempt))
        : Number.isFinite(retryAfter)
          ? Math.max(1000, retryAfter * 1000)
          : Math.min(120_000, 1000 * 2 ** attempt);
      console.error(`[fetch] ${source.id}: row ${offset} length ${attemptPageSize} HTTP ${response ? response.status : (fetchError && fetchError.name) || "network"}, retry in ${Math.round(delayMs / 1000)}s`);
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    if (!response || !response.ok) {
      await new Promise((resolve) => writer.end(resolve));
      if (fetchError) throw fetchError;
      throw new Error(`Failed to fetch HF row ${offset} for ${source.id}: HTTP ${response ? response.status : "network"}`);
    }
    const payload = await response.json();
    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    if (!rows.length) break;
    for (const item of rows) {
      await writeBuffer(writer, Buffer.from(`${JSON.stringify(item.row)}\n`));
    }
    if (attemptPageSize === 1 && pageSize > 1) singleRowMode = true;
    offset += rows.length;
    console.error(`[fetch] ${source.id}: cached ${offset} rows`);
  }
  await new Promise((resolve) => writer.end(resolve));
  return cachePath;
}

async function ensureGitRepository(source, options) {
  const cacheDir = options.cacheDir || DEFAULT_TRACE_CACHE_DIR;
  const repoDir = path.join(cacheDir, source.localDir || source.id);
  if (!fs.existsSync(repoDir)) {
    await fsp.mkdir(cacheDir, { recursive: true });
    execFileSync("git", ["clone", "--depth=1", source.gitUrl, repoDir], { stdio: "inherit" });
  }
  return repoDir;
}

function parseWekaNestedRecord(record) {
  if (typeof record !== "string") return record;
  try {
    return JSON.parse(record);
  } catch {
    return null;
  }
}

function flattenWekaRequests(records, output, timestampOffset = 0) {
  (records || []).forEach((rawRecord) => {
    const record = parseWekaNestedRecord(rawRecord);
    if (record && Array.isArray(record.hash_ids) && record.hash_ids.length) {
      output.push({
        timestamp: timestampOffset + toNumber(record.t ?? record.timestamp, 0),
        inputLength: positiveInteger(record.in ?? record.input_length, record.hash_ids.length * 64),
        outputTokens: Math.max(0, Math.floor(toNumber(record.out ?? record.output_length, 0))),
        hashIds: record.hash_ids,
      });
    }
    if (Array.isArray(record && record.requests)) {
      flattenWekaRequests(record.requests, output, timestampOffset + toNumber(record.t, 0));
    }
  });
}

class RequestEventWriter {
  constructor(source, traceDir) {
    this.source = source;
    this.traceDir = traceDir;
    this.blockSize = positiveInteger(source.nativeBlockSize, 64);
    this.requestIdsPath = path.join(traceDir, "request-ids.bin");
    this.requestIdStream = fs.createWriteStream(this.requestIdsPath);
    this.descriptors = [];
    this.trackGlobalUniqueIds = source.id === "bailian_qwen_trace_a";
    this.uniqueIds = this.trackGlobalUniqueIds ? new Set() : null;
    this.uniqueIdCount = 0;
    this.rawEventCount = 0;
    this.requestCount = 0;
    this.outputTokens = 0;
    this.nextBase = 1;
  }

  async appendRequest(hashIds, inputLength, timestamp, namespaceBase = 0, ordinal = this.requestCount) {
    if (!Array.isArray(hashIds) || !hashIds.length) return;
    const count = hashIds.length;
    const buffer = Buffer.allocUnsafe(count * 4);
    for (let index = 0; index < count; index += 1) {
      const id = namespaceBase + positiveInteger(hashIds[index], 0);
      if (id > UINT32_MAX) throw new Error(`${this.source.id} generated block id ${id}, which does not fit uint32`);
      buffer.writeUInt32LE(id, index * 4);
      if (this.uniqueIds) this.uniqueIds.add(id);
    }
    await writeBuffer(this.requestIdStream, buffer);
    this.descriptors.push({
      timestamp: toNumber(timestamp, 0),
      ordinal,
      offset: this.rawEventCount,
      count,
      inputLength: positiveInteger(inputLength, count * this.blockSize),
    });
    this.rawEventCount += count;
    this.requestCount += 1;
  }

  allocateNamespace(maxHashId) {
    const base = this.nextBase;
    this.nextBase += positiveInteger(maxHashId, 0) + 1;
    if (this.nextBase > UINT32_MAX) throw new Error(`${this.source.id} block namespace exceeded uint32`);
    return base;
  }

  addNamespacedUniqueCount(count) {
    this.uniqueIdCount += Math.max(0, Math.floor(Number(count) || 0));
  }

  async close() {
    await new Promise((resolve) => this.requestIdStream.end(resolve));
    if (this.uniqueIds) this.uniqueIdCount = this.uniqueIds.size;
  }
}

function requestHashStats(requests) {
  let maxHashId = 0;
  const unique = new Set();
  for (const request of requests) {
    for (const rawId of request.hashIds) {
      const id = positiveInteger(rawId, 0);
      if (id > maxHashId) maxHashId = id;
      unique.add(id);
    }
  }
  return { maxHashId, uniqueCount: unique.size };
}

async function readJsonlRecords(filePath, onRecord) {
  const rl = readline.createInterface({ input: fs.createReadStream(filePath), crlfDelay: Infinity });
  for await (const line of rl) {
    if (line.trim()) await onRecord(JSON.parse(line));
  }
}

async function buildRequestStore(source, options, traceDir) {
  const writer = new RequestEventWriter(source, traceDir);
  let ordinal = 0;
  if (source.id === "bailian_qwen_trace_a") {
    const rawPath = path.join(options.cacheDir || DEFAULT_TRACE_CACHE_DIR, source.localFile || `${source.id}.jsonl`);
    if (!fs.existsSync(rawPath) || options.force) await downloadFile(source.url, rawPath, { force: options.force });
    await readJsonlRecords(rawPath, async (record) => {
      await writer.appendRequest(record.hash_ids, record.input_length, record.timestamp, 0, ordinal);
      ordinal += 1;
    });
  } else if (source.id === "kv_cache_tester_claude_code") {
    const repoDir = await ensureGitRepository(source, options);
    const tracePath = path.join(repoDir, source.traceDir || "");
    const files = (await fsp.readdir(tracePath)).filter((file) => file.endsWith(".json")).sort();
    for (const file of files) {
      const record = JSON.parse(await fsp.readFile(path.join(tracePath, file), "utf8"));
      const requests = [];
      flattenWekaRequests(record.requests || [], requests);
      requests.sort((left, right) => left.timestamp - right.timestamp);
      const stats = requestHashStats(requests);
      const base = writer.allocateNamespace(stats.maxHashId);
      writer.addNamespacedUniqueCount(stats.uniqueCount);
      for (const request of requests) {
        await writer.appendRequest(request.hashIds, request.inputLength, request.timestamp, base, ordinal);
        ordinal += 1;
      }
    }
  } else if (source.parser === "weka_session") {
    const cachePath = await fetchHfRowsToCache(source, options);
    await readJsonlRecords(cachePath, async (record) => {
      const requests = [];
      flattenWekaRequests(record.requests || [], requests);
      requests.sort((left, right) => left.timestamp - right.timestamp);
      const stats = requestHashStats(requests);
      const base = writer.allocateNamespace(stats.maxHashId);
      writer.addNamespacedUniqueCount(stats.uniqueCount);
      for (const request of requests) {
        await writer.appendRequest(request.hashIds, request.inputLength, request.timestamp, base, ordinal);
        ordinal += 1;
      }
    });
  } else {
    throw new Error(`Unsupported full-trace source: ${source.id}`);
  }
  await writer.close();
  return writer;
}

async function createSortedEventFiles(writer, traceDir, options) {
  const idsPath = path.join(traceDir, "ids.bin");
  const tokensPath = path.join(traceDir, "tokens.u16.bin");
  const requestEndsPath = path.join(traceDir, "request-ends.u32.bin");
  const metadataPath = path.join(traceDir, "events.json");
  writer.descriptors.sort((left, right) => left.timestamp - right.timestamp || left.ordinal - right.ordinal);

  const warmupRequests = Math.min(
    writer.descriptors.length,
    Math.max(0, Math.floor(writer.descriptors.length * toNumber(options.warmupFraction, DEFAULT_WARMUP_FRACTION))),
  );
  const readHandle = await fsp.open(writer.requestIdsPath, "r");
  const idStream = fs.createWriteStream(idsPath);
  const tokenStream = fs.createWriteStream(tokensPath);
  const requestEndStream = fs.createWriteStream(requestEndsPath);
  let totalBlocks = 0;
  let totalInputTokens = 0;
  let warmupEventStart = 0;

  for (let requestIndex = 0; requestIndex < writer.descriptors.length; requestIndex += 1) {
    const descriptor = writer.descriptors[requestIndex];
    if (requestIndex === warmupRequests) warmupEventStart = totalBlocks;
    const idBuffer = Buffer.allocUnsafe(descriptor.count * 4);
    await readHandle.read(idBuffer, 0, idBuffer.length, descriptor.offset * 4);
    await writeBuffer(idStream, idBuffer);
    const tokenBuffer = Buffer.allocUnsafe(descriptor.count * 2);
    for (let index = 0; index < descriptor.count; index += 1) {
      const tokens = blockTokens(descriptor.inputLength, writer.blockSize, index, descriptor.count);
      tokenBuffer.writeUInt16LE(tokens, index * 2);
      totalInputTokens += tokens;
    }
    await writeBuffer(tokenStream, tokenBuffer);
    totalBlocks += descriptor.count;
    const requestEndBuffer = Buffer.allocUnsafe(4);
    requestEndBuffer.writeUInt32LE(totalBlocks, 0);
    await writeBuffer(requestEndStream, requestEndBuffer);
  }
  if (warmupRequests === writer.descriptors.length) warmupEventStart = totalBlocks;
  await readHandle.close();
  await new Promise((resolve) => idStream.end(resolve));
  await new Promise((resolve) => tokenStream.end(resolve));
  await new Promise((resolve) => requestEndStream.end(resolve));

  const metadata = {
    id: writer.source.id,
    label: writer.source.label,
    scenario: writer.source.scenario,
    sourceKind: writer.source.sourceKind,
    blockSize: writer.blockSize,
    sourceBlockSizeNote: writer.source.sourceBlockSizeNote,
    sources: writer.source.sources || [],
    requestCount: writer.descriptors.length,
    warmupRequests,
    warmupFraction: toNumber(options.warmupFraction, DEFAULT_WARMUP_FRACTION),
    warmupEventStart,
    totalBlocks,
    totalInputTokens,
    averageInputTokens: writer.descriptors.length ? totalInputTokens / writer.descriptors.length : 0,
    uniqueBlocks: writer.uniqueIdCount,
    idsPath,
    tokensPath,
    requestEndsPath,
    nextPath: path.join(traceDir, "next.u32.bin"),
    generatedAt: new Date().toISOString(),
  };
  await fsp.writeFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);
  return metadata;
}

async function ensureEventStream(source, options) {
  const traceDir = path.join(options.fullCacheDir || path.join(os.tmpdir(), "kv-cache-lab-full"), source.id);
  const metadataPath = path.join(traceDir, "events.json");
  const applyWarmup = (metadata) => {
    const warmupRequests = Math.min(
      metadata.requestCount,
      Math.max(0, Math.floor(metadata.requestCount * toNumber(options.warmupFraction, DEFAULT_WARMUP_FRACTION))),
    );
    metadata.warmupRequests = warmupRequests;
    metadata.warmupFraction = toNumber(options.warmupFraction, DEFAULT_WARMUP_FRACTION);
    return metadata;
  };
  if (!options.forceEvents && fs.existsSync(metadataPath)) {
    const metadata = applyWarmup(JSON.parse(await fsp.readFile(metadataPath, "utf8")));
    if (metadata.requestEndsPath && fs.existsSync(metadata.requestEndsPath)) return metadata;
  }
  await fsp.rm(traceDir, { recursive: true, force: true });
  await fsp.mkdir(traceDir, { recursive: true });
  console.error(`[full] ${source.id}: building request store`);
  const writer = await buildRequestStore(source, options, traceDir);
  console.error(`[full] ${source.id}: ${writer.requestCount} requests, ${writer.rawEventCount} raw block events, ${writer.uniqueIdCount} unique`);
  return applyWarmup(await createSortedEventFiles(writer, traceDir, options));
}

async function scanEvents(metadata, withNext, onChunk) {
  const idHandle = await fsp.open(metadata.idsPath, "r");
  const tokenHandle = await fsp.open(metadata.tokensPath, "r");
  const nextHandle = withNext ? await fsp.open(metadata.nextPath, "r") : null;
  try {
    for (let start = 0; start < metadata.totalBlocks; start += EVENT_CHUNK) {
      const count = Math.min(EVENT_CHUNK, metadata.totalBlocks - start);
      const ids = Buffer.allocUnsafe(count * 4);
      const tokens = Buffer.allocUnsafe(count * 2);
      const next = withNext ? Buffer.allocUnsafe(count * 4) : null;
      await idHandle.read(ids, 0, ids.length, start * 4);
      await tokenHandle.read(tokens, 0, tokens.length, start * 2);
      if (withNext) await nextHandle.read(next, 0, next.length, start * 4);
      await onChunk({ start, count, ids, tokens, next });
    }
  } finally {
    await idHandle.close();
    await tokenHandle.close();
    if (nextHandle) await nextHandle.close();
  }
}

async function readRequestEnds(metadata) {
  if (!metadata.requestEndsPath) throw new Error(`${metadata.id} is missing requestEndsPath; rebuild event stream with scripts/kv-cache-lab-full-precompute.mjs`);
  const buffer = await fsp.readFile(metadata.requestEndsPath);
  if (buffer.length !== metadata.requestCount * 4) throw new Error(`${metadata.id} request-ends file has unexpected size`);
  const ends = new Uint32Array(metadata.requestCount);
  for (let index = 0; index < metadata.requestCount; index += 1) {
    ends[index] = buffer.readUInt32LE(index * 4);
  }
  return ends;
}

function makeRequestCursor(requestEnds) {
  let requestIndex = 0;
  return {
    current(eventIndex) {
      while (requestIndex < requestEnds.length && eventIndex >= requestEnds[requestIndex]) {
        requestIndex += 1;
      }
      return requestIndex;
    },
  };
}

function makeRequestSampler(metadata, requestEnds, capacity) {
  let requestIndex = 0;
  let usefulCacheBlockSamples = 0;
  let usefulCacheSamples = 0;
  return {
    sample(eventIndex, usefulCount) {
      const eventEnd = eventIndex + 1;
      while (requestIndex < requestEnds.length && eventEnd >= requestEnds[requestIndex]) {
        if (requestIndex >= metadata.warmupRequests) {
          usefulCacheBlockSamples += usefulCount;
          usefulCacheSamples += 1;
        }
        requestIndex += 1;
      }
    },
    finish(hitTokens, totalTokens) {
      return {
        hitTokens,
        totalTokens,
        hitRate: totalTokens ? hitTokens / totalTokens : 0,
        usefulCacheBlockSamples,
        usefulCacheSamples,
        usefulCacheRate: capacity > 0 && usefulCacheSamples > 0 ? usefulCacheBlockSamples / (usefulCacheSamples * capacity) : 0,
      };
    },
  };
}

async function computeCeiling(metadata, options = {}) {
  if (options.nativeSimPath) {
    console.error(`[full] ${metadata.id}: computing infinite-cache ceiling with native helper`);
    const nativeResult = await simulateNativePolicy(metadata, Math.max(1, metadata.uniqueBlocks), "ceiling", options);
    if (Number.isFinite(Number(nativeResult.trieNodeCount))) {
      metadata.trieNodeCount = Number(nativeResult.trieNodeCount);
      metadata.uniqueBlocks = Math.max(0, metadata.trieNodeCount - 1);
    }
    return { warmupRequests: metadata.warmupRequests, ...nativeResult };
  }

  await ensureNextFile(metadata, options);
  const seen = new Map();
  let hitTokens = 0;
  let totalTokens = 0;
  await scanEvents(metadata, true, ({ start, count, ids, tokens, next }) => {
    for (let index = 0; index < count; index += 1) {
      const id = ids.readUInt32LE(index * 4);
      const tokenCount = tokens.readUInt16LE(index * 2);
      const nextUse = next.readUInt32LE(index * 4);
      const hit = seen.has(id);
      totalTokens += tokenCount;
      if (hit) hitTokens += tokenCount;
      seen.set(id, nextUse);
    }
  });
  return { warmupRequests: 0, hitTokens, totalTokens, hitRate: totalTokens ? hitTokens / totalTokens : 0 };
}

function policyWorkingSet(metadata, policy) {
  const trieNodeCount = Number(metadata.trieNodeCount);
  if (Number.isFinite(trieNodeCount) && trieNodeCount > 0) return Math.max(0, trieNodeCount - 1);
  return Math.max(0, Number(metadata.uniqueBlocks) || 0);
}

async function ensureNextFile(metadata, options) {
  const expectedBytes = metadata.totalBlocks * 4;
  if (!options.forceNext && fs.existsSync(metadata.nextPath)) {
    const stat = await fsp.stat(metadata.nextPath);
    if (stat.size === expectedBytes) return;
  }
  if (metadata.totalBlocks + 1 > UINT32_MAX) throw new Error(`${metadata.id} has too many block events for uint32 next-use positions`);
  if (options.nativeSimPath) {
    console.error(`[full] ${metadata.id}: building next-use file with native helper`);
    await execFileAsync(path.resolve(options.nativeSimPath), [
      "--policy",
      "build-next",
      "--ids",
      metadata.idsPath,
      "--tokens",
      metadata.tokensPath,
      "--total-blocks",
      String(metadata.totalBlocks),
      "--warmup-event-start",
      String(metadata.warmupEventStart),
      "--next",
      metadata.nextPath,
    ], {
      encoding: "utf8",
      maxBuffer: 1024 * 1024,
    });
    options.forceNext = false;
    return;
  }
  console.error(`[full] ${metadata.id}: building next-use file`);
  const idHandle = await fsp.open(metadata.idsPath, "r");
  const nextHandle = await fsp.open(metadata.nextPath, "w");
  const lastSeen = new Map();
  const never = metadata.totalBlocks + 1;
  try {
    await nextHandle.truncate(metadata.totalBlocks * 4);
    for (let end = metadata.totalBlocks; end > 0; end -= EVENT_CHUNK) {
      const start = Math.max(0, end - EVENT_CHUNK);
      const count = end - start;
      const ids = Buffer.allocUnsafe(count * 4);
      const next = Buffer.allocUnsafe(count * 4);
      await idHandle.read(ids, 0, ids.length, start * 4);
      for (let index = count - 1; index >= 0; index -= 1) {
        const eventIndex = start + index;
        const id = ids.readUInt32LE(index * 4);
        next.writeUInt32LE(lastSeen.has(id) ? lastSeen.get(id) : never, index * 4);
        lastSeen.set(id, eventIndex);
      }
      await nextHandle.write(next, 0, next.length, start * 4);
      if (start % (EVENT_CHUNK * 20) === 0) console.error(`[full] ${metadata.id}: next-use ${start}/${metadata.totalBlocks}`);
    }
  } finally {
    await idHandle.close();
    await nextHandle.close();
  }
  options.forceNext = false;
}

class MaxHeap {
  constructor() {
    this.items = [];
  }
  push(item) {
    this.items.push(item);
    let index = this.items.length - 1;
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (this.items[parent].nextUse >= this.items[index].nextUse) break;
      [this.items[parent], this.items[index]] = [this.items[index], this.items[parent]];
      index = parent;
    }
  }
  pop() {
    if (!this.items.length) return null;
    const top = this.items[0];
    const last = this.items.pop();
    if (this.items.length && last) {
      this.items[0] = last;
      let index = 0;
      for (;;) {
        const left = index * 2 + 1;
        const right = left + 1;
        let largest = index;
        if (left < this.items.length && this.items[left].nextUse > this.items[largest].nextUse) largest = left;
        if (right < this.items.length && this.items[right].nextUse > this.items[largest].nextUse) largest = right;
        if (largest === index) break;
        [this.items[largest], this.items[index]] = [this.items[index], this.items[largest]];
        index = largest;
      }
    }
    return top;
  }
}

async function simulateFifo(metadata, capacity) {
  if (capacity <= 0) return { hitTokens: 0, totalTokens: metadata.totalInputTokens || 0, hitRate: 0 };
  await ensureNextFile(metadata, {});
  const requestEnds = await readRequestEnds(metadata);
  const requestCursor = makeRequestCursor(requestEnds);
  const cache = new Set();
  let queue = [];
  let head = 0;
  let hitTokens = 0;
  let totalTokens = 0;
  let measurementStartRequest = -1;
  await scanEvents(metadata, true, ({ start, count, ids, tokens, next }) => {
    for (let index = 0; index < count; index += 1) {
      const eventIndex = start + index;
      const requestIndex = requestCursor.current(eventIndex);
      const id = ids.readUInt32LE(index * 4);
      const tokenCount = tokens.readUInt16LE(index * 2);
      const hit = cache.has(id);
      if (measurementStartRequest >= 0 && requestIndex >= measurementStartRequest) {
        totalTokens += tokenCount;
        if (hit) hitTokens += tokenCount;
      }
      if (!hit) {
        while (cache.size >= capacity && head < queue.length) {
          if (measurementStartRequest < 0) measurementStartRequest = requestIndex + 1;
          const victim = queue[head];
          head += 1;
          if (cache.delete(victim)) break;
        }
        if (cache.size < capacity) {
          cache.add(id);
          queue.push(id);
        }
      }
      if (head > 1_000_000 && head * 2 > queue.length) {
        queue = queue.slice(head);
        head = 0;
      }
    }
  });
  if (measurementStartRequest < 0 || totalTokens <= 0) return computeCeiling(metadata);
  return { hitTokens, totalTokens, hitRate: totalTokens ? hitTokens / totalTokens : 0 };
}

async function simulateLru(metadata, capacity) {
  if (capacity <= 0) return { hitTokens: 0, totalTokens: metadata.totalInputTokens || 0, hitRate: 0 };
  await ensureNextFile(metadata, {});
  const requestEnds = await readRequestEnds(metadata);
  const requestCursor = makeRequestCursor(requestEnds);
  const cache = new Map();
  let hitTokens = 0;
  let totalTokens = 0;
  let measurementStartRequest = -1;
  await scanEvents(metadata, true, ({ start, count, ids, tokens, next }) => {
    for (let index = 0; index < count; index += 1) {
      const eventIndex = start + index;
      const requestIndex = requestCursor.current(eventIndex);
      const id = ids.readUInt32LE(index * 4);
      const tokenCount = tokens.readUInt16LE(index * 2);
      const hit = cache.has(id);
      if (measurementStartRequest >= 0 && requestIndex >= measurementStartRequest) {
        totalTokens += tokenCount;
        if (hit) hitTokens += tokenCount;
      }
      if (hit) {
        cache.delete(id);
      } else {
        while (cache.size >= capacity) {
          if (measurementStartRequest < 0) measurementStartRequest = requestIndex + 1;
          const victim = cache.keys().next().value;
          cache.delete(victim);
        }
      }
      cache.set(id, true);
    }
  });
  if (measurementStartRequest < 0 || totalTokens <= 0) return computeCeiling(metadata);
  return { hitTokens, totalTokens, hitRate: totalTokens ? hitTokens / totalTokens : 0 };
}

async function simulateOptimal(metadata, capacity, options) {
  if (capacity <= 0) return { hitTokens: 0, totalTokens: metadata.totalInputTokens || 0, hitRate: 0 };
  await ensureNextFile(metadata, options);
  const requestEnds = await readRequestEnds(metadata);
  const requestCursor = makeRequestCursor(requestEnds);
  const cache = new Map();
  const heap = new MaxHeap();
  let hitTokens = 0;
  let totalTokens = 0;
  let measurementStartRequest = -1;
  function pushState(id, nextUse) {
    const previous = cache.get(id);
    const state = { nextUse, version: (previous?.version || 0) + 1 };
    cache.set(id, state);
    heap.push({ id, nextUse, version: state.version });
  }
  function evictForCandidate(candidateNextUse) {
    for (;;) {
      const candidate = heap.pop();
      if (!candidate) return cache.size < capacity;
      const current = cache.get(candidate.id);
      if (!current || current.version !== candidate.version || current.nextUse !== candidate.nextUse) continue;
      if (current.nextUse > candidateNextUse) {
        cache.delete(candidate.id);
        return true;
      }
      heap.push(candidate);
      return false;
    }
  }
  await scanEvents(metadata, true, ({ start, count, ids, tokens, next }) => {
    for (let index = 0; index < count; index += 1) {
      const eventIndex = start + index;
      const requestIndex = requestCursor.current(eventIndex);
      const id = ids.readUInt32LE(index * 4);
      const tokenCount = tokens.readUInt16LE(index * 2);
      const nextUse = next.readUInt32LE(index * 4);
      const hit = cache.has(id);
      if (measurementStartRequest >= 0 && requestIndex >= measurementStartRequest) {
        totalTokens += tokenCount;
        if (hit) hitTokens += tokenCount;
      }
      if (hit) {
        pushState(id, nextUse);
      } else if (cache.size < capacity) {
        pushState(id, nextUse);
      } else {
        if (measurementStartRequest < 0) measurementStartRequest = requestIndex + 1;
        if (evictForCandidate(nextUse)) {
          if (cache.size < capacity) pushState(id, nextUse);
        } else {
          // Belady-with-bypass: do not admit a miss that would be used no sooner
          // than every resident block.
        }
      }
    }
  });
  if (measurementStartRequest < 0 || totalTokens <= 0) return computeCeiling(metadata);
  return { hitTokens, totalTokens, hitRate: totalTokens ? hitTokens / totalTokens : 0 };
}

async function simulateNativePolicy(metadata, capacity, policy, options) {
  if (!options.nativeSimPath) return null;
  if (policy === "optimal") await ensureNextFile(metadata, options);
  else await ensureNextFile(metadata, options);
  const args = [
    "--policy",
    policy,
    "--ids",
    metadata.idsPath,
    "--tokens",
    metadata.tokensPath,
    "--total-blocks",
    String(metadata.totalBlocks),
    "--warmup-event-start",
    String(metadata.warmupEventStart),
    "--capacity",
    String(capacity),
    "--request-ends",
    metadata.requestEndsPath,
    "--request-count",
    String(metadata.requestCount),
    "--warmup-requests",
    String(metadata.warmupRequests),
  ];
  args.push("--next", metadata.nextPath);
  const { stdout } = await execFileAsync(path.resolve(options.nativeSimPath), args, {
    encoding: "utf8",
    maxBuffer: 1024 * 1024,
  });
  return JSON.parse(stdout);
}

function normalizePolicyResult(policy, cacheBlocks, result) {
  const measurementStartRequest = Number.isFinite(Number(result.measurementStartRequest))
    ? Number(result.measurementStartRequest)
    : null;
  return {
    policy,
    cacheBlocks,
    warmupRequests: Number.isFinite(Number(result.warmupRequests)) ? Number(result.warmupRequests) : 0,
    measurementStartRequest: measurementStartRequest != null && measurementStartRequest >= 0 ? measurementStartRequest : null,
    measurementMode: measurementStartRequest === -2
      ? "underfilled_at_window"
      : measurementStartRequest != null && measurementStartRequest >= 0
        ? "fixed_window"
        : "fixed_window_ceiling",
    hitTokens: result.hitTokens,
    totalTokens: result.totalTokens,
    hitRate: result.hitRate,
  };
}

async function simulateNativeAllPolicies(metadata, capacity, options) {
  if (!options.nativeSimPath) return null;
  await ensureNextFile(metadata, options);
  const args = [
    "--policy",
    "all",
    "--ids",
    metadata.idsPath,
    "--tokens",
    metadata.tokensPath,
    "--total-blocks",
    String(metadata.totalBlocks),
    "--warmup-event-start",
    String(metadata.warmupEventStart),
    "--capacity",
    String(capacity),
    "--request-ends",
    metadata.requestEndsPath,
    "--request-count",
    String(metadata.requestCount),
    "--warmup-requests",
    String(metadata.warmupRequests),
    "--next",
    metadata.nextPath,
  ];
  const { stdout } = await execFileAsync(path.resolve(options.nativeSimPath), args, {
    encoding: "utf8",
    maxBuffer: 1024 * 1024,
  });
  const parsed = JSON.parse(stdout);
  return {
    fifo: normalizePolicyResult("fifo", capacity, parsed.fifo),
    lru: normalizePolicyResult("lru", capacity, parsed.lru),
    optimal: normalizePolicyResult("optimal", capacity, parsed.optimal),
  };
}

async function simulatePolicy(metadata, capacity, policy, ceiling, options) {
  const cacheBlocks = Math.max(0, Math.floor(capacity));
  const workingSet = policyWorkingSet(metadata, policy);
  if (workingSet > 0 && cacheBlocks > workingSet) return null;
  console.error(`[full] ${metadata.id}: simulate ${policy} capacity=${cacheBlocks}`);
  const nativeResult = await simulateNativePolicy(metadata, cacheBlocks, policy, options);
  if (nativeResult) {
    return normalizePolicyResult(policy, cacheBlocks, nativeResult);
  }
  const result =
    policy === "fifo"
      ? await simulateFifo(metadata, cacheBlocks)
      : policy === "lru"
        ? await simulateLru(metadata, cacheBlocks)
        : await simulateOptimal(metadata, cacheBlocks, options);
  return { policy, cacheBlocks, warmupRequests: 0, ...result };
}

function maxSweepHitRate(points) {
  let maxHitRate = -Infinity;
  for (const point of points || []) {
    for (const result of Object.values(point.results || {})) {
      const hitRate = Number(result && result.hitRate);
      if (Number.isFinite(hitRate)) maxHitRate = Math.max(maxHitRate, hitRate);
    }
  }
  return maxHitRate === -Infinity ? undefined : maxHitRate;
}

function modelSettingFor(model) {
  return {
    modelId: model.id,
    precision: model.formula === "deepseek_v4_hybrid" ? "fp8_int8" : "bf16_fp16",
    indexerPrecision: Number.isFinite(Number(model.fields && model.fields.index_head_dim))
      ? model.formula === "deepseek_v4_hybrid"
        ? "fp4_int4"
        : "bf16_fp16"
      : undefined,
    includeDraftKvCache: false,
  };
}

function seedSimulationResults(existingTrace, metadata) {
  return new Map();
}

function simulationResultHasCurrentStats(result) {
  return Boolean(result)
    && Number.isFinite(Number(result.hitRate))
    && Number.isFinite(Number(result.hitTokens))
    && Number.isFinite(Number(result.totalTokens));
}

function simulationResultCacheDir(metadata) {
  return path.join(path.dirname(metadata.idsPath), SIMULATION_RESULT_CACHE_DIR);
}

function simulationResultCachePath(metadata, policy, capacity) {
  return path.join(simulationResultCacheDir(metadata), `${policy}-${capacity}.json`);
}

async function seedCachedSimulationResults(simulationResults, metadata) {
  const dir = simulationResultCacheDir(metadata);
  if (!fs.existsSync(dir)) return 0;
  let count = 0;
  const files = await fsp.readdir(dir);
  for (const file of files) {
    const match = /^(fifo|lru|optimal)-(\d+)\.json$/.exec(file);
    if (!match) continue;
    const result = JSON.parse(await fsp.readFile(path.join(dir, file), "utf8"));
    if (!simulationResultHasCurrentStats(result)) continue;
    simulationResults.set(`${match[1]}|${match[2]}`, result);
    count += 1;
  }
  return count;
}

async function writeCachedSimulationResult(metadata, policy, capacity, result) {
  if (!simulationResultHasCurrentStats(result)) return;
  await fsp.mkdir(simulationResultCacheDir(metadata), { recursive: true });
  const destination = simulationResultCachePath(metadata, policy, capacity);
  const temporary = `${destination}.${process.pid}.tmp`;
  await fsp.writeFile(temporary, `${JSON.stringify(result)}\n`);
  await fsp.rename(temporary, destination);
}

async function runCapacitySimulations(metadata, capacities, simulationResults, options) {
  const missingCapacities = capacities.filter((capacity) =>
    POLICIES.some((policy) => capacity <= policyWorkingSet(metadata, policy) && !simulationResults.has(`${policy}|${capacity}`)),
  );
  console.error(
    `[full] ${metadata.id}: ${capacities.length} curve capacities below working set, ${capacities.length - missingCapacities.length} reused, ${missingCapacities.length} missing`,
  );
  if (options.nativeSimPath && missingCapacities.length) {
    await ensureNextFile(metadata, options);
  }
  const tasks = [];
  if (options.nativeSimPath) {
    for (const capacity of missingCapacities) tasks.push({ capacity });
  } else {
    for (const capacity of missingCapacities) {
      for (const policy of POLICIES) {
        if (simulationResults.has(`${policy}|${capacity}`)) continue;
        tasks.push({ capacity, policy });
      }
    }
  }
  const nativeJobs = options.nativeSimPath ? Math.max(1, Math.floor(options.nativeJobs || 1)) : 1;
  async function runTask(task) {
    if (task.policy) {
      const result = await simulatePolicy(metadata, task.capacity, task.policy, null, options);
      simulationResults.set(`${task.policy}|${task.capacity}`, result);
      await writeCachedSimulationResult(metadata, task.policy, task.capacity, result);
      return;
    }
    console.error(`[full] ${metadata.id}: simulate all policies capacity=${task.capacity}`);
    const results = await simulateNativeAllPolicies(metadata, task.capacity, options);
    for (const policy of POLICIES) {
      if (simulationResults.has(`${policy}|${task.capacity}`)) continue;
      const result = results[policy];
      simulationResults.set(`${policy}|${task.capacity}`, result);
      await writeCachedSimulationResult(metadata, policy, task.capacity, result);
    }
  }
  async function runTaskList(taskList, jobs) {
    let nextTask = 0;
    async function worker() {
      for (;;) {
        const index = nextTask;
        nextTask += 1;
        if (index >= taskList.length) return;
        await runTask(taskList[index]);
      }
    }
    await Promise.all(Array.from({ length: Math.min(jobs, taskList.length) }, () => worker()));
  }
  if (nativeJobs > 1) {
    const serialThreshold = Number.isFinite(options.nativeSerialCapacityThreshold)
      ? options.nativeSerialCapacityThreshold
      : Infinity;
    const concurrentTasks = tasks.filter((task) => task.capacity <= serialThreshold);
    const serialTasks = tasks.filter((task) => task.capacity > serialThreshold);
    if (concurrentTasks.length) await runTaskList(concurrentTasks, nativeJobs);
    for (const task of serialTasks) {
      await runTask(task);
    }
  } else {
    for (const task of tasks) {
      await runTask(task);
    }
  }
}

async function precomputeBlockCurveTrace(source, options, metadata, ceiling) {
  const simulationResults = new Map();
  const localSeedCount = await seedCachedSimulationResults(simulationResults, metadata);
  const maxCapacity = Math.max(...POLICIES.map((policy) => policyWorkingSet(metadata, policy)), 1);
  const capacities = blockCapacityGrid(maxCapacity, { curvePoints: options.curvePoints || 64 });
  console.error(`[full] ${source.id}: block-capacity curve grid has ${capacities.length} candidate points (${localSeedCount} cached results loaded)`);
  await runCapacitySimulations(metadata, capacities, simulationResults, options);

  const points = [];
  for (const capacity of capacities) {
    if (POLICIES.some((policy) => capacity > policyWorkingSet(metadata, policy))) {
      break;
    }
    const results = {};
    let underfilled = false;
    POLICIES.forEach((policy) => {
      const result = simulationResults.get(`${policy}|${capacity}`);
      results[policy] = result;
      if (!result || result.measurementMode === "underfilled_at_window") underfilled = true;
    });
    if (underfilled) {
      break;
    }
    points.push({ cacheBlocks: capacity, results });
  }

  const generatedAt = new Date().toISOString();
  const totalMeasuredTokens = Number.isFinite(Number(ceiling.totalTokens))
    ? Number(ceiling.totalTokens)
    : Number(metadata.totalMeasuredTokens) || 0;
  return {
    id: source.id,
    label: source.label,
    scenario: source.scenario,
    sourceKind: source.sourceKind,
    nativeBlockSize: metadata.blockSize,
    sourceBlockSizeNote: source.sourceBlockSizeNote,
    sources: source.sources || [],
    summary: {
      requests: metadata.requestCount,
      totalInputTokens: metadata.totalInputTokens,
      averageInputTokens: metadata.averageInputTokens,
      uniqueBlocks: metadata.uniqueBlocks,
      totalBlocks: metadata.totalBlocks,
      warmupRequests: metadata.warmupRequests,
      totalMeasuredTokens,
      infiniteHitRate: ceiling.hitRate,
    },
    blockCapacityCurve: {
      blockSize: metadata.blockSize,
      policies: POLICIES,
      capacityBlocks: points.map((point) => point.cacheBlocks),
      points,
      warmupRequests: metadata.warmupRequests,
      totalTokens: totalMeasuredTokens,
      reuseCeiling: ceiling.hitRate,
      interpolation: "log_linear_cache_blocks",
      generatedAt,
    },
  };
}

async function precomputeTrace(source, options, modelsData, existingTrace = null) {
  const metadata = await ensureEventStream(source, options);
  console.error(`[full] ${source.id}: computing infinite-cache ceiling`);
  const ceiling = await computeCeiling(metadata, options);
  if (options.blockCurve) {
    return precomputeBlockCurveTrace(source, options, metadata, ceiling);
  }
  const selectedModels = selectModels(modelsData.models, options);
  const precisionOptions = filterPrecisionOptions(modelsData.precision_options, options.precisionIds);
  const indexerPrecisionOptions = filterPrecisionOptions(modelsData.indexer_precision_options, options.indexerPrecisionIds);
  const modelGroups = options.dedupeKvArchitecture
    ? kvArchitectureGroups(selectedModels)
    : selectedModels.map((model) => ({ key: null, models: [model] }));
  console.error(
    `[full] ${source.id}: selected ${selectedModels.length} models across ${modelGroups.length} KV architecture groups`,
  );
  const capacityByKey = new Map();
  const sweepInputs = [];

  for (const group of modelGroups) {
    const model = group.models[0];
    const rawSettings = options.defaultSettings && !options.allPrecisions
      ? [modelSettingFor(model)]
      : allModelSettings([model], {
          precisionOptions,
          indexerPrecisionOptions,
          includeDraftKvCache: !options.noDraft,
        });
    for (const rawSetting of rawSettings) {
      const accountingSettings = {
        estimateTokens: positiveInteger(metadata.averageInputTokens, model.default_tokens || 4096),
        precision: rawSetting.precision,
        indexerPrecision: rawSetting.indexerPrecision,
        includeDraftKvCache: Boolean(rawSetting.includeDraftKvCache),
        precisionOptions,
        indexerPrecisionOptions,
      };
      const bytesPerToken = lab.estimateBytesPerToken(model, accountingSettings);
      const bytesPerBlock = bytesPerToken * metadata.blockSize;
      const capacities = options.capacityGiBValues.map((gib) => ({
        gib,
        cacheBlocks: lab.cacheBlocksForGiB(gib, bytesPerBlock),
      }));
      capacities.forEach((point) => {
        if (POLICIES.some((policy) => point.cacheBlocks <= policyWorkingSet(metadata, policy))) {
          capacityByKey.set(point.cacheBlocks, point.cacheBlocks);
        }
      });
      const aliases = group.models.map((aliasModel) => ({
        model: aliasModel,
        rawSetting: {
          ...rawSetting,
          modelId: aliasModel.id,
        },
      }));
      sweepInputs.push({ rawSetting, model, aliases, kvArchitectureKey: group.key, bytesPerToken, bytesPerBlock, capacities });
    }
  }

  const simulationResults = seedSimulationResults(existingTrace, metadata);
  const outputSeedCount = simulationResults.size;
  const localSeedCount = await seedCachedSimulationResults(simulationResults, metadata);
  const capacities = Array.from(capacityByKey.values()).sort((left, right) => left - right);
  const missingCapacities = capacities.filter((capacity) =>
    POLICIES.some((policy) => capacity <= policyWorkingSet(metadata, policy) && !simulationResults.has(`${policy}|${capacity}`)),
  );
  console.error(
    `[full] ${source.id}: ${capacities.length} finite capacities below unique working set, ${capacities.length - missingCapacities.length} reused (${outputSeedCount} output, ${localSeedCount} local), ${missingCapacities.length} missing`,
  );
  if (options.nativeSimPath && missingCapacities.length) {
    await ensureNextFile(metadata, options);
  }
  const tasks = [];
  if (options.nativeSimPath) {
    for (const capacity of missingCapacities) tasks.push({ capacity });
  } else {
    for (const capacity of missingCapacities) {
      for (const policy of POLICIES) {
        if (simulationResults.has(`${policy}|${capacity}`)) continue;
        tasks.push({ capacity, policy });
      }
    }
  }
  const nativeJobs = options.nativeSimPath ? Math.max(1, Math.floor(options.nativeJobs || 1)) : 1;
  async function runTask(task) {
    if (task.policy) {
      const result = await simulatePolicy(metadata, task.capacity, task.policy, ceiling, options);
      simulationResults.set(`${task.policy}|${task.capacity}`, result);
      await writeCachedSimulationResult(metadata, task.policy, task.capacity, result);
      return;
    }
    console.error(`[full] ${metadata.id}: simulate all policies capacity=${task.capacity}`);
    const results = await simulateNativeAllPolicies(metadata, task.capacity, options);
    for (const policy of POLICIES) {
      if (simulationResults.has(`${policy}|${task.capacity}`)) continue;
      const result = results[policy];
      simulationResults.set(`${policy}|${task.capacity}`, result);
      await writeCachedSimulationResult(metadata, policy, task.capacity, result);
    }
  }
  async function runTaskList(taskList, jobs) {
    let nextTask = 0;
    async function worker() {
      for (;;) {
        const index = nextTask;
        nextTask += 1;
        if (index >= taskList.length) return;
        await runTask(taskList[index]);
      }
    }
    await Promise.all(Array.from({ length: Math.min(jobs, taskList.length) }, () => worker()));
  }
  if (nativeJobs > 1) {
    const serialThreshold = Number.isFinite(options.nativeSerialCapacityThreshold)
      ? options.nativeSerialCapacityThreshold
      : Infinity;
    const concurrentTasks = tasks.filter((task) => task.capacity <= serialThreshold);
    const serialTasks = tasks.filter((task) => task.capacity > serialThreshold);
    if (concurrentTasks.length) await runTaskList(concurrentTasks, nativeJobs);
    for (const task of serialTasks) {
      await runTask(task);
    }
  } else {
    for (const task of tasks) {
      await runTask(task);
    }
  }

  const generatedAt = new Date().toISOString();
  const modelSweeps = {};
  for (const input of sweepInputs) {
    const points = [];
    for (const point of input.capacities) {
      if (POLICIES.some((policy) => {
        const workingSet = policyWorkingSet(metadata, policy);
        return workingSet > 0 && point.cacheBlocks > workingSet;
      })) {
        break;
      }
      const results = {};
      let underfilled = false;
      POLICIES.forEach((policy) => {
        const key = `${policy}|${point.cacheBlocks}`;
        const result = simulationResults.get(key);
        results[policy] = result;
        if (!result || result.measurementMode === "underfilled_at_window") underfilled = true;
      });
      if (underfilled) {
        break;
      }
      points.push({ gib: point.gib, cacheBlocks: point.cacheBlocks, results });
    }
    const reuseCeiling = ceiling.hitRate;
    for (const alias of input.aliases) {
      modelSweeps[modelSweepKey(alias.rawSetting)] = {
        modelId: alias.model.id,
        modelLabel: alias.model.label,
        kvArchitectureKey: input.kvArchitectureKey,
        kvArchitectureModelIds: input.aliases.map((item) => item.model.id),
        precision: alias.rawSetting.precision,
        indexerPrecision: alias.rawSetting.indexerPrecision || null,
        includeDraftKvCache: Boolean(alias.rawSetting.includeDraftKvCache),
        blockSize: metadata.blockSize,
        bytesPerToken: input.bytesPerToken,
        bytesPerBlock: input.bytesPerBlock,
        points,
        policies: POLICIES,
        reuseCeiling,
        warmupRequests: metadata.warmupRequests,
        sourceKind: metadata.sourceKind,
        generatedAt,
      };
    }
  }

  return {
    id: source.id,
    label: source.label,
    scenario: source.scenario,
    sourceKind: source.sourceKind,
    nativeBlockSize: metadata.blockSize,
    sourceBlockSizeNote: source.sourceBlockSizeNote,
    sources: source.sources || [],
    summary: {
      requests: metadata.requestCount,
      totalInputTokens: metadata.totalInputTokens,
      averageInputTokens: metadata.averageInputTokens,
      uniqueBlocks: metadata.uniqueBlocks,
      totalBlocks: metadata.totalBlocks,
      warmupRequests: metadata.warmupRequests,
      infiniteHitRate: ceiling.hitRate,
    },
    modelSweeps,
  };
}

function refreshOutputMetadata(output, options) {
  output.metadata = {
    ...(output.metadata || {}),
    mode: "precomputed_real_traces",
    note: "Raw traces are downloaded into a local cache and are not committed. This file contains derived hit-rate curves and provenance.",
    setting_mode: options.blockCurve
      ? "model_independent_block_capacity_curve"
      : options.dedupeKvArchitecture
        ? "deduped_kv_architecture_precision_indexer_settings"
      : options.defaultSettings && options.modelIds && options.modelIds.length === 1
        ? "default_model_default_precision_only"
        : options.defaultSettings
          ? "default_model_settings"
          : "all_model_precision_indexer_draft_settings",
    capacity_gib_values: options.capacityGiBValues,
    curve_points: options.blockCurve ? (options.curvePoints || 64) : output.metadata?.curve_points,
    warmup_fraction: DEFAULT_WARMUP_FRACTION,
    policies: POLICIES,
    sources: { ...(output.metadata && output.metadata.sources ? output.metadata.sources : {}), ...SOURCE_LINKS },
    reference_sources: output.metadata?.reference_sources || REFERENCE_SOURCES,
    include_families: options.includeFamilies,
    exclude_families: options.excludeFamilies,
    dedupe_kv_architecture: Boolean(options.dedupeKvArchitecture),
    include_draft_kv_cache: !options.noDraft,
    full_trace_updated_at: new Date().toISOString(),
  };
}

async function writeOutputCheckpoint(outputPath, output) {
  await fsp.mkdir(path.dirname(outputPath), { recursive: true });
  await fsp.writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`);
}

const options = parseArgs(process.argv.slice(2));
const selectedTraceIds = options.traceIds || Array.from(FULL_TRACE_IDS);
const modelsData = loadModelsData(options.modelsPath);
const outputPath = path.resolve(options.outputPath || DEFAULT_OUTPUT_PATH);
const output = fs.existsSync(outputPath)
  ? JSON.parse(await fsp.readFile(outputPath, "utf8"))
  : { metadata: {}, traces: {} };

for (const traceId of selectedTraceIds) {
  const source = sourceById(traceId);
  if (options.eventsOnly) {
    const metadata = await ensureEventStream(source, options);
    console.log(JSON.stringify({ trace: traceId, metadata }, null, 2));
    continue;
  }
  const trace = await precomputeTrace(source, options, modelsData, output.traces && output.traces[traceId]);
  output.traces = { ...(output.traces || {}), [traceId]: trace };
  refreshOutputMetadata(output, options);
  await writeOutputCheckpoint(outputPath, output);
  console.error(`[full] ${traceId}: checkpoint written to ${outputPath}`);
}

if (options.eventsOnly) process.exit(0);

refreshOutputMetadata(output, options);
await writeOutputCheckpoint(outputPath, output);
console.log(JSON.stringify({ outputPath, traces: selectedTraceIds }, null, 2));
