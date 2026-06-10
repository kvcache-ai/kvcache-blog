import crypto from "node:crypto";
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";

const require = createRequire(import.meta.url);
const calculator = require("../../assets/js/kv-cache-calculator.js");
const lab = require("../../assets/js/kv-cache-lab.js");

export const DEFAULT_TRACE_CACHE_DIR = path.join(os.tmpdir(), "kvcache-lab-traces");
export const DEFAULT_OUTPUT_PATH = path.resolve("data/kv_cache_lab/precomputed.json");
export const DEFAULT_CAPACITY_GIB_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
export const DEFAULT_WARMUP_FRACTION = 0.3;
export const POLICIES = ["fifo", "lru", "optimal"];

export const TRACE_SOURCES = [
  {
    id: "mooncake_fast25",
    label: "Mooncake FAST25 Tool&Agent Trace",
    scenario: "Production Tool&Agent hash trace",
    sourceKind: "hash",
    parser: "mooncake",
    nativeBlockSize: 512,
    sourceBlockSizeNote: "hash_ids align with 512-token Mooncake blocks; the last block is weighted by the input-length remainder.",
    url: "https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl",
    localFile: "mooncake_fast25.jsonl",
    sources: ["mooncake_fast25", "nvidia_aiperf_mooncake"],
  },
  {
    id: "bailian_qwen_trace_a",
    label: "Qwen Bailian Trace A",
    scenario: "Anonymous production chat/search/code traffic",
    sourceKind: "hash",
    parser: "bailian",
    nativeBlockSize: 16,
    sourceBlockSizeNote: "published as qwen_traceA_blksz_16.jsonl; hash_ids are 16-token buckets.",
    url: "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/raw/refs/heads/main/qwen_traceA_blksz_16.jsonl",
    localFile: "qwen_traceA_blksz_16.jsonl",
    sources: ["qwen_bailian_trace"],
  },
  {
    id: "ragpulse",
    label: "RAGPulse",
    scenario: "RAG workload hash trace",
    sourceKind: "hash",
    parser: "ragpulse",
    nativeBlockSize: 512,
    sourceBlockSizeNote: "RAGPulse exposes categorized hash_ids; v2 namespaces categories and weights token counts across the listed ids.",
    parquetUrl: "https://huggingface.co/datasets/flashserve/RAGPulse/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    localFile: "ragpulse.parquet",
    hfDataset: "flashserve/RAGPulse",
    hfConfig: "default",
    hfSplit: "train",
    sources: ["ragpulse"],
  },
  {
    id: "lmcache_agentic_sample",
    label: "LMCache Agentic Traces",
    scenario: "SWE-bench style agent conversations",
    sourceKind: "agent_text",
    parser: "lmcache_agentic",
    nativeBlockSize: 64,
    rowLimit: 1200,
    sourceBlockSizeNote: "message text is normalized and hashed into estimated 64-token buckets; source does not publish block hash ids.",
    hfDataset: "zeelHz/lmcache-agentic-traces",
    hfConfig: "default",
    hfSplit: "train",
    sources: ["lmcache_agentic_traces"],
  },
  {
    id: "semianalysis_weka_no_subagents",
    label: "Weka Claude Code",
    scenario: "Agentic coding production hash trace",
    sourceKind: "hash",
    parser: "weka_session",
    nativeBlockSize: 64,
    sourceBlockSizeNote: "Claude Code request hash_ids are source-native 64-token blocks; hash ids are local to each trace.",
    hfDataset: "semianalysisai/cc-traces-weka-no-subagents-051226",
    hfConfig: "default",
    hfSplit: "train",
    hfPageSize: 1,
    sources: ["semianalysis_weka_no_subagents"],
  },
  {
    id: "semianalysis_weka_with_subagents_256k",
    label: "Weka Claude Code + Subagents",
    scenario: "Agentic coding hash trace with sub-agent fan-out",
    sourceKind: "hash",
    parser: "weka_session",
    nativeBlockSize: 64,
    sourceBlockSizeNote: "Claude Code main-agent and sub-agent requests use source-native 64-token local hash ids; rows above 256k proxy tokens are filtered by the source dataset.",
    hfDataset: "semianalysisai/cc-traces-weka-with-subagents-052726-256k",
    hfConfig: "default",
    hfSplit: "train",
    hfPageSize: 1,
    sources: ["semianalysis_weka_with_subagents_256k"],
  },
  {
    id: "kv_cache_tester_claude_code",
    label: "kv-cache-tester Claude Code",
    scenario: "Agentic coding trace replay hash corpus",
    sourceKind: "hash",
    parser: "weka_session",
    nativeBlockSize: 64,
    sourceBlockSizeNote: "The trace replay corpus stores one input hash id per 64-token block; sub-agent inner requests share the parent trace namespace.",
    gitUrl: "https://github.com/callanjfox/kv-cache-tester.git",
    localDir: "kv-cache-tester",
    traceDir: "traces",
    sources: ["kv_cache_tester"],
  },
  {
    id: "exgentic_agent_sample",
    label: "Exgentic Agent Traces",
    scenario: "Agent benchmark spans",
    sourceKind: "agent_text",
    parser: "exgentic_agent",
    nativeBlockSize: 64,
    rowLimit: 300,
    sourceBlockSizeNote: "span payloads are normalized and hashed into estimated 64-token buckets; source does not publish block hash ids.",
    hfDataset: "Exgentic/agent-llm-traces",
    hfConfig: "default",
    hfSplit: "train",
    sources: ["exgentic_agent_traces"],
  },
];

export const REFERENCE_SOURCES = [
  {
    id: "burstgpt",
    label: "BurstGPT",
    role: "arrival/load reference only",
    reason: "Contains timestamps and token lengths but no block/prefix identity, so it should not drive cache hit-rate curves.",
    url: "https://huggingface.co/datasets/lzzmm/BurstGPT",
  },
  {
    id: "swissai_serving_trace",
    label: "SwissAI Serving Trace",
    role: "arrival/length reference only",
    reason: "Useful for serving-token distribution; bucketized token traces should only become hit-rate traces when identity semantics are explicit.",
    url: "https://huggingface.co/datasets/eth-easl/swissai-serving-trace",
  },
];

export const SOURCE_LINKS = {
  mooncake_fast25: "https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/arxiv-trace",
  nvidia_aiperf_mooncake: "https://docs.nvidia.com/aiperf/benchmark-modes/trace-replay-with-mooncake-traces/",
  qwen_bailian_trace: "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon",
  ragpulse: "https://huggingface.co/datasets/flashserve/RAGPulse",
  lmcache_agentic_traces: "https://huggingface.co/datasets/zeelHz/lmcache-agentic-traces",
  semianalysis_weka_no_subagents: "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226",
  semianalysis_weka_with_subagents_256k: "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-052726-256k",
  kv_cache_tester: "https://github.com/callanjfox/kv-cache-tester",
  exgentic_agent_traces: "https://huggingface.co/datasets/Exgentic/agent-llm-traces",
  burstgpt: "https://huggingface.co/datasets/lzzmm/BurstGPT",
  swissai_serving_trace: "https://huggingface.co/datasets/eth-easl/swissai-serving-trace",
};

export const FEATURED_MODEL_SETTINGS = [
  { modelId: "deepseek-v4-pro", precision: "fp8_int8", indexerPrecision: "fp4_int4", includeDraftKvCache: false },
  { modelId: "qwen3-32b", precision: "bf16_fp16", includeDraftKvCache: false },
  { modelId: "llama-3.1-8b", precision: "bf16_fp16", includeDraftKvCache: false },
  { modelId: "kimi-k2.6", precision: "bf16_fp16", includeDraftKvCache: false },
  { modelId: "mimo-v2.5-pro", precision: "bf16_fp16", includeDraftKvCache: false },
];

function modelFieldNumber(model, name, fallback) {
  const parsed = Number(model && model.fields ? model.fields[name] : undefined);
  return Number.isFinite(parsed) ? parsed : fallback;
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

function optionIds(options, fallback) {
  const ids = (options || []).map((option) => option.id || option).filter(Boolean);
  return ids.length ? ids : fallback;
}

function defaultPrecisionId(model, precisionIds) {
  if (isDeepSeekV4(model) && precisionIds.includes("fp8_int8")) return "fp8_int8";
  if (precisionIds.includes("bf16_fp16")) return "bf16_fp16";
  return precisionIds[0];
}

function defaultIndexerPrecisionId(model, indexerPrecisionIds, fallbackPrecisionId) {
  if (isDeepSeekV4(model) && indexerPrecisionIds.includes("fp4_int4")) return "fp4_int4";
  if (fallbackPrecisionId && indexerPrecisionIds.includes(fallbackPrecisionId)) return fallbackPrecisionId;
  if (indexerPrecisionIds.includes("bf16_fp16")) return "bf16_fp16";
  return indexerPrecisionIds.includes("fp4_int4") ? "fp4_int4" : indexerPrecisionIds[0];
}

function orderedValues(values, preferred) {
  const result = [];
  if (preferred && values.includes(preferred)) result.push(preferred);
  values.forEach((value) => {
    if (!result.includes(value)) result.push(value);
  });
  return result;
}

export function defaultModelSetting(model) {
  const precision = isDeepSeekV4(model) ? "fp8_int8" : "bf16_fp16";
  const setting = {
    modelId: model.id,
    precision,
    includeDraftKvCache: false,
  };
  if (hasIndexerCache(model)) {
    setting.indexerPrecision = isDeepSeekV4(model) ? "fp4_int4" : precision;
  }
  return setting;
}

export function allDefaultModelSettings(models) {
  return (models || []).map(defaultModelSetting);
}

export function allModelSettings(models, options = {}) {
  const precisionIds = optionIds(options.precisionOptions, ["bf16_fp16", "fp8_int8", "fp4_int4"]);
  const indexerPrecisionIds = optionIds(options.indexerPrecisionOptions, precisionIds);
  const settings = [];
  (models || []).forEach((model) => {
    const preferredPrecision = defaultPrecisionId(model, precisionIds);
    const precisions = orderedValues(precisionIds, preferredPrecision);
    const includeDraftValues = hasDraftKvCache(model) ? [false, true] : [false];
    precisions.forEach((precision) => {
      if (hasIndexerCache(model)) {
        const preferredIndexerPrecision = defaultIndexerPrecisionId(model, indexerPrecisionIds, precision);
        orderedValues(indexerPrecisionIds, preferredIndexerPrecision).forEach((indexerPrecision) => {
          includeDraftValues.forEach((includeDraftKvCache) => {
            settings.push({ modelId: model.id, precision, indexerPrecision, includeDraftKvCache });
          });
        });
      } else {
        includeDraftValues.forEach((includeDraftKvCache) => {
          settings.push({ modelId: model.id, precision, includeDraftKvCache });
        });
      }
    });
  });
  return settings;
}

export function modelSweepKey(setting) {
  return [
    setting.modelId,
    `precision=${setting.precision || ""}`,
    `indexer=${setting.indexerPrecision || ""}`,
    `draft=${setting.includeDraftKvCache ? "1" : "0"}`,
  ].join("|");
}

export function stableHash(input) {
  return crypto.createHash("sha256").update(String(input)).digest("hex").slice(0, 20);
}

export function estimateTokensFromText(text) {
  const normalized = String(text || "").replace(/\s+/g, " ").trim();
  if (!normalized) return 1;
  return Math.max(1, Math.ceil(normalized.length / 4));
}

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

function weightedNamespacedBlocks(groups, inputLength, blockSize) {
  const entries = [];
  Object.entries(groups || {}).forEach(([group, ids]) => {
    if (!Array.isArray(ids)) return;
    ids.forEach((id) => entries.push([group, id]));
  });
  if (!entries.length) return [];
  const perBlockTokens = Math.max(1, Math.round(inputLength / entries.length));
  return entries.map(([group, id]) => ({
    id: `${group}:${String(id)}`,
    tokens: Math.min(blockSize, perBlockTokens),
  }));
}

export function normalizeMooncakeRecord(record, source = {}) {
  const blockSize = positiveInteger(source.nativeBlockSize, 512);
  const inputLength = positiveInteger(record.input_length, 1);
  return {
    id: record.id,
    timestamp: toNumber(record.timestamp, 0),
    inputTokens: inputLength,
    outputTokens: Math.max(0, Math.floor(toNumber(record.output_length, 0))),
    inputBlocks: namespacedBlocks(record.hash_ids, source.id || "mooncake", inputLength, blockSize),
    appendBlocks: [],
  };
}

export function normalizeBailianRecord(record, source = {}) {
  const blockSize = positiveInteger(source.nativeBlockSize, 16);
  const inputLength = positiveInteger(record.input_length, 1);
  return {
    id: record.chat_id,
    parentId: record.parent_chat_id,
    timestamp: toNumber(record.timestamp, 0),
    inputTokens: inputLength,
    outputTokens: Math.max(0, Math.floor(toNumber(record.output_length, 0))),
    type: record.type || "unknown",
    turn: Math.max(1, Math.floor(toNumber(record.turn, 1))),
    inputBlocks: namespacedBlocks(record.hash_ids, source.id || "bailian", inputLength, blockSize),
    appendBlocks: [],
  };
}

export function normalizeRagPulseRecord(record, source = {}) {
  const blockSize = positiveInteger(source.nativeBlockSize, 512);
  const inputLength = positiveInteger(record.input_length, 1);
  return {
    id: record.session_id,
    timestamp: toNumber(record.timestamp, 0),
    inputTokens: inputLength,
    outputTokens: Math.max(0, Math.floor(toNumber(record.output_length, 0))),
    sessionId: record.session_id,
    inputBlocks: weightedNamespacedBlocks(record.hash_ids || {}, inputLength, blockSize),
    appendBlocks: [],
  };
}

function normalizeWekaRequest(record, source, context) {
  if (!record || !Array.isArray(record.hash_ids) || !record.hash_ids.length) return null;
  const blockSize = positiveInteger(context.blockSize, positiveInteger(source.nativeBlockSize, 64));
  const inputLength = positiveInteger(record.in ?? record.input_length, record.hash_ids.length * blockSize);
  const traceId = context.traceId || record.id || "trace";
  const ordinal = context.ordinal;
  return {
    id: `${traceId}:${context.group || "main"}:${ordinal}`,
    timestamp: toNumber(context.timestampOffset, 0) + toNumber(record.t ?? record.timestamp, 0),
    inputTokens: inputLength,
    outputTokens: Math.max(0, Math.floor(toNumber(record.out ?? record.output_length, 0))),
    model: record.model,
    type: record.type || "request",
    traceId,
    inputBlocks: namespacedBlocks(record.hash_ids, `${source.id || "weka"}:${traceId}`, inputLength, blockSize),
    appendBlocks: [],
  };
}

function parseWekaNestedRecord(record) {
  if (typeof record !== "string") return record;
  try {
    return JSON.parse(record);
  } catch {
    return null;
  }
}

function flattenWekaRequests(records, source, context, output) {
  let ordinal = context.ordinal || 0;
  (records || []).forEach((rawRecord, index) => {
    const record = parseWekaNestedRecord(rawRecord);
    const normalized = normalizeWekaRequest(record, source, { ...context, ordinal });
    if (normalized) {
      output.push(normalized);
      ordinal += 1;
    }
    if (Array.isArray(record && record.requests)) {
      ordinal = flattenWekaRequests(
        record.requests,
        source,
        {
          ...context,
          group: `${context.group || "main"}:sub${index}`,
          timestampOffset: toNumber(context.timestampOffset, 0) + toNumber(record.t, 0),
          ordinal,
        },
        output,
      );
    }
  });
  return ordinal;
}

export function normalizeWekaSessionRecord(record, source = {}) {
  const blockSize = positiveInteger(record.block_size, positiveInteger(source.nativeBlockSize, 64));
  const traceId = record.id || record.session_id || stableHash(JSON.stringify(record).slice(0, 1024));
  const requests = [];
  flattenWekaRequests(record.requests || [], source, { traceId, blockSize, group: "main", timestampOffset: 0, ordinal: 0 }, requests);
  requests.sort((left, right) => toNumber(left.timestamp, 0) - toNumber(right.timestamp, 0));
  const perSessionLimit = source.maxRequestsPerSession ? positiveInteger(source.maxRequestsPerSession, requests.length) : Infinity;
  return requests.slice(0, perSessionLimit);
}

function normalizeMessage(message) {
  if (!message || typeof message !== "object") return String(message || "");
  const role = message.role || message.name || "message";
  const content = typeof message.content === "string" ? message.content : JSON.stringify(message.content || "");
  const toolCalls = message.tool_calls ? JSON.stringify(message.tool_calls) : "";
  const toolId = message.tool_call_id || "";
  return [role, message.name || "", toolId, content, toolCalls].filter(Boolean).join("\n");
}

function textBlocks(prefix, text, blockSize) {
  const tokens = estimateTokensFromText(text);
  const count = Math.max(1, Math.ceil(tokens / blockSize));
  const blocks = [];
  for (let index = 0; index < count; index += 1) {
    const chunkId = stableHash(`${prefix}:${index}:${String(text).slice(index * 256, (index + 1) * 256)}`);
    blocks.push({
      id: `${prefix}:${chunkId}`,
      tokens: blockTokens(tokens, blockSize, index, count),
    });
  }
  return { tokens, blocks };
}

export function normalizeLmcacheAgenticRecord(record, source = {}) {
  const blockSize = positiveInteger(source.nativeBlockSize, 64);
  const messages = Array.isArray(record.input) ? record.input.map(normalizeMessage) : [String(record.input || "")];
  const text = messages.join("\n---\n");
  const sessionId = record.session_id || stableHash(text);
  const normalized = textBlocks(`${source.id || "lmcache"}:${sessionId}`, text, blockSize);
  return {
    id: `${sessionId}:${record.pre_gap || 0}`,
    timestamp: toNumber(record.pre_gap, 0),
    inputTokens: normalized.tokens,
    outputTokens: Math.max(0, Math.floor(toNumber(record.output_length, 0))),
    sessionId,
    inputBlocks: normalized.blocks,
    appendBlocks: [],
  };
}

export function normalizeExgenticAgentRecord(record, source = {}) {
  const blockSize = positiveInteger(source.nativeBlockSize, 64);
  let spansText = record.spans || "";
  try {
    const spans = JSON.parse(spansText);
    spansText = Array.isArray(spans)
      ? spans.map((span) => [span.name, span.input, span.output, span.attributes && JSON.stringify(span.attributes)].filter(Boolean).join("\n")).join("\n---\n")
      : JSON.stringify(spans);
  } catch (error) {
    spansText = String(spansText || "");
  }
  const sessionId = record.session_id || stableHash(spansText);
  const normalized = textBlocks(`${source.id || "exgentic"}:${sessionId}`, spansText, blockSize);
  return {
    id: sessionId,
    timestamp: Date.parse(record.collected_at || "") || 0,
    inputTokens: normalized.tokens,
    outputTokens: 0,
    sessionId,
    inputBlocks: normalized.blocks,
    appendBlocks: [],
  };
}

export function normalizeRecord(record, source) {
  if (source.parser === "mooncake") return normalizeMooncakeRecord(record, source);
  if (source.parser === "bailian") return normalizeBailianRecord(record, source);
  if (source.parser === "ragpulse") return normalizeRagPulseRecord(record, source);
  if (source.parser === "lmcache_agentic") return normalizeLmcacheAgenticRecord(record, source);
  if (source.parser === "weka_session") return normalizeWekaSessionRecord(record, source);
  if (source.parser === "exgentic_agent") return normalizeExgenticAgentRecord(record, source);
  throw new Error(`Unsupported trace parser: ${source.parser}`);
}

function normalizeRecordList(record, source) {
  const normalized = normalizeRecord(record, source);
  return Array.isArray(normalized) ? normalized : [normalized];
}

export async function downloadFile(url, destination, options = {}) {
  await fsp.mkdir(path.dirname(destination), { recursive: true });
  const existingSize = fs.existsSync(destination) && !options.force ? fs.statSync(destination).size : 0;
  const headers = existingSize > 0 ? { Range: `bytes=${existingSize}-` } : {};
  const response = await fetch(url, { headers });
  if (!response.ok && response.status !== 206) {
    throw new Error(`Failed to download ${url}: HTTP ${response.status}`);
  }
  const append = existingSize > 0 && response.status === 206;
  const file = fs.createWriteStream(destination, { flags: append ? "a" : "w" });
  await new Promise((resolve, reject) => {
    response.body.pipeTo(
      new WritableStream({
        write(chunk) {
          file.write(Buffer.from(chunk));
        },
        close() {
          file.end(resolve);
        },
        abort(error) {
          file.destroy(error);
          reject(error);
        },
      }),
    ).catch(reject);
  });
  return fileHash(destination);
}

export async function fileHash(filePath) {
  const hash = crypto.createHash("sha256");
  await new Promise((resolve, reject) => {
    fs.createReadStream(filePath)
      .on("data", (chunk) => hash.update(chunk))
      .on("error", reject)
      .on("end", resolve);
  });
  return hash.digest("hex");
}

export async function fetchTraceSources(options = {}) {
  const cacheDir = options.cacheDir || DEFAULT_TRACE_CACHE_DIR;
  const selected = new Set(options.traceIds || TRACE_SOURCES.map((source) => source.id));
  const manifest = [];
  for (const source of TRACE_SOURCES) {
    if (!selected.has(source.id)) continue;
    if (source.gitUrl) {
      const repoDir = await ensureGitRepository(source, options);
      manifest.push({ id: source.id, path: repoDir, gitUrl: source.gitUrl });
      continue;
    }
    if (!source.url) continue;
    const destination = path.join(cacheDir, source.localFile || `${source.id}.jsonl`);
    const sha256 = await downloadFile(source.url, destination, { force: options.force });
    const stat = await fsp.stat(destination);
    manifest.push({ id: source.id, path: destination, bytes: stat.size, sha256, url: source.url });
  }
  await fsp.mkdir(cacheDir, { recursive: true });
  await fsp.writeFile(path.join(cacheDir, "manifest.json"), JSON.stringify({ generated_at: new Date().toISOString(), files: manifest }, null, 2));
  return manifest;
}

async function readJsonl(filePath, source, options = {}) {
  const requests = [];
  const limit = options.requestLimit || source.requestLimit || Infinity;
  let buffer = "";
  function appendRecord(record) {
    for (const request of normalizeRecordList(record, source)) {
      requests.push(request);
      if (requests.length >= limit) return true;
    }
    return false;
  }
  for await (const chunk of fs.createReadStream(filePath, { encoding: "utf8" })) {
    buffer += chunk;
    let newline = buffer.indexOf("\n");
    while (newline >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line && appendRecord(JSON.parse(line))) return requests;
      newline = buffer.indexOf("\n");
    }
  }
  if (buffer.trim() && requests.length < limit) appendRecord(JSON.parse(buffer));
  return requests;
}

async function ensureGitRepository(source, options = {}) {
  const cacheDir = options.cacheDir || DEFAULT_TRACE_CACHE_DIR;
  const repoDir = path.join(cacheDir, source.localDir || source.id);
  if (!fs.existsSync(repoDir)) {
    await fsp.mkdir(cacheDir, { recursive: true });
    execFileSync("git", ["clone", "--depth=1", source.gitUrl, repoDir], { stdio: "inherit" });
  } else if (!fs.existsSync(path.join(repoDir, ".git"))) {
    throw new Error(`${repoDir} exists but is not a git repository`);
  }
  return repoDir;
}

async function readJsonDirectory(directory, source, options = {}) {
  const requests = [];
  const limit = options.requestLimit || source.requestLimit || Infinity;
  const files = (await fsp.readdir(directory))
    .filter((file) => file.endsWith(".json"))
    .sort();
  for (const file of files) {
    const record = JSON.parse(await fsp.readFile(path.join(directory, file), "utf8"));
    for (const request of normalizeRecordList(record, source)) {
      requests.push(request);
      if (requests.length >= limit) return requests;
    }
  }
  return requests;
}

async function fetchHfRows(source, options = {}) {
  const cacheDir = options.cacheDir || DEFAULT_TRACE_CACHE_DIR;
  const cachePath = path.join(cacheDir, `${source.id}.hf-rows.jsonl`);
  if (fs.existsSync(cachePath) && !options.force) {
    return readJsonl(cachePath, source, options);
  }
  await fsp.mkdir(cacheDir, { recursive: true });
  const writer = fs.createWriteStream(cachePath, { flags: "w" });
  const limit = options.requestLimit || source.requestLimit || source.rowLimit || Infinity;
  const pageSize = Math.min(source.hfPageSize || 100, limit);
  const rows = [];
  for (let offset = 0; rows.length < limit; offset += pageSize) {
    const length = Math.min(pageSize, limit - rows.length);
    const url = new URL("https://datasets-server.huggingface.co/rows");
    url.searchParams.set("dataset", source.hfDataset);
    url.searchParams.set("config", source.hfConfig || "default");
    url.searchParams.set("split", source.hfSplit || "train");
    url.searchParams.set("offset", String(offset));
    url.searchParams.set("length", String(length));
    let response = null;
    let fetchError = null;
    for (let attempt = 0; attempt < 5; attempt += 1) {
      try {
        response = await fetch(url);
        fetchError = null;
      } catch (error) {
        response = null;
        fetchError = error;
      }
      if (response && (response.ok || (response.status !== 429 && response.status < 500))) break;
      await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** attempt));
    }
    if (!response || !response.ok) {
      if (rows.length > 0) break;
      await new Promise((resolve) => writer.end(resolve));
      if (fetchError) throw fetchError;
      throw new Error(`Failed to fetch HF rows for ${source.id}: HTTP ${response ? response.status : "network"}`);
    }
    const payload = await response.json();
    const pageRows = Array.isArray(payload.rows) ? payload.rows : [];
    if (!pageRows.length) break;
    for (const row of pageRows) {
      writer.write(`${JSON.stringify(row.row)}\n`);
      for (const request of normalizeRecordList(row.row, source)) {
        rows.push(request);
        if (rows.length >= limit) break;
      }
      if (rows.length >= limit) break;
    }
    if (pageRows.length < length) break;
  }
  await new Promise((resolve) => writer.end(resolve));
  return rows;
}

async function readParquetJsonl(source, options = {}) {
  const cacheDir = options.cacheDir || DEFAULT_TRACE_CACHE_DIR;
  const parquetPath = path.join(cacheDir, source.localFile || `${source.id}.parquet`);
  const jsonPath = path.join(cacheDir, `${source.id}.parquet.jsonl`);
  if (!fs.existsSync(parquetPath) || options.force) {
    await downloadFile(source.parquetUrl, parquetPath, { force: options.force });
  }
  if (!fs.existsSync(jsonPath) || options.force) {
    const limit = options.requestLimit ? ` LIMIT ${Math.max(1, Math.floor(options.requestLimit))}` : "";
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
        `COPY (SELECT * FROM data${limit}) TO '${jsonPath}' (FORMAT JSON)`,
      ],
      { stdio: "ignore", maxBuffer: 20 * 1024 * 1024 },
    );
  }
  return readJsonl(jsonPath, source, options);
}

export async function normalizeTraceSource(source, options = {}) {
  const cacheDir = options.cacheDir || DEFAULT_TRACE_CACHE_DIR;
  const rawPath = source.url ? path.join(cacheDir, source.localFile || `${source.id}.jsonl`) : null;
  if (rawPath && !fs.existsSync(rawPath)) {
    await downloadFile(source.url, rawPath, { force: false });
  }
  let requests;
  if (source.gitUrl) {
    const repoDir = await ensureGitRepository(source, options);
    requests = await readJsonDirectory(path.join(repoDir, source.traceDir || ""), source, options);
  } else if (source.parquetUrl) {
    requests = await readParquetJsonl(source, options);
  } else if (source.hfDataset) {
    requests = await fetchHfRows(source, options);
  } else {
    requests = await readJsonl(rawPath, source, options);
  }
  return buildTrace(source, requests);
}

export function buildTrace(source, requests) {
  const safeRequests = requests.filter((request) => Array.isArray(request.inputBlocks) && request.inputBlocks.length);
  safeRequests.sort((a, b) => toNumber(a.timestamp, 0) - toNumber(b.timestamp, 0));
  const totalInputTokens = safeRequests.reduce((sum, request) => sum + request.inputBlocks.reduce((inner, block) => inner + toNumber(block.tokens, 0), 0), 0);
  const uniqueBlocks = new Set();
  let totalBlocks = 0;
  safeRequests.forEach((request) => {
    request.inputBlocks.forEach((block) => {
      uniqueBlocks.add(block.id);
      totalBlocks += 1;
    });
  });
  return {
    id: source.id,
    label: source.label,
    scenario: source.scenario,
    sourceKind: source.sourceKind,
    blockSize: source.nativeBlockSize,
    sourceBlockSizeNote: source.sourceBlockSizeNote,
    sources: source.sources || [],
    requests: safeRequests,
    summary: {
      requests: safeRequests.length,
      totalInputTokens,
      averageInputTokens: safeRequests.length ? totalInputTokens / safeRequests.length : 0,
      uniqueBlocks: uniqueBlocks.size,
      totalBlocks,
    },
  };
}

export function infiniteCacheReuse(trace, options = {}) {
  const requests = Array.isArray(trace.requests) ? trace.requests : [];
  const warmupRequests = Math.min(requests.length, Math.max(0, Math.floor(toNumber(options.warmupRequests, requests.length * toNumber(options.warmupFraction, DEFAULT_WARMUP_FRACTION)))));
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

class MaxHeap {
  constructor() {
    this.items = [];
  }
  push(item) {
    this.items.push(item);
    this.bubbleUp(this.items.length - 1);
  }
  pop() {
    if (!this.items.length) return null;
    const top = this.items[0];
    const last = this.items.pop();
    if (this.items.length && last) {
      this.items[0] = last;
      this.sinkDown(0);
    }
    return top;
  }
  bubbleUp(index) {
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (this.items[parent].nextUse >= this.items[index].nextUse) break;
      [this.items[parent], this.items[index]] = [this.items[index], this.items[parent]];
      index = parent;
    }
  }
  sinkDown(index) {
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
}

function flattenTrace(trace) {
  const eventIds = [];
  const eventTokens = [];
  const eventRequest = [];
  const requestStarts = [];
  const requests = Array.isArray(trace.requests) ? trace.requests : [];
  requests.forEach((request, requestIndex) => {
    requestStarts.push(eventIds.length);
    request.inputBlocks.forEach((block) => {
      eventIds.push(block.id);
      eventTokens.push(toNumber(block.tokens, 0));
      eventRequest.push(requestIndex);
    });
  });
  return { eventIds, eventTokens, eventRequest, requestStarts, requestCount: requests.length };
}

export function buildExecutionPlan(trace, options = {}) {
  const flat = flattenTrace(trace);
  const warmupRequests = Math.min(
    flat.requestCount,
    Math.max(0, Math.floor(toNumber(options.warmupRequests, flat.requestCount * toNumber(options.warmupFraction, DEFAULT_WARMUP_FRACTION)))),
  );
  const nextPositions = new Float64Array(flat.eventIds.length);
  const lastSeen = new Map();
  const never = flat.eventIds.length + 1;
  let totalMeasuredTokens = 0;
  for (let index = flat.eventIds.length - 1; index >= 0; index -= 1) {
    const id = flat.eventIds[index];
    nextPositions[index] = lastSeen.has(id) ? lastSeen.get(id) : never;
    lastSeen.set(id, index);
    if (flat.eventRequest[index] >= warmupRequests) totalMeasuredTokens += flat.eventTokens[index];
  }
  return { ...flat, nextPositions, warmupRequests, totalMeasuredTokens };
}

function simulateFinitePolicy(trace, cacheBlocks, policy, options = {}) {
  if (options.plan && options.plan.nextInput) return lab.simulatePlanPolicy(options.plan, cacheBlocks, policy);
  if (options.plan) return simulatePlanPolicy(options.plan, cacheBlocks, policy);
  if (policy === "optimal") return simulateOptimalPolicy(trace, cacheBlocks, options);
  return lab.simulatePolicy(trace, cacheBlocks, policy, options);
}

function simulatePlannedPolicy(plan, cacheBlocks, policy) {
  return plan && plan.nextInput
    ? lab.simulatePlanPolicy(plan, cacheBlocks, policy)
    : simulatePlanPolicy(plan, cacheBlocks, policy);
}

function simulatePlanPolicy(plan, cacheBlocks, policy) {
  const normalizedPolicy = policy || "lru";
  const capacity = Math.max(0, Math.floor(toNumber(cacheBlocks, 0)));
  if (!plan.eventIds.length || capacity <= 0) {
    return {
      policy: normalizedPolicy,
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
  if (normalizedPolicy === "optimal") return simulateOptimalPlan(plan, capacity);

  const cache = new Map();
  const fifoQueue = [];
  let fifoHead = 0;
  let hitTokens = 0;
  let usefulCount = 0;
  let usefulCacheBlockSamples = 0;
  let usefulCacheSamples = 0;
  const never = plan.eventIds.length + 1;

  function updateUseful(previousNextUse, nextUse) {
    if (previousNextUse != null && previousNextUse < never) usefulCount -= 1;
    if (nextUse < never) usefulCount += 1;
  }

  function sampleUseful(index) {
    const requestIndex = plan.eventRequest[index];
    if (requestIndex < plan.warmupRequests) return;
    if (index !== plan.eventIds.length - 1 && plan.eventRequest[index + 1] === requestIndex) return;
    usefulCacheBlockSamples += usefulCount;
    usefulCacheSamples += 1;
  }

  function evictFifo() {
    while (cache.size >= capacity && fifoHead < fifoQueue.length) {
      const victim = fifoQueue[fifoHead];
      fifoHead += 1;
      const state = cache.get(victim);
      if (cache.delete(victim)) {
        updateUseful(state && state.nextUse, never);
        return;
      }
    }
  }

  for (let index = 0; index < plan.eventIds.length; index += 1) {
    const id = plan.eventIds[index];
    const nextUse = plan.nextPositions[index];
    const measured = plan.eventRequest[index] >= plan.warmupRequests;
    const previous = cache.get(id);
    const hit = previous !== undefined;
    if (measured && hit) hitTokens += plan.eventTokens[index];
    if (normalizedPolicy === "fifo") {
      if (!hit) {
        evictFifo();
        if (cache.size < capacity) {
          updateUseful(null, nextUse);
          cache.set(id, { nextUse });
          fifoQueue.push(id);
        }
      } else {
        updateUseful(previous.nextUse, nextUse);
        cache.set(id, { nextUse });
      }
    } else if (hit) {
      updateUseful(previous.nextUse, nextUse);
      cache.delete(id);
      cache.set(id, { nextUse });
    } else {
      while (cache.size >= capacity) {
        const victim = cache.keys().next().value;
        const victimState = cache.get(victim);
        cache.delete(victim);
        updateUseful(victimState && victimState.nextUse, never);
      }
      updateUseful(null, nextUse);
      cache.set(id, { nextUse });
    }
    sampleUseful(index);
  }

  return {
    policy: normalizedPolicy,
    cacheBlocks: capacity,
    warmupRequests: plan.warmupRequests,
    hitTokens,
    totalTokens: plan.totalMeasuredTokens,
    hitRate: plan.totalMeasuredTokens ? hitTokens / plan.totalMeasuredTokens : 0,
    usefulCacheBlockSamples,
    usefulCacheSamples,
    usefulCacheRate: usefulCacheSamples ? usefulCacheBlockSamples / (usefulCacheSamples * capacity) : 0,
  };
}

function simulateOptimalPolicy(trace, cacheBlocks, options = {}) {
  return simulateOptimalPlan(buildExecutionPlan(trace, options), Math.max(0, Math.floor(toNumber(cacheBlocks, 0))));
}

function simulateOptimalPlan(plan, capacity) {
  if (!plan.eventIds.length || capacity <= 0) {
    return {
      policy: "optimal",
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
  const cache = new Map();
  const heap = new MaxHeap();
  let hitTokens = 0;
  let usefulCount = 0;
  let usefulCacheBlockSamples = 0;
  let usefulCacheSamples = 0;
  const never = plan.eventIds.length + 1;

  function pushState(id, nextUse) {
    const previous = cache.get(id);
    if (previous && previous.nextUse < never) usefulCount -= 1;
    if (nextUse < never) usefulCount += 1;
    const state = { nextUse, version: (previous?.version || 0) + 1 };
    cache.set(id, state);
    heap.push({ id, nextUse, version: state.version });
  }

  function evictOne() {
    for (;;) {
      const candidate = heap.pop();
      if (!candidate) return;
      const current = cache.get(candidate.id);
      if (!current || current.version !== candidate.version || current.nextUse !== candidate.nextUse) continue;
      if (current.nextUse < never) usefulCount -= 1;
      cache.delete(candidate.id);
      return;
    }
  }

  function sampleUseful(index) {
    const requestIndex = plan.eventRequest[index];
    if (requestIndex < plan.warmupRequests) return;
    if (index !== plan.eventIds.length - 1 && plan.eventRequest[index + 1] === requestIndex) return;
    usefulCacheBlockSamples += usefulCount;
    usefulCacheSamples += 1;
  }

  for (let index = 0; index < plan.eventIds.length; index += 1) {
    const id = plan.eventIds[index];
    const measured = plan.eventRequest[index] >= plan.warmupRequests;
    const hit = cache.has(id);
    if (measured && hit) hitTokens += plan.eventTokens[index];
    if (hit) {
      pushState(id, plan.nextPositions[index]);
    } else {
      while (cache.size >= capacity) evictOne();
      if (cache.size < capacity) pushState(id, plan.nextPositions[index]);
    }
    sampleUseful(index);
  }

  return {
    policy: "optimal",
    cacheBlocks: capacity,
    warmupRequests: plan.warmupRequests,
    hitTokens,
    totalTokens: plan.totalMeasuredTokens,
    hitRate: plan.totalMeasuredTokens ? hitTokens / plan.totalMeasuredTokens : 0,
    usefulCacheBlockSamples,
    usefulCacheSamples,
    usefulCacheRate: usefulCacheSamples ? usefulCacheBlockSamples / (usefulCacheSamples * capacity) : 0,
  };
}

export function modelSettingFor(model, setting) {
  return {
    precision: setting.precision,
    indexerPrecision: setting.indexerPrecision,
    includeDraftKvCache: Boolean(setting.includeDraftKvCache),
    precisionOptions: setting.precisionOptions,
    indexerPrecisionOptions: setting.indexerPrecisionOptions,
    blockSize: setting.blockSize,
    capacityGiBValues: setting.capacityGiBValues || DEFAULT_CAPACITY_GIB_VALUES,
    warmupFraction: setting.warmupFraction ?? DEFAULT_WARMUP_FRACTION,
  };
}

export function precomputeSweep(trace, model, setting, options = {}) {
  const blockSize = positiveInteger(setting.blockSize, trace.blockSize || 64);
  const accountingSettings = {
    estimateTokens: positiveInteger(trace.summary && trace.summary.averageInputTokens, model.default_tokens || 4096),
    precision: setting.precision,
    indexerPrecision: setting.indexerPrecision,
    includeDraftKvCache: Boolean(setting.includeDraftKvCache),
    precisionOptions: setting.precisionOptions,
    indexerPrecisionOptions: setting.indexerPrecisionOptions,
  };
  const bytesPerToken = lab.estimateBytesPerToken(model, accountingSettings);
  const bytesPerBlock = bytesPerToken * blockSize;
  const capacityGiBValues = setting.capacityGiBValues || DEFAULT_CAPACITY_GIB_VALUES;
  const warmupFraction = setting.warmupFraction ?? DEFAULT_WARMUP_FRACTION;
  const policies = setting.policies || POLICIES;
  const plan = options.plan || buildExecutionPlan(trace, { warmupFraction });
  const simulationCache = options.simulationCache || null;
  const ceiling = options.ceiling || infiniteCacheReuse(trace, { warmupFraction });
  const ceilingStats = options.ceilingStats || simulatePlannedPolicy(plan, Math.max(1, trace.summary && trace.summary.uniqueBlocks ? trace.summary.uniqueBlocks : 1), "lru");
  const uniqueBlocks = trace && trace.summary && Number.isFinite(Number(trace.summary.uniqueBlocks))
    ? Number(trace.summary.uniqueBlocks)
    : Infinity;
  function resultFromCeiling(policy, cacheBlocks) {
    const usefulCacheBlockSamples = Number(ceilingStats.usefulCacheBlockSamples) || 0;
    const usefulCacheSamples = Number(ceilingStats.usefulCacheSamples) || 0;
    return {
      policy,
      cacheBlocks,
      warmupRequests: ceiling.warmupRequests,
      hitTokens: ceiling.hitTokens,
      totalTokens: ceiling.totalTokens,
      hitRate: ceiling.hitRate,
      usefulCacheBlockSamples,
      usefulCacheSamples,
      usefulCacheRate: cacheBlocks > 0 && usefulCacheSamples > 0 ? usefulCacheBlockSamples / (usefulCacheSamples * cacheBlocks) : 0,
    };
  }
  const points = capacityGiBValues.map((gib) => {
    const cacheBlocks = lab.cacheBlocksForGiB(gib, bytesPerBlock);
    const results = {};
    policies.forEach((policy) => {
      if (cacheBlocks >= uniqueBlocks) {
        results[policy] = resultFromCeiling(policy, cacheBlocks);
        return;
      }
      const cacheKey = `${policy}|${cacheBlocks}`;
      if (simulationCache && simulationCache.has(cacheKey)) {
        results[policy] = Object.assign({}, simulationCache.get(cacheKey));
      } else {
        const result = simulateFinitePolicy(trace, cacheBlocks, policy, { warmupFraction, plan });
        if (simulationCache) simulationCache.set(cacheKey, result);
        results[policy] = Object.assign({}, result);
      }
    });
    return { gib, cacheBlocks, results };
  });
  return {
    blockSize,
    bytesPerToken,
    bytesPerBlock,
    points,
    policies,
    reuseCeiling: ceiling.hitRate,
    warmupRequests: ceiling.warmupRequests,
    sourceKind: trace.sourceKind,
    generatedAt: options.generatedAt || new Date().toISOString(),
  };
}

export function loadModelsData(modelsPath = "data/kv_cache_calculator/models.yaml") {
  const absolute = path.resolve(modelsPath);
  const json = execFileSync("ruby", ["-ryaml", "-rjson", "-e", "data=YAML.load_file(ARGV[0]); puts JSON.generate(data)", absolute], {
    encoding: "utf8",
    maxBuffer: 80 * 1024 * 1024,
  });
  return JSON.parse(json);
}

export async function precomputeCurves(options = {}) {
  const modelsData = loadModelsData(options.modelsPath);
  const modelById = new Map(modelsData.models.map((model) => [model.id, model]));
  const traceIds = new Set(options.traceIds || TRACE_SOURCES.map((source) => source.id));
  const modelSettings = options.allModels
    ? allModelSettings(modelsData.models, {
        precisionOptions: modelsData.precision_options,
        indexerPrecisionOptions: modelsData.indexer_precision_options,
      })
    : options.modelIds && options.modelIds.length
    ? FEATURED_MODEL_SETTINGS.filter((setting) => options.modelIds.includes(setting.modelId))
    : options.modelSettings || FEATURED_MODEL_SETTINGS;
  const generatedAt = new Date().toISOString();
  const traces = {};
  for (const source of TRACE_SOURCES) {
    if (!traceIds.has(source.id)) continue;
    console.error(`[precompute] normalize ${source.id}`);
    const trace = await normalizeTraceSource(source, options);
    const warmupFraction = options.warmupFraction ?? DEFAULT_WARMUP_FRACTION;
    const plan = lab.buildExecutionPlan(trace, { warmupFraction });
    const ceiling = infiniteCacheReuse(trace, { warmupFraction });
    const ceilingStats = simulatePlannedPolicy(plan, Math.max(trace.summary.uniqueBlocks || 1, 1), "lru");
    const simulationCache = new Map();
    const modelSweeps = {};
    console.error(`[precompute] ${source.id}: ${trace.summary.requests} requests, ${trace.summary.totalBlocks} blocks, ${trace.summary.uniqueBlocks} unique`);
    for (const rawSetting of modelSettings) {
      const model = modelById.get(rawSetting.modelId);
      if (!model) continue;
      const setting = {
        ...rawSetting,
        blockSize: trace.blockSize,
        capacityGiBValues: options.capacityGiBValues || DEFAULT_CAPACITY_GIB_VALUES,
        warmupFraction,
        precisionOptions: modelsData.precision_options,
        indexerPrecisionOptions: modelsData.indexer_precision_options,
      };
      modelSweeps[modelSweepKey(rawSetting)] = {
        modelId: model.id,
        modelLabel: model.label,
        precision: rawSetting.precision,
        indexerPrecision: rawSetting.indexerPrecision || null,
        includeDraftKvCache: Boolean(rawSetting.includeDraftKvCache),
        ...precomputeSweep(trace, model, setting, { generatedAt, plan, simulationCache, ceiling, ceilingStats }),
      };
    }
    traces[source.id] = {
      id: source.id,
      label: source.label,
      scenario: source.scenario,
      sourceKind: source.sourceKind,
      nativeBlockSize: source.nativeBlockSize,
      sourceBlockSizeNote: source.sourceBlockSizeNote,
      sources: source.sources || [],
      summary: {
        ...trace.summary,
        warmupRequests: ceiling.warmupRequests,
        infiniteHitRate: ceiling.hitRate,
      },
      modelSweeps,
    };
  }
  return {
    metadata: {
      generated_at: generatedAt,
      mode: "precomputed_real_traces",
      note: "Raw traces are downloaded into a local cache and are not committed. This file contains derived hit-rate curves and provenance.",
      setting_mode: options.allModels ? "all_model_precision_indexer_draft_settings" : "selected_model_settings",
      capacity_gib_values: options.capacityGiBValues || DEFAULT_CAPACITY_GIB_VALUES,
      warmup_fraction: options.warmupFraction ?? DEFAULT_WARMUP_FRACTION,
      policies: POLICIES,
      sources: SOURCE_LINKS,
      reference_sources: REFERENCE_SOURCES,
    },
    traces,
  };
}

export async function writePrecomputedCurves(options = {}) {
  const outputPath = path.resolve(options.outputPath || DEFAULT_OUTPUT_PATH);
  const data = await precomputeCurves(options);
  await fsp.mkdir(path.dirname(outputPath), { recursive: true });
  await fsp.writeFile(outputPath, `${JSON.stringify(data, null, 2)}\n`);
  return { outputPath, data };
}

export function parseArgs(argv) {
  const args = { traceIds: [], modelIds: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--trace") args.traceIds.push(argv[++index]);
    else if (arg === "--model") args.modelIds.push(argv[++index]);
    else if (arg === "--all-models") args.allModels = true;
    else if (arg === "--cache-dir") args.cacheDir = argv[++index];
    else if (arg === "--output") args.outputPath = argv[++index];
    else if (arg === "--force") args.force = true;
    else if (arg === "--request-limit") args.requestLimit = Number(argv[++index]);
    else if (arg === "--warmup-fraction") args.warmupFraction = Number(argv[++index]);
  }
  if (!args.traceIds.length) delete args.traceIds;
  if (!args.modelIds.length) delete args.modelIds;
  return args;
}
