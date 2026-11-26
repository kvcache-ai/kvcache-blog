import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import test from "node:test";

const BASELINE_COMMIT = "85ad63fd97899a183fda37288441bcc45f69c9c5";
const HEAD_MODEL_PATH = "data/kv_cache_calculator/models.yaml";
const modelPaths = [
  new URL("../data/kv_cache_calculator/models.yaml", import.meta.url),
  new URL("../packages/kvcache-simulator/src/kvcache_sim/resources/models.yaml", import.meta.url),
];
const catalogPath = new URL("../data/kv_cache_calculator/model_catalog.json", import.meta.url);
const keptModelIds = [
  "cohere-north-micro-vision-instruct",
  "cohere-north-mini-code-1.0",
  "deepseek-v4.1-flash-1017",
  "gemma-4-12b",
  "inkling-2-small",
  "kimi-k3.1-pro",
  "mimo-v2.5-asr-text-decoder",
  "mimo-v2.6-distill-qwen-9b",
  "mimo-v2.6-flash",
  "mimo-v2.6-pro",
  "minimax-h3-image-pe",
  "minimax-m3.1-flash",
  "qwen-image-2.1-pe-decoder",
  "qwen3.8-flash-next",
];
const keptAliases = new Map([
  ["Qwen/Qwen-AgentWorld-35B-A3B", "qwen3.5-35b-a3b"],
  ["deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "deepseek-v4-flash"],
  ["Qwen/Qwen-Image-2.1-PE-I2I", "qwen-image-2.1-pe-decoder"],
  ["Qwen/Qwen-Drive-1.0-4B", "qwen3.5-4b"],
]);

function modelBlocks(text, source) {
  const lines = [];
  let offset = 0;
  for (const line of text.matchAll(/.*(?:\r?\n|$)/g)) {
    if (line[0] === "") continue;
    lines.push({ start: offset, end: offset + line[0].length, content: line[0].replace(/\r?\n$/, "") });
    offset += line[0].length;
  }
  const headerIndex = lines.findIndex(({ content }) => content === "models:");
  assert.notEqual(headerIndex, -1, `${source}: missing models section`);
  const header = lines[headerIndex];
  const end = lines.slice(headerIndex + 1).find((line) => /^[A-Za-z_][A-Za-z0-9_.-]*\s*:/.test(line.content))?.start ?? text.length;
  const bodyStart = header.end;
  const body = text.slice(bodyStart, end);
  const bodyLines = [...body.matchAll(/.*(?:\r?\n|$)/g)]
    .filter((line) => line[0] !== "")
    .map((line) => ({ start: line.index, content: line[0].replace(/\r?\n$/, "") }));
  const starts = bodyLines.filter(({ content }) => /^  - id: /.test(content));
  const blocks = [];
  for (let index = 0; index < starts.length; index += 1) {
    const current = starts[index];
    const next = starts[index + 1];
    const rawEnd = next?.start ?? body.length;
    const raw = body.slice(current.start, rawEnd);
    blocks.push({ id: current.content.slice("  - id: ".length), raw });
  }
  return blocks;
}

test("scope retains only the 14 accepted new profiles and four distinct aliases", () => {
  const headYaml = execFileSync("git", ["show", `${BASELINE_COMMIT}:${HEAD_MODEL_PATH}`], { encoding: "utf8" });
  const currentYamls = modelPaths.map((path) => readFileSync(path, "utf8"));
  assert.equal(currentYamls[0], currentYamls[1], "the two shipped models.yaml files must stay byte-identical");

  const baseBlocks = modelBlocks(headYaml, "HEAD");
  const currentBlocks = modelBlocks(currentYamls[0], "current");
  const expectedBaseBlocks = baseBlocks.map((block) =>
    ["deepseek-v4-pro", "deepseek-v4-flash"].includes(block.id)
      ? { ...block, raw: block.raw.replace('      head_dim: 512\n', '      head_dim: 512\n      kv_precision_ids: ["bf16_fp16", "fp8_int8", "fp8_sglang_packed"]\n') }
      : block,
  );
  assert.equal(baseBlocks.length, 62);
  assert.equal(currentBlocks.length, 76);
  const withoutDisplayMetadata = ({ id, raw }) => ({
    id,
    raw: raw
      .replace(/^    label: .*\n/gm, "")
      .replace(/^    (?:official_release_(?:date|source_url)|weights_public_date(?:_source_url)?): .*\n/gm, ""),
  });
  assert.deepEqual(
    currentBlocks.slice(0, baseBlocks.length).map(withoutDisplayMetadata),
    expectedBaseBlocks.map(withoutDisplayMetadata),
    "existing model blocks may only change display metadata and the two DeepSeek V4 KV precision allowlists",
  );

  const actualAddedIds = currentBlocks.slice(baseBlocks.length).map(({ id }) => id).sort();
  assert.deepEqual(actualAddedIds, [...keptModelIds].sort(), "no skipped or unreviewed preset may remain");

  const catalog = JSON.parse(readFileSync(catalogPath, "utf8"));
  assert.equal(catalog.length, 4);
  assert.deepEqual(
    catalog.map(({ official_model_id, preset_id }) => [official_model_id, preset_id]).sort(([a], [b]) => a.localeCompare(b)),
    [...keptAliases].map(([id, preset]) => [id, preset]).sort(([a], [b]) => a.localeCompare(b)),
  );
  const modelIds = new Set(currentBlocks.map(({ id }) => id));
  for (const entry of catalog) assert.ok(modelIds.has(entry.preset_id), `dangling preset ${entry.preset_id}`);
});
