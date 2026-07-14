#!/usr/bin/env node
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  compactPrecomputed,
  mergeCompactPrecomputed,
} from "./lib/kv-cache-lab-precomputed.mjs";

function parseArgs(argv) {
  const args = { inputs: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--base") args.base = argv[++index];
    else if (arg === "--input") args.inputs.push(argv[++index]);
    else if (arg === "--output") args.output = argv[++index];
  }
  if (!args.output) throw new Error("Missing --output");
  if (!args.inputs.length) throw new Error("At least one --input is required");
  return args;
}

async function readJsonIfExists(filePath, fallback) {
  if (!filePath || !fs.existsSync(filePath)) return fallback;
  return JSON.parse(await fsp.readFile(filePath, "utf8"));
}

export function mergePrecomputed(baseData, inputDataList) {
  let output = compactPrecomputed(baseData || { metadata: {}, traces: {} });

  for (const inputData of inputDataList) {
    if (!inputData) continue;
    output = mergeCompactPrecomputed(output, inputData);
  }

  output.metadata.merged_at = new Date().toISOString();
  return output;
}

export async function mergePrecomputedFiles(options) {
  const base = await readJsonIfExists(options.base, { metadata: {}, traces: {} });
  const inputs = [];
  for (const inputPath of options.inputs) {
    inputs.push(await readJsonIfExists(inputPath, null));
  }
  const output = mergePrecomputed(base, inputs);
  const outputPath = path.resolve(options.output);
  await fsp.mkdir(path.dirname(outputPath), { recursive: true });
  await fsp.writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`);
  return { outputPath, output };
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);
if (isMain) {
  const { outputPath, output } = await mergePrecomputedFiles(parseArgs(process.argv.slice(2)));
  console.log(JSON.stringify({ outputPath, traces: Object.keys(output.traces || {}) }, null, 2));
}
