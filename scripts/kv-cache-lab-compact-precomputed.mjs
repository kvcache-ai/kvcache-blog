#!/usr/bin/env node
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { compactPrecomputed } from "./lib/kv-cache-lab-precomputed.mjs";

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--input") args.inputPath = argv[++index];
    else if (arg === "--output") args.outputPath = argv[++index];
  }
  if (!args.inputPath) throw new Error("Missing --input");
  if (!args.outputPath) args.outputPath = args.inputPath;
  return args;
}

export async function compactPrecomputedFile(options) {
  const inputPath = path.resolve(options.inputPath);
  const outputPath = path.resolve(options.outputPath);
  const source = JSON.parse(await fsp.readFile(inputPath, "utf8"));
  const compact = compactPrecomputed(source);
  await fsp.mkdir(path.dirname(outputPath), { recursive: true });
  await fsp.writeFile(outputPath, `${JSON.stringify(compact, null, 2)}\n`);
  return { inputPath, outputPath, data: compact };
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);
if (isMain) {
  const result = await compactPrecomputedFile(parseArgs(process.argv.slice(2)));
  console.log(JSON.stringify({
    inputPath: result.inputPath,
    outputPath: result.outputPath,
    traces: Object.keys(result.data.traces || {}).length,
  }, null, 2));
}
