#!/usr/bin/env node
import { fetchTraceSources, parseArgs } from "./lib/kv-cache-lab-traces.mjs";

const options = parseArgs(process.argv.slice(2));
const manifest = await fetchTraceSources(options);
console.log(JSON.stringify({ files: manifest }, null, 2));
