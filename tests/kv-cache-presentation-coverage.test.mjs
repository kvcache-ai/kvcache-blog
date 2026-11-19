import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { loadModelsData } from '../scripts/lib/kv-cache-lab-traces.mjs';

const require = createRequire(import.meta.url);
const { calculate, expandModelCatalog, formulaTrace, metricsForResult, precisionOptionIdsForModel } = require('../assets/js/kv-cache-calculator.js');
const data = loadModelsData(new URL('../data/kv_cache_calculator/models.yaml', import.meta.url).pathname);
const catalog = JSON.parse(readFileSync(new URL('../data/kv_cache_calculator/model_catalog.json', import.meta.url), 'utf8'));
const models = expandModelCatalog(data.models, catalog);
const options = { precisionOptions: data.precision_options, indexerPrecisionOptions: data.indexer_precision_options, recurrentStatePrecisionOptions: data.recurrent_state_precision_options };

function evaluateTerms(terms) {
  const expression = terms.map((term) => term.value).join(' ').replace(/\bx\b/g, '*').replace(/\binfinity\b/gi, 'Infinity');
  const punctuation = expression.replace(/\b(?:floor|ceil|min|max|count|true|false|Infinity)\b/g, '');
  assert.match(punctuation, /^[\d.eE+\-*/%()?:><=!,&|\s]*$/, `unexpected expression: ${expression}`);
  return Number(Function('floor', 'ceil', 'min', 'max', 'count', `return (${expression})`)(Math.floor, Math.ceil, Math.min, Math.max, (...args) => args.length));
}

test('each visible metric has an equal, fully sourced formula across model and precision options', () => {
  let checked = 0;
  for (const model of models) {
    const kvIds = precisionOptionIdsForModel(model, data.precision_options, 'kv');
    const indexIds = [undefined, ...precisionOptionIdsForModel(model, data.indexer_precision_options, 'indexer')];
    for (const precision of kvIds) for (const indexerPrecision of indexIds) for (const enabled of [false, true]) {
      const input = { tokens: 1025, sequences: 2, precision, indexerPrecision, includeDraftKvCache: enabled, includeSwaCache: enabled, includeLinearAttentionState: enabled, includeSconvState: enabled };
      const result = calculate(model, input, options);
      const rows = formulaTrace(model, input, result, options);
      const context = `${model.id}, ${precision}/${indexerPrecision}, enabled=${enabled}`;
      for (const row of rows) {
        assert.ok(row.terms?.length, `${context}: ${row.name} has no aligned terms`);
        const actual = evaluateTerms(row.terms);
        assert.ok(Number.isFinite(actual), `${context}: ${row.name} is not finite`);
        assert.ok(Math.abs(actual - row.value) <= Math.max(1e-6, Math.abs(row.value) * 1e-12), `${context}: ${row.name} evaluates to ${actual}, displayed ${row.value}`);
      }
      for (const [label, value] of metricsForResult(result)) {
        const row = rows.find((entry) => entry.metricLabel === label);
        assert.ok(row, `${context}: missing ${label}`);
        assert.ok(row.terms?.length, `${context}: missing terms for ${label}`);
        assert.ok(Math.abs(row.value - value) < 1e-6, `${context}: ${label} expected ${value}, got ${row.value}`);
        for (const term of row.terms.filter((entry) => entry.variable)) assert.notEqual(term.value, 'undefined', `${context}: ${term.label}`);
      }
      checked++;
    }
  }
  assert.ok(checked > 1000);
});
