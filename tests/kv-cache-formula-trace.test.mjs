import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import { loadModelsData } from '../scripts/lib/kv-cache-lab-traces.mjs';
const require = createRequire(import.meta.url);
const { calculate, formulaTrace } = require('../assets/js/kv-cache-calculator.js');
const data = loadModelsData(new URL('../data/kv_cache_calculator/models.yaml', import.meta.url).pathname);
const options = { precisionOptions: data.precision_options, indexerPrecisionOptions: data.indexer_precision_options, recurrentStatePrecisionOptions: data.recurrent_state_precision_options };
const model = (id) => data.models.find((entry) => entry.id === id);
const trace = (m, input) => { const result = calculate(m, input, options); return { result, rows: Object.fromEntries(formulaTrace(m, input, result, options).map((row) => [row.name, row])) }; };

test('GLM substitutions follow draft, tokens, precision and checkpoint controls', () => {
  const m = model('glm-5.3-flash');
  const base = trace(m, { tokens: 1024, sequences: 1, precision: 'bf16_fp16', includeLinearAttentionState: true });
  assert.equal(base.rows.main_layers.value, 45);
  assert.equal(base.rows.active_full_attention_layers.value, 11);
  assert.match(base.rows.active_full_attention_layers.substitution, /false \? 1 : 0/);
  assert.equal(base.rows.mla_kv_bytes.value, 11534336);
  assert.deepEqual(base.rows.mla_kv_bytes.terms.filter((term) => term.variable).map(({label, value}) => [label, value]), [
    ['tokens', '1024'], ['sequences', '1'], ['active_dsa_mla_layers', '11'],
    ['kv_lora_rank', '512'], ['qk_rope_head_dim', '0'], ['kv_precision_bytes', '2'],
  ]);
  assert.equal(base.rows.indexer_pools.value, 256);
  assert.equal(base.rows.indexer_bytes.value, 360448);
  assert.equal(base.rows.total_bytes.value, base.result.totalBytes);
  const updated = trace(m, { tokens: 1025, sequences: 2, precision: 'fp8_int8', includeDraftKvCache: true, includeLinearAttentionState: true, kdaCheckpointPolicy: 'fixed_interval', kdaCheckpointInterval: 512 });
  assert.match(updated.rows.active_full_attention_layers.substitution, /true \? 1 : 0/);
  assert.equal(updated.rows.active_full_attention_layers.terms.find((term) => term.label === 'draft_layers_if_enabled').value, '1');
  assert.equal(updated.rows.draft_layers_if_enabled.terms.find((term) => term.label === 'include_draft_kv_cache').value, 'true');
  assert.equal(updated.rows.active_full_attention_layers.value, 12);
  assert.equal(updated.rows.active_indexer_layers.value, 12);
  assert.equal(updated.rows.index_tail_bytes.value, updated.result.cacheGroups.find((group) => group.role === 'index_tail').bytes);
  assert.equal(updated.rows.kda_checkpoint_count.value, 3);
  assert.equal(updated.rows.total_bytes.value, updated.result.totalBytes);
  assert.ok(updated.rows.total_bytes.substitution);
});

test('representative formulas substitute to totals with state and grouped precision', () => {
  for (const id of ['dots3-note-prev', 'deepseek-v4.1-flash', 'deepseek-v4-pro', 'deepseek-v3.2', 'deepseek-v3', 'glm-5.3-flash', 'kimi-k3', 'inkling', 'qwen3.8-2.4t-a95b', 'qwen3-235b-a22b', 'gemma-4-e2b', 'minimax-m3', 'qwen3.8-flash-next']) {
    const m = model(id);
    const input = { tokens: 1025, sequences: 2, includeDraftKvCache: true, includeSwaCache: true, includeLinearAttentionState: true, includeSconvState: true };
    const {result, rows} = trace(m, input);
    assert.equal(rows.total_bytes.value, result.totalBytes, id);
    assert.ok(rows.total_bytes.substitution, `${id}: total cannot be substituted`);
  }
});

test('indexer fallback follows the chosen KV precision in displayed calculations', () => {
  const { result, rows } = trace(model('deepseek-v3.2'), { tokens: 1025, sequences: 2, precision: 'fp8_int8' });
  assert.equal(rows.indexer_bytes.value, result.indexerBytes);
  assert.ok(rows.total_bytes.substitution);
  assert.equal(rows.total_bytes.value, result.totalBytes);
});

test('KDA subtotal matches the summary for Kimi and GLM', () => {
  for (const id of ['kimi-k3', 'glm-5.3-flash']) {
    const m = model(id);
    const input = { tokens: 1024, sequences: 2, includeLinearAttentionState: true, kdaCheckpointPolicy: 'fixed_interval', kdaCheckpointInterval: 512 };
    const {result, rows} = trace(m, input);
    assert.equal(rows.kda_checkpoint_state_bytes.value, result.cacheGroups.find((group) => group.label === 'KDA checkpoint state').bytes);
    assert.equal(rows.kda_checkpoint_state_bytes.value, rows.kda_conv_state_bytes.value + rows.kda_recurrent_state_bytes.value);
    assert.equal(trace(m, {...input, includeLinearAttentionState: false}).rows.kda_checkpoint_state_bytes, undefined);
  }
});

test('DeepSeek V4 compressed elements expose the layer groups and update with tokens', () => {
  const m = model('deepseek-v4-flash');
  for (const tokens of [1024, 1025, 2048]) {
    const {result, rows} = trace(m, {tokens, sequences: 2, precision: 'fp8_int8'});
    const terms = rows.compressed_elements.terms.filter((term) => term.variable);
    assert.deepEqual(terms.map(({label, value}) => [label, value]), [
      ['ratio_4_layers', '21'], ['tokens', String(tokens)], ['compress_ratio', '4'],
      ['ratio_128_layers', '20'], ['tokens', String(tokens)], ['compress_ratio', '128'], ['head_dim', '512'],
    ]);
    assert.equal(rows.compressed_elements.value, (21 * Math.floor(tokens / 4) + 20 * Math.floor(tokens / 128)) * 512);
    assert.equal(rows.compressed_kv_bytes.value, rows.compressed_elements.value);
    assert.equal(rows.total_bytes.value, result.totalBytes);
  }
});

test('checkpoint trace uses the same effective interval as the calculation', () => {
  for (const [id, prefix, name] of [
    ['kimi-k3', 'kda', 'kda_checkpoint_count'], ['glm-5.3-flash', 'kda', 'kda_checkpoint_count'],
    ['qwen3.8-2.4t-a95b', 'qwen', 'linear_state_checkpoint_count'], ['qwen3.8-flash-next', 'qwen', 'checkpoint_count'],
    ['inkling', 'sconv', 'sconv_checkpoint_count'],
  ]) {
    const m = model(id);
    const base = { tokens: 32768, includeLinearAttentionState: true, includeSconvState: true, [`${prefix}CheckpointPolicy`]: 'fixed_interval' };
    for (const [interval, expected, display] of [['', 1, 'infinity'], ['10240', 4, '10240']]) {
      const { rows } = trace(m, { ...base, [`${prefix}CheckpointInterval`]: interval });
      assert.equal(rows[name].value, expected, id);
      assert.equal(rows[name].terms.find((term) => term.label === 'checkpoint_interval').value, display, id);
    }
    if (id === 'qwen3.8-flash-next') {
      const { rows } = trace(m, { ...base, includeLinearAttentionState: false });
      assert.equal(rows[name].value, 0);
      assert.equal(rows[name].terms.find((term) => term.label === 'state_enabled').value, 'false');
    }
  }
  const { rows } = trace(model('inkling-2-small'), { tokens: 1024 });
  assert.doesNotMatch(rows.active_global_layers.substitution, /undefined/);
  assert.doesNotMatch(rows.active_swa_layers.substitution, /undefined/);
});
