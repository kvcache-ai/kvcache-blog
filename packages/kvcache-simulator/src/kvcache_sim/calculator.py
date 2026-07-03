from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import shlex
from typing import Any

from ._resources import package_resource_path

BYTES_PER_GB = 1_000_000_000
BYTES_PER_GIB = 1024 ** 3
QWEN_LINEAR_CONV_BYTES_PER_ELEMENT = 2
QWEN_LINEAR_RECURRENT_BYTES_PER_ELEMENT = 4

DEFAULT_PRECISIONS = {
    "bf16_fp16": {"label": "BF16 / FP16", "bytes_per_element": 2.0},
    "fp8_int8": {"label": "FP8 / INT8", "bytes_per_element": 1.0},
    "fp4_int4": {"label": "FP4 / INT4", "bytes_per_element": 0.5},
}


@dataclass(frozen=True)
class CacheSizeResult:
    model_id: str
    model_label: str
    precision: str
    precision_label: str
    indexer_precision: str | None
    indexer_precision_label: str | None
    bytes_per_token: float
    bytes_per_block: float | None
    kv_bytes: float
    indexer_bytes: float
    total_bytes: float
    total_gib: float


def default_models_path() -> Path:
    return package_resource_path("models.yaml")


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in shlex.split(inner.replace(",", " "))]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?(\d+\.\d*|\d*\.\d+)([eE][+-]?\d+)?", value) or re.fullmatch(r"-?\d+[eE][+-]?\d+", value):
        return float(value)
    return value


def _split_key_value(line: str) -> tuple[str, Any]:
    key, value = line.split(":", 1)
    return key.strip(), _parse_scalar(value)


def _load_models_data_minimal_yaml(path: Path) -> dict[str, Any]:
    """Parse the small YAML subset used by data/kv_cache_calculator/models.yaml.

    This fallback keeps the CLI usable without PyYAML. It is intentionally
    narrow: top-level maps, lists of maps, and a nested scalar `fields` map.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    data: dict[str, Any] = {}
    current_section: str | None = None
    current_item: dict[str, Any] | None = None
    in_fields = False
    in_serving_references = False

    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        text = raw.strip()

        if indent == 0:
            key, value = _split_key_value(text)
            current_section = key
            current_item = None
            in_fields = False
            in_serving_references = False
            if value == {}:
                data[key] = [] if key in {"precision_options", "indexer_precision_options", "models"} else {}
            else:
                data[key] = value
            continue

        if current_section is None:
            continue

        if current_section == "metadata":
            if indent == 2:
                key, value = _split_key_value(text)
                data["metadata"][key] = value
                in_serving_references = key == "serving_references" and value == {}
            elif indent == 4 and in_serving_references:
                key, value = _split_key_value(text)
                data["metadata"].setdefault("serving_references", {})[key] = value
            continue

        if current_section in {"precision_options", "indexer_precision_options", "models"}:
            if indent == 2 and text.startswith("- "):
                current_item = {}
                data[current_section].append(current_item)
                key, value = _split_key_value(text[2:])
                current_item[key] = value
                in_fields = False
                continue
            if current_item is None:
                continue
            if indent == 4:
                key, value = _split_key_value(text)
                current_item[key] = value
                in_fields = key == "fields" and value == {}
                if in_fields:
                    current_item["fields"] = {}
                continue
            if indent == 6 and in_fields:
                key, value = _split_key_value(text)
                current_item["fields"][key] = value

    return data


def load_models_data(path: str | Path | None = None) -> dict[str, Any]:
    model_path = Path(path) if path else default_models_path()
    try:
        import yaml  # type: ignore

        with model_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except ModuleNotFoundError:
        return _load_models_data_minimal_yaml(model_path)


def precision_options(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["id"]: {
            "label": item.get("label", item["id"]),
            "bytes_per_element": float(item.get("bytes_per_element", item.get("bytesPerElement", 0))),
        }
        for item in data.get("precision_options", [])
    } or DEFAULT_PRECISIONS


def indexer_precision_options(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = data.get("indexer_precision_options") or data.get("precision_options") or []
    return {
        item["id"]: {
            "label": item.get("label", item["id"]),
            "bytes_per_element": float(item.get("bytes_per_element", item.get("bytesPerElement", 0))),
        }
        for item in items
    } or precision_options(data)


def models_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {model["id"]: model for model in data.get("models", [])}


def _safe_number(value: Any, fallback: float = 0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if math.isfinite(parsed) else fallback


def _positive_int(value: Any, fallback: int) -> int:
    parsed = math.floor(_safe_number(value, fallback))
    return max(1, parsed) if parsed > 0 else max(1, fallback)


def _field(model: dict[str, Any], name: str) -> float:
    fields = model.get("fields") or {}
    if name not in fields:
        raise ValueError(f"Model {model.get('id', '')} is missing numeric field {name}")
    parsed = _safe_number(fields[name], math.nan)
    if not math.isfinite(parsed):
        raise ValueError(f"Model {model.get('id', '')} is missing numeric field {name}")
    return parsed


def _optional_field(model: dict[str, Any], name: str, fallback: float) -> float:
    fields = model.get("fields") or {}
    return _safe_number(fields.get(name), fallback) if name in fields else fallback


def _is_deepseek_v4(model: dict[str, Any]) -> bool:
    return model.get("formula") == "deepseek_v4_hybrid"


def _has_indexer_cache(model: dict[str, Any]) -> bool:
    fields = model.get("fields") or {}
    return math.isfinite(_safe_number(fields.get("index_head_dim"), math.nan))


def _draft_layer_count(model: dict[str, Any]) -> int:
    fields = model.get("fields") or {}
    if fields.get("disable_draft_kv_cache") is True:
        return 0
    nextn_layers = int(_safe_number(fields.get("num_nextn_predict_layers"), 0))
    if nextn_layers > 0:
        return nextn_layers
    if fields.get("use_mtp") is True:
        return int(_safe_number(fields.get("num_mtp_modules"), 0) * _safe_number(fields.get("mtp_transformer_layers"), 0))
    return 0


def has_draft_kv_cache(model: dict[str, Any]) -> bool:
    fields = model.get("fields") or {}
    if _is_deepseek_v4(model):
        layers = int(_safe_number(fields.get("num_hidden_layers"), 0))
        ratios = fields.get("compress_ratios")
        return isinstance(ratios, list) and len(ratios) > layers
    return _draft_layer_count(model) > 0


def _fixed_indexer_precision_id(model: dict[str, Any]) -> str | None:
    value = (model.get("fields") or {}).get("indexer_fixed_precision_id")
    return value if isinstance(value, str) else None


def default_precision_id(model: dict[str, Any], options: dict[str, dict[str, Any]]) -> str:
    if _is_deepseek_v4(model) and "fp8_int8" in options:
        return "fp8_int8"
    if "bf16_fp16" in options:
        return "bf16_fp16"
    return next(iter(options))


def default_indexer_precision_id(model: dict[str, Any], options: dict[str, dict[str, Any]], fallback_precision: str | None) -> str:
    fixed = _fixed_indexer_precision_id(model)
    if fixed and fixed in options:
        return fixed
    if _is_deepseek_v4(model) and "fp4_int4" in options:
        return "fp4_int4"
    if fallback_precision and fallback_precision in options:
        return fallback_precision
    if "bf16_fp16" in options:
        return "bf16_fp16"
    return "fp4_int4" if "fp4_int4" in options else next(iter(options))


def _indexer_layer_plan(model: dict[str, Any], layers: int, draft_layers: int) -> tuple[int, int, int, int]:
    main = int(_optional_field(model, "indexer_full_layers", layers))
    shared = int(_optional_field(model, "indexer_shared_layers", max(0, layers - main)))
    draft = int(_optional_field(model, "draft_indexer_layers", draft_layers)) if draft_layers > 0 else 0
    return main, shared, draft, main + draft


def _calculate_byte_groups(model: dict[str, Any], tokens: int, include_draft_kv_cache: bool, include_linear_attention_state: bool) -> tuple[float, list[dict[str, Any]]]:
    formula = model.get("formula")
    fields = model.get("fields") or {}
    draft_layers = _draft_layer_count(model) if include_draft_kv_cache else 0

    if formula == "standard_gqa":
        layers = int(_field(model, "num_hidden_layers")) + draft_layers
        elements_per_token = layers * 2 * _field(model, "num_key_value_heads") * _field(model, "head_dim")
        return elements_per_token, [{"role": "kv", "label": "KV cache", "elements": elements_per_token * tokens}]

    if formula == "mla":
        layers = int(_field(model, "num_hidden_layers")) + draft_layers
        elements_per_token = layers * (_field(model, "kv_lora_rank") + _field(model, "qk_rope_head_dim"))
        return elements_per_token, [{"role": "kv", "label": "KV cache", "elements": elements_per_token * tokens}]

    if formula == "dsa_mla":
        layers = int(_field(model, "num_hidden_layers"))
        active_layers = layers + draft_layers
        _, _, _, active_indexer_layers = _indexer_layer_plan(model, layers, draft_layers)
        kv_elements_per_token = active_layers * (_field(model, "kv_lora_rank") + _field(model, "qk_rope_head_dim"))
        indexer_elements_per_token = active_indexer_layers * _field(model, "index_head_dim")
        return kv_elements_per_token + indexer_elements_per_token, [
            {"role": "kv", "label": "KV cache", "elements": kv_elements_per_token * tokens},
            {"role": "indexer", "label": "Indexer cache", "elements": indexer_elements_per_token * tokens},
        ]

    if formula == "qwen_linear_full_hybrid":
        full_layers = _field(model, "full_attention_layers")
        kv_heads = _field(model, "num_key_value_heads")
        head_dim = _field(model, "head_dim")
        elements_per_token = full_layers * 2 * kv_heads * head_dim
        groups = [{"role": "kv", "label": "Full-attention KV cache", "elements": elements_per_token * tokens}]
        if include_linear_attention_state:
            linear_layers = _field(model, "linear_attention_layers")
            conv_kernel = _field(model, "linear_conv_kernel_dim")
            key_heads = _field(model, "linear_num_key_heads")
            key_dim = _field(model, "linear_key_head_dim")
            value_heads = _field(model, "linear_num_value_heads")
            value_dim = _field(model, "linear_value_head_dim")
            conv_elements = linear_layers * conv_kernel * (2 * key_heads * key_dim + value_heads * value_dim)
            recurrent_elements = linear_layers * value_heads * key_dim * value_dim
            groups.append({
                "role": "linear_state",
                "label": "Linear-attention state",
                "bytes_per_sequence": conv_elements * QWEN_LINEAR_CONV_BYTES_PER_ELEMENT + recurrent_elements * QWEN_LINEAR_RECURRENT_BYTES_PER_ELEMENT,
            })
        return elements_per_token, groups

    if formula == "mixed_full_sliding_gqa":
        full_layers = _field(model, "full_attention_layers")
        sliding_layers = _field(model, "sliding_attention_layers")
        kv_heads = _field(model, "num_key_value_heads")
        head_dim = _field(model, "head_dim")
        full_kv_heads = _optional_field(model, "num_global_key_value_heads", kv_heads)
        full_head_dim = _optional_field(model, "global_head_dim", head_dim)
        full_v_dim = _optional_field(model, "global_v_head_dim", _optional_field(model, "v_head_dim", full_head_dim))
        sliding_kv_heads = _optional_field(model, "swa_num_key_value_heads", _optional_field(model, "sliding_num_key_value_heads", kv_heads))
        sliding_head_dim = _optional_field(model, "swa_head_dim", _optional_field(model, "sliding_head_dim", head_dim))
        sliding_v_dim = _optional_field(model, "swa_v_head_dim", _optional_field(model, "sliding_v_head_dim", _optional_field(model, "v_head_dim", sliding_head_dim)))
        retained_sliding_tokens = min(tokens, int(_field(model, "sliding_window")))
        full_elements = tokens * full_layers * full_kv_heads * (full_head_dim + full_v_dim)
        sliding_elements = retained_sliding_tokens * sliding_layers * sliding_kv_heads * (sliding_head_dim + sliding_v_dim)
        return (full_elements + sliding_elements) / tokens, [
            {"role": "kv", "label": "Full-attention KV cache", "elements": full_elements},
            {"role": "kv", "label": "Sliding-window KV cache", "elements": sliding_elements},
        ]

    if formula == "minimax_msa":
        layers = _field(model, "num_hidden_layers")
        sparse_layers = _field(model, "sparse_attention_layers")
        kv_elements_per_token = layers * 2 * _field(model, "num_key_value_heads") * _field(model, "head_dim")
        indexer_elements_per_token = sparse_layers * _field(model, "index_head_dim")
        return kv_elements_per_token + indexer_elements_per_token, [
            {"role": "kv", "label": "KV cache", "elements": kv_elements_per_token * tokens},
            {"role": "indexer", "label": "Indexer cache", "elements": indexer_elements_per_token * tokens},
        ]

    if formula == "deepseek_v4_hybrid":
        head_dim = _field(model, "head_dim")
        index_dim = _field(model, "index_head_dim")
        sliding_window = _field(model, "sliding_window")
        layers = int(_field(model, "num_hidden_layers"))
        ratios = [float(ratio) for ratio in fields.get("compress_ratios", [])]
        active_ratios = ratios[:layers] + (ratios[layers:] if include_draft_kv_cache else [])
        if not active_ratios:
            raise ValueError(f"Model {model.get('id', '')} is missing compress_ratios")
        window_elements = 0.0
        compressed_elements = 0.0
        indexer_elements = 0.0
        for ratio in active_ratios:
            window_elements += sliding_window * head_dim
            if ratio > 0:
                compressed_elements += math.floor(tokens / ratio) * head_dim
            if ratio == 4:
                indexer_elements += math.floor(tokens / 4) * index_dim
        attention_elements = window_elements + compressed_elements
        return (attention_elements + indexer_elements) / tokens, [
            {"role": "kv", "label": "KV cache", "elements": attention_elements},
            {"role": "indexer", "label": "Indexer cache", "elements": indexer_elements},
        ]

    raise ValueError(f"Unsupported formula: {formula}")


def _bytes_for_group(group: dict[str, Any], kv_precision_bytes: float, indexer_precision_bytes: float | None) -> float:
    if "bytes_per_sequence" in group:
        return float(group["bytes_per_sequence"])
    role = group["role"]
    if role == "indexer" and indexer_precision_bytes is not None:
        bytes_per_element = indexer_precision_bytes
    else:
        bytes_per_element = kv_precision_bytes
    return float(group["elements"]) * bytes_per_element


def calculate_cache_size(
    model: dict[str, Any],
    *,
    tokens: int,
    precision: str | None = None,
    indexer_precision: str | None = None,
    block_size: int | None = None,
    include_draft_kv_cache: bool = False,
    include_linear_attention_state: bool = False,
    models_data: dict[str, Any] | None = None,
) -> CacheSizeResult:
    data = models_data or load_models_data()
    precision_by_id = precision_options(data)
    indexer_precision_by_id = indexer_precision_options(data)
    precision_id = precision or default_precision_id(model, precision_by_id)
    if precision_id not in precision_by_id:
        raise ValueError(f"Unknown KV precision: {precision_id}")
    precision_profile = precision_by_id[precision_id]
    kv_precision_bytes = float(precision_profile["bytes_per_element"])

    indexer_precision_id: str | None = None
    indexer_precision_label: str | None = None
    indexer_precision_bytes: float | None = None
    if _has_indexer_cache(model):
        fixed_indexer_precision = _fixed_indexer_precision_id(model)
        indexer_precision_id = fixed_indexer_precision or indexer_precision or default_indexer_precision_id(model, indexer_precision_by_id, precision_id)
        if indexer_precision_id not in indexer_precision_by_id:
            raise ValueError(f"Unknown indexer precision: {indexer_precision_id}")
        indexer_profile = indexer_precision_by_id[indexer_precision_id]
        indexer_precision_label = str(indexer_profile["label"])
        indexer_precision_bytes = float(indexer_profile["bytes_per_element"])

    active_draft = include_draft_kv_cache and has_draft_kv_cache(model)
    tokens = _positive_int(tokens, int(model.get("default_tokens") or 4096))
    _, groups = _calculate_byte_groups(model, tokens, active_draft, include_linear_attention_state)
    kv_bytes = 0.0
    indexer_bytes = 0.0
    total_bytes = 0.0
    for group in groups:
        group_bytes = _bytes_for_group(group, kv_precision_bytes, indexer_precision_bytes)
        total_bytes += group_bytes
        if group["role"] == "indexer":
            indexer_bytes += group_bytes
        else:
            kv_bytes += group_bytes

    bytes_per_token = total_bytes / tokens
    return CacheSizeResult(
        model_id=str(model["id"]),
        model_label=str(model.get("label") or model["id"]),
        precision=precision_id,
        precision_label=str(precision_profile["label"]),
        indexer_precision=indexer_precision_id,
        indexer_precision_label=indexer_precision_label,
        bytes_per_token=bytes_per_token,
        bytes_per_block=bytes_per_token * block_size if block_size else None,
        kv_bytes=kv_bytes,
        indexer_bytes=indexer_bytes,
        total_bytes=total_bytes,
        total_gib=total_bytes / BYTES_PER_GIB,
    )
