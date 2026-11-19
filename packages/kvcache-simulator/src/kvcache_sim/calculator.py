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
KIMI_KDA_CONV_BYTES_PER_ELEMENT = 2
KIMI_KDA_RECURRENT_BYTES_PER_ELEMENT = 4
DEFAULT_RECURRENT_STATE_PRECISIONS = {
    "bf16_fp16": {"label": "BF16 / FP16", "bytes_per_element": 2.0},
    "fp32": {"label": "FP32", "bytes_per_element": 4.0},
}
LINEAR_STATE_FORMULAS = ("qwen_linear_full_hybrid", "kimi_kda_mla_hybrid", "kimi_kda_dsa_mla_hybrid")

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
    hit_rate_bytes_per_token: float | None
    recurrent_state_precision: str | None = None
    recurrent_state_precision_label: str | None = None


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
    return _precision_options(data.get("precision_options", [])) or DEFAULT_PRECISIONS


def indexer_precision_options(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = data.get("indexer_precision_options") or data.get("precision_options") or []
    return _precision_options(items) or precision_options(data)


def _precision_options(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    options = {}
    for item in items:
        option = {
            "label": item.get("label", item["id"]),
            "bytes_per_element": float(item.get("bytes_per_element", item.get("bytesPerElement", 0))),
            "bytes_per_entry": float(item.get("bytes_per_entry", 0)),
            "model_only": item.get("model_only", False),
            "formulas": item.get("formulas", []),
        }
        if "group_size" in item:
            option["group_size"] = float(item["group_size"])
        if "scale_bytes_per_group" in item:
            option["scale_bytes_per_group"] = float(item["scale_bytes_per_group"])
        options[item["id"]] = option
    return options


def _model_precision_options(
    options: dict[str, dict[str, Any]],
    model: dict[str, Any],
    allowed_field: str,
    fixed_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    fields = model.get("fields") or {}
    allowed = fields.get(allowed_field)
    if isinstance(allowed, list):
        options = {key: value for key, value in options.items() if key in allowed}
    else:
        formula = model.get("formula")
        fields = model.get("fields") or {}
        options = {
            key: value
            for key, value in options.items()
            if not (
                formula == "deepseek_v4_hybrid"
                and allowed_field == "indexer_precision_ids"
                and key in {"fp4_int4", "fp8_int8"}
            )
            and (
                not value.get("model_only")
                or formula in value.get("formulas", [])
            )
            and (
                key != "fp8_sglang_packed"
                or formula != "deepseek_v4_hybrid"
                or _safe_number(fields.get("head_dim"), 0) == 512
            )
        }
    if fixed_id:
        options = {key: value for key, value in options.items() if key == fixed_id}
    return options


def recurrent_state_precision_options(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["id"]: {
            "label": item.get("label", item["id"]),
            "bytes_per_element": float(item.get("bytes_per_element", item.get("bytesPerElement", 0))),
        }
        for item in data.get("recurrent_state_precision_options", [])
    } or dict(DEFAULT_RECURRENT_STATE_PRECISIONS)


def has_linear_attention_state(model: dict[str, Any]) -> bool:
    return model.get("formula") in LINEAR_STATE_FORMULAS


def default_recurrent_state_precision_id(model: dict[str, Any], options: dict[str, dict[str, Any]]) -> str:
    """Mirrors the web calculator: the model's default_recurrent_state_precision_id, else bf16 when the KDA field says 2 bytes, else fp32."""
    fields = model.get("fields") or {}
    explicit = fields.get("default_recurrent_state_precision_id")
    field_bytes = _safe_number(fields.get("kda_recurrent_state_bytes_per_element"), 0)
    model_default = explicit if isinstance(explicit, str) else ("bf16_fp16" if field_bytes == 2 else "fp32")
    if model_default in options:
        return model_default
    return "fp32" if "fp32" in options else next(iter(options))


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
    model_default = (model.get("fields") or {}).get("default_precision_id")
    if isinstance(model_default, str) and model_default in options:
        return model_default
    if _is_deepseek_v4(model) and "fp8_int8" in options:
        return "fp8_int8"
    if "bf16_fp16" in options:
        return "bf16_fp16"
    return next(iter(options))


def default_indexer_precision_id(model: dict[str, Any], options: dict[str, dict[str, Any]], fallback_precision: str | None) -> str:
    fixed = _fixed_indexer_precision_id(model)
    if fixed and fixed in options:
        return fixed
    model_default = (model.get("fields") or {}).get("default_indexer_precision_id")
    if isinstance(model_default, str) and model_default in options:
        return model_default
    if _is_deepseek_v4(model):
        for precision_id in ("fp4_int4_g32_s1", "fp4_int4"):
            if precision_id in options:
                return precision_id
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


def _calculate_byte_groups(
    model: dict[str, Any],
    tokens: int,
    include_draft_kv_cache: bool,
    include_linear_attention_state: bool,
    recurrent_state_bytes_per_element: float | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    formula = model.get("formula")
    fields = model.get("fields") or {}
    draft_layers = _draft_layer_count(model) if include_draft_kv_cache else 0

    if formula == "standard_gqa":
        layers = int(_field(model, "num_hidden_layers")) + draft_layers
        elements_per_token = layers * 2 * _field(model, "num_key_value_heads") * _field(model, "head_dim")
        vector_dim = _field(model, "head_dim")
        vectors = layers * 2 * _field(model, "num_key_value_heads") * tokens
        return elements_per_token, [{"role": "kv", "label": "KV cache", "elements": elements_per_token * tokens, "vector_dim": vector_dim, "vector_count": vectors}]

    if formula == "mla":
        layers = int(_field(model, "num_hidden_layers")) + draft_layers
        elements_per_token = layers * (_field(model, "kv_lora_rank") + _field(model, "qk_rope_head_dim"))
        vector_dim = _field(model, "kv_lora_rank") + _field(model, "qk_rope_head_dim")
        return elements_per_token, [{"role": "kv", "label": "KV cache", "elements": elements_per_token * tokens, "vector_dim": vector_dim, "vector_count": layers * tokens}]

    if formula == "dsa_mla":
        layers = int(_field(model, "num_hidden_layers"))
        active_layers = layers + draft_layers
        _, _, _, active_indexer_layers = _indexer_layer_plan(model, layers, draft_layers)
        kv_elements_per_token = active_layers * (_field(model, "kv_lora_rank") + _field(model, "qk_rope_head_dim"))
        indexer_elements_per_token = active_indexer_layers * _field(model, "index_head_dim")
        kv_dim = _field(model, "kv_lora_rank") + _field(model, "qk_rope_head_dim")
        indexer_dim = _field(model, "index_head_dim")
        return kv_elements_per_token + indexer_elements_per_token, [
            {"role": "kv", "label": "KV cache", "elements": kv_elements_per_token * tokens, "vector_dim": kv_dim, "vector_count": active_layers * tokens},
            {"role": "indexer", "label": "Indexer cache", "elements": indexer_elements_per_token * tokens, "vector_dim": indexer_dim, "vector_count": active_indexer_layers * tokens},
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
            conv_elements = linear_layers * (conv_kernel - 1) * (2 * key_heads * key_dim + value_heads * value_dim)   # kernel - 1 past inputs, as the web calculator counts
            recurrent_elements = linear_layers * value_heads * key_dim * value_dim
            groups.append({
                "role": "linear_state",
                "label": "Linear-attention state",
                "bytes_per_sequence": conv_elements * QWEN_LINEAR_CONV_BYTES_PER_ELEMENT + recurrent_elements * (recurrent_state_bytes_per_element or QWEN_LINEAR_RECURRENT_BYTES_PER_ELEMENT),
            })
        return elements_per_token, groups

    if formula in {"kimi_kda_mla_hybrid", "kimi_kda_dsa_mla_hybrid"}:
        full_layers = _field(model, "full_attention_layers")
        active_full_layers = full_layers + draft_layers
        kda_layers = _field(model, "kda_layers")
        kv_rank = _field(model, "kv_lora_rank")
        rope_dim = _field(model, "qk_rope_head_dim")
        kda_heads = _field(model, "kda_num_heads")
        kda_head_dim = _field(model, "kda_head_dim")
        key_heads = _optional_field(model, "kda_num_key_heads", kda_heads)
        key_dim = _optional_field(model, "kda_key_head_dim", kda_head_dim)
        value_heads = _optional_field(model, "kda_num_value_heads", kda_heads)
        value_dim = _optional_field(model, "kda_value_head_dim", kda_head_dim)
        conv_kernel = _field(model, "kda_conv_kernel_size")
        conv_bytes_per_element = _optional_field(
            model,
            "kda_conv_state_bytes_per_element",
            KIMI_KDA_CONV_BYTES_PER_ELEMENT,
        )
        recurrent_bytes_per_element = recurrent_state_bytes_per_element or _optional_field(
            model,
            "kda_recurrent_state_bytes_per_element",
            KIMI_KDA_RECURRENT_BYTES_PER_ELEMENT,
        )
        elements_per_token = active_full_layers * (kv_rank + rope_dim)
        kv_dim = kv_rank + rope_dim
        groups = [
            {
                "role": "kv",
                "label": "MLA latent KV cache",
                "elements": elements_per_token * tokens,
                "vector_dim": kv_dim,
                "vector_count": active_full_layers * tokens,
            }
        ]
        if formula == "kimi_kda_dsa_mla_hybrid":
            layers = int(_field(model, "num_hidden_layers"))
            _, _, _, active_indexer_layers = _indexer_layer_plan(model, layers, draft_layers)
            index_dim = _field(model, "index_head_dim")
            index_kpool = _positive_int(fields.get("index_kpool", 1), 1)
            indexer_vectors = (tokens // index_kpool) * active_indexer_layers
            groups.append({
                "role": "indexer",
                "label": "Indexer cache",
                "elements": indexer_vectors * index_dim,
                "vector_dim": index_dim,
                "vector_count": indexer_vectors,
            })
            if tokens % index_kpool:
                index_tail_bytes_per_element = _optional_field(model, "index_tail_bytes_per_element", 2)
                index_tail_elements = active_indexer_layers * index_kpool * 2 * index_dim
                groups.append({
                    "role": "index_tail",
                    "label": "Index tail cache",
                    "elements": index_tail_elements,
                    "bytes_per_sequence": index_tail_elements * index_tail_bytes_per_element,
                })
        if include_linear_attention_state:
            conv_elements = (
                kda_layers
                * (conv_kernel - 1)
                * (
                    kda_heads * kda_head_dim
                    + key_heads * key_dim
                    + value_heads * value_dim
                )
            )
            recurrent_elements = (
                kda_layers
                * value_heads
                * value_dim
                * key_dim
            )
            groups.append(
                {
                    "role": "linear_state",
                    "label": "KDA active state",
                    "bytes_per_sequence": (
                        conv_elements * conv_bytes_per_element
                        + recurrent_elements * recurrent_bytes_per_element
                    ),
                }
            )
        return elements_per_token + sum(
            group["elements"] / tokens
            for group in groups
            if group["role"] == "indexer"
        ), groups

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
        kv_heads = _field(model, "num_key_value_heads")
        kv_dim = _field(model, "head_dim")
        indexer_dim = _field(model, "index_head_dim")
        kv_elements_per_token = layers * 2 * kv_heads * kv_dim
        indexer_elements_per_token = sparse_layers * indexer_dim
        return kv_elements_per_token + indexer_elements_per_token, [
            {"role": "kv", "label": "KV cache", "elements": kv_elements_per_token * tokens, "vector_dim": int(kv_dim), "vector_count": int(layers * 2 * kv_heads * tokens)},
            {"role": "indexer", "label": "Indexer cache", "elements": indexer_elements_per_token * tokens, "vector_dim": int(indexer_dim), "vector_count": int(sparse_layers * tokens)},
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
        window_entries = 0
        compressed_entries = 0
        indexer_entries = 0
        for ratio in active_ratios:
            window_elements += sliding_window * head_dim
            window_entries += int(sliding_window)
            if ratio > 0:
                entries = math.floor(tokens / ratio)
                compressed_entries += entries
                compressed_elements += entries * head_dim
            if ratio == 4:
                entries = math.floor(tokens / 4)
                indexer_entries += entries
                indexer_elements += entries * index_dim
        attention_elements = window_elements + compressed_elements
        return (attention_elements + indexer_elements) / tokens, [
            {"role": "kv", "label": "KV cache", "elements": attention_elements, "vector_dim": int(head_dim), "vector_count": window_entries + compressed_entries},
            {"role": "indexer", "label": "Indexer cache", "elements": indexer_elements, "vector_dim": int(index_dim), "vector_count": indexer_entries},
        ]

    raise ValueError(f"Unsupported formula: {formula}")


def _bytes_for_group(
    group: dict[str, Any],
    model: dict[str, Any],
    kv_precision_id: str,
    kv_precision: dict[str, Any],
    indexer_precision_id: str | None,
    indexer_precision: dict[str, Any] | None,
) -> float:
    if "bytes_per_sequence" in group:
        return float(group["bytes_per_sequence"])
    role = group["role"]
    if role == "indexer" and indexer_precision is not None:
        profile = indexer_precision
        precision_id = indexer_precision_id
    else:
        profile = kv_precision
        precision_id = kv_precision_id

    if "vector_dim" in group:
        vector_dim = float(group["vector_dim"])
        vector_count = float(group["vector_count"])
        if role == "kv" and precision_id == "fp8_sglang_packed":
            return vector_count * _packed_entry_bytes(model, precision_id, profile)

        bytes_per_element = float(profile["bytes_per_element"])
        group_size = float(profile.get("group_size", 0))
        scale_bytes_per_group = float(profile.get("scale_bytes_per_group", 0))
        round_scale_groups_up = group_size > 0
        if (
            role == "indexer"
            and model.get("formula") == "kimi_kda_dsa_mla_hybrid"
            and precision_id in {"fp8_int8", "fp4_int4"}
        ):
            scale_prefix = "indexer_fp8" if precision_id == "fp8_int8" else "indexer_mxfp4"
            if group_size <= 0:
                group_size = _optional_field(model, f"{scale_prefix}_quant_block_size", 0)
                round_scale_groups_up = False
            if scale_bytes_per_group <= 0:
                scale_bytes_per_group = _optional_field(
                    model,
                    f"{scale_prefix}_scale_bytes_per_element",
                    0,
                )
        if group_size > 0 and scale_bytes_per_group > 0:
            scale_groups = (
                math.ceil(vector_dim / group_size)
                if round_scale_groups_up
                else math.floor(vector_dim / group_size)
            )
            return vector_count * (
                vector_dim * bytes_per_element
                + scale_groups * scale_bytes_per_group
            )
    return float(group["elements"]) * float(profile["bytes_per_element"])


def _packed_entry_bytes(model: dict[str, Any], precision_id: str, profile: dict[str, Any]) -> float:
    fields = model.get("fields") or {}
    entry_bytes = fields.get(f"kv_entry_bytes_{precision_id}")
    if entry_bytes is not None:
        return float(entry_bytes)
    if model.get("formula") == "kimi_kda_dsa_mla_hybrid":
        return 656.0
    profile_entry_bytes = float(profile.get("bytes_per_entry", 0))
    if profile_entry_bytes > 0:
        return profile_entry_bytes
    if _is_deepseek_v4(model) and _safe_number(fields.get("head_dim"), 0) == 512:
        return 584.0
    raise ValueError(f"Model {model.get('id', '')} is missing packed KV entry size")


def calculate_cache_size(
    model: dict[str, Any],
    *,
    tokens: int,
    precision: str | None = None,
    indexer_precision: str | None = None,
    block_size: int | None = None,
    include_draft_kv_cache: bool = False,
    include_linear_attention_state: bool = False,
    recurrent_state_precision: str | None = None,
    models_data: dict[str, Any] | None = None,
) -> CacheSizeResult:
    data = models_data or load_models_data()
    all_precision_by_id = precision_options(data)
    all_indexer_precision_by_id = indexer_precision_options(data)
    fixed_kv_precision = (model.get("fields") or {}).get("kv_fixed_precision_id")
    precision_by_id = _model_precision_options(all_precision_by_id, model, "kv_precision_ids", fixed_kv_precision)
    indexer_precision_by_id = _model_precision_options(all_indexer_precision_by_id, model, "indexer_precision_ids", _fixed_indexer_precision_id(model))
    precision_id = fixed_kv_precision or precision or default_precision_id(model, precision_by_id)
    if precision_id not in precision_by_id:
        if precision_id in all_precision_by_id:
            raise ValueError(f"KV precision {precision_id} is not supported for model {model.get('id', '')}")
        raise ValueError(f"Unknown KV precision: {precision_id}")
    precision_profile = precision_by_id[precision_id]
    kv_precision_bytes = float(precision_profile["bytes_per_element"])

    indexer_precision_id: str | None = None
    indexer_precision_label: str | None = None
    if _has_indexer_cache(model):
        fixed_indexer_precision = _fixed_indexer_precision_id(model)
        indexer_precision_id = fixed_indexer_precision or indexer_precision or default_indexer_precision_id(model, indexer_precision_by_id, precision_id)
        if fixed_indexer_precision is None and indexer_precision is not None and _is_deepseek_v4(model):
            indexer_precision_id = {
                "fp8_int8": "fp8_int8_g128_s4",
                "fp4_int4": "fp4_int4_g32_s1",
            }.get(indexer_precision_id, indexer_precision_id)
        elif (
            fixed_indexer_precision is None
            and indexer_precision is not None
            and model.get("formula") == "kimi_kda_dsa_mla_hybrid"
        ):
            allowed_indexer_precisions = (model.get("fields") or {}).get("indexer_precision_ids")
            if isinstance(allowed_indexer_precisions, list):
                kimi_aliases = {
                    "fp8_int8": "fp8_int8_g128_s4",
                    "fp4_int4": "fp4_int4_g32_s1",
                }
                mapped_precision = kimi_aliases.get(indexer_precision_id)
                if mapped_precision in allowed_indexer_precisions:
                    indexer_precision_id = mapped_precision
        if indexer_precision_id not in indexer_precision_by_id:
            if indexer_precision_id in all_indexer_precision_by_id:
                raise ValueError(f"Indexer precision {indexer_precision_id} is not supported for model {model.get('id', '')}")
            raise ValueError(f"Unknown indexer precision: {indexer_precision_id}")
        indexer_profile = indexer_precision_by_id[indexer_precision_id]
        indexer_precision_label = str(indexer_profile["label"])

    recurrent_precision_id: str | None = None
    recurrent_precision_label: str | None = None
    recurrent_bytes: float | None = None
    if has_linear_attention_state(model):
        recurrent_options = recurrent_state_precision_options(data)
        recurrent_precision_id = recurrent_state_precision or default_recurrent_state_precision_id(model, recurrent_options)
        if recurrent_precision_id not in recurrent_options:
            raise ValueError(f"Unknown recurrent state precision: {recurrent_precision_id}")
        recurrent_precision_label = str(recurrent_options[recurrent_precision_id]["label"])
        recurrent_bytes = float(recurrent_options[recurrent_precision_id]["bytes_per_element"])

    active_draft = include_draft_kv_cache and has_draft_kv_cache(model)
    tokens = _positive_int(tokens, int(model.get("default_tokens") or 4096))
    _, groups = _calculate_byte_groups(
        model,
        tokens,
        active_draft,
        include_linear_attention_state,
        recurrent_bytes,
    )
    kv_bytes = 0.0
    indexer_bytes = 0.0
    total_bytes = 0.0
    for group in groups:
        group_bytes = _bytes_for_group(
            group,
            model,
            precision_id,
            precision_profile,
            indexer_precision_id,
            indexer_precision_by_id.get(indexer_precision_id) if indexer_precision_id else None,
        )
        total_bytes += group_bytes
        if group["role"] == "indexer":
            indexer_bytes += group_bytes
        else:
            kv_bytes += group_bytes

    bytes_per_token = total_bytes / tokens
    hit_rate_bytes_per_token = None
    if model.get("formula") in {"kimi_kda_mla_hybrid", "kimi_kda_dsa_mla_hybrid"}:
        mla_layers = _field(model, "full_attention_layers") + (
            _draft_layer_count(model) if include_draft_kv_cache and has_draft_kv_cache(model) else 0
        )
        mla_dim = _field(model, "kv_lora_rank") + _field(model, "qk_rope_head_dim")
        if precision_id == "fp8_sglang_packed":
            hit_rate_bytes_per_token = mla_layers * _packed_entry_bytes(model, precision_id, precision_profile)
        else:
            group_size = float(precision_profile.get("group_size", 0))
            scale_bytes_per_group = float(precision_profile.get("scale_bytes_per_group", 0))
            hit_rate_bytes_per_token = mla_dim * kv_precision_bytes
            if group_size > 0 and scale_bytes_per_group > 0:
                hit_rate_bytes_per_token += math.ceil(mla_dim / group_size) * scale_bytes_per_group
            hit_rate_bytes_per_token *= mla_layers
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
        hit_rate_bytes_per_token=hit_rate_bytes_per_token,
        recurrent_state_precision=recurrent_precision_id,
        recurrent_state_precision_label=recurrent_precision_label,
    )
