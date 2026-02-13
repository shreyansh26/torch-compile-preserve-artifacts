from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_positive_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError(f"All values must be positive integers, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one integer must be provided")
    return values


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no CUDA device is visible.")
    return torch.device("cuda")


def resolve_dtype(dtype_name: str, device: torch.device) -> tuple[torch.dtype, str]:
    normalized = dtype_name.strip().lower()
    if normalized == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16, "bfloat16"
            return torch.float16, "float16"
        return torch.float32, "float32"

    mapping: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        choices = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Choose one of: auto, {choices}")
    resolved = mapping[normalized]
    canonical = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[resolved]
    return resolved, canonical


def load_model_and_tokenizer(model_id: str, device: torch.device, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return model, tokenizer


class CausalLMLogitsWrapper(nn.Module):
    """
    Export-friendly wrapper that returns only logits tensor.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
        return outputs[0]


def resolve_num_hidden_layers(model: nn.Module) -> int:
    value = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            "Unable to resolve model.config.num_hidden_layers for KV cache wrappers."
        )
    return value


def flatten_past_key_values(past_key_values: Any) -> tuple[torch.Tensor, ...]:
    if past_key_values is None:
        return ()
    to_legacy = getattr(past_key_values, "to_legacy_cache", None)
    if callable(to_legacy):
        past_key_values = to_legacy()

    flat: list[torch.Tensor] = []
    for layer in past_key_values:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise ValueError("Invalid past_key_values layer format.")
        key, value = layer[0], layer[1]
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError("Invalid past_key_values tensor entry.")
        flat.extend((key, value))
    return tuple(flat)


def unflatten_past_key_values(
    past_flat: tuple[torch.Tensor, ...], *, num_hidden_layers: int
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    expected = num_hidden_layers * 2
    if len(past_flat) != expected:
        raise ValueError(
            f"Expected {expected} KV tensors ({num_hidden_layers} layers), got {len(past_flat)}."
        )
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(num_hidden_layers):
        key = past_flat[2 * layer_idx]
        value = past_flat[2 * layer_idx + 1]
        layers.append((key, value))
    return tuple(layers)


def legacy_to_dynamic_cache(
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
):
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return past_key_values
    return DynamicCache.from_legacy_cache(past_key_values)


def trim_flattened_kv_cache(
    past_flat: tuple[torch.Tensor, ...], *, target_seq_len: int
) -> tuple[torch.Tensor, ...]:
    if target_seq_len <= 0:
        raise ValueError("target_seq_len must be positive.")
    trimmed: list[torch.Tensor] = []
    for tensor in past_flat:
        if tensor.dim() < 2:
            raise ValueError("KV cache tensor rank must be >= 2.")
        seq_dim = tensor.dim() - 2
        current_len = int(tensor.shape[seq_dim])
        if current_len < target_seq_len:
            raise ValueError(
                f"Cannot trim KV cache from length {current_len} to {target_seq_len}."
            )
        if current_len == target_seq_len:
            trimmed.append(tensor)
            continue
        slices = [slice(None)] * tensor.dim()
        slices[seq_dim] = slice(0, target_seq_len)
        trimmed.append(tensor[tuple(slices)].contiguous())
    return tuple(trimmed)


def clone_flattened_kv_cache(
    past_flat: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.clone() for tensor in past_flat)


def active_positions_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must have shape [batch, seq].")
    if attention_mask.shape[0] != 1:
        raise ValueError("Only batch_size=1 is supported in this runtime helper.")
    positions = torch.nonzero(attention_mask[0] > 0, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("attention_mask has no active tokens.")
    return positions


def compact_flattened_kv_cache(
    past_flat: tuple[torch.Tensor, ...], *, active_positions: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    compact: list[torch.Tensor] = []
    for tensor in past_flat:
        if tensor.dim() < 3:
            raise ValueError("KV cache tensor rank must be >= 3.")
        seq_dim = tensor.dim() - 2
        indices = active_positions.to(device=tensor.device, dtype=torch.long)
        compact.append(torch.index_select(tensor, seq_dim, indices).contiguous())
    return tuple(compact)


class CausalLMPrefillWrapper(nn.Module):
    """
    Export/compile-friendly prefill wrapper.
    Returns (logits, *flattened_past_key_values).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=False,
        )
        logits = outputs[0]
        flat_cache = flatten_past_key_values(outputs[1])
        return (logits, *flat_cache)


class CausalLMDecodeWrapper(nn.Module):
    """
    Export/compile-friendly decode wrapper.
    Inputs: (input_ids[batch, 1], attention_mask[batch, past+1], *flattened_past)
    Returns: (logits, *flattened_updated_past)
    """

    def __init__(self, model: nn.Module, *, num_hidden_layers: int | None = None) -> None:
        super().__init__()
        self.model = model
        self.num_hidden_layers = (
            num_hidden_layers
            if num_hidden_layers is not None
            else resolve_num_hidden_layers(model)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        legacy_cache = unflatten_past_key_values(
            tuple(past_key_values_flat),
            num_hidden_layers=self.num_hidden_layers,
        )
        dynamic_cache = legacy_to_dynamic_cache(legacy_cache)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=dynamic_cache,
            use_cache=True,
            return_dict=False,
        )
        logits = outputs[0]
        flat_cache = flatten_past_key_values(outputs[1])
        return (logits, *flat_cache)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def runtime_fingerprint(device: torch.device) -> dict[str, Any]:
    gpu_info: dict[str, Any] = {}
    if device.type == "cuda":
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu_info = {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
            "cuda_runtime_version": torch.version.cuda,
        }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device_type": device.type,
        "gpu": gpu_info,
    }


def resolve_dynamic_seq_dim(
    *,
    min_seq_len: int,
    max_seq_len: int,
    seq_multiple: int,
):
    if min_seq_len <= 0:
        raise ValueError("min_seq_len must be positive")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if max_seq_len < min_seq_len:
        raise ValueError("max_seq_len must be >= min_seq_len")
    if seq_multiple <= 0:
        raise ValueError("seq_multiple must be positive")

    if seq_multiple == 1:
        seq_dim = torch.export.Dim("seq_len", min=min_seq_len, max=max_seq_len)
        return seq_dim, min_seq_len, max_seq_len

    unit_min = (min_seq_len + seq_multiple - 1) // seq_multiple
    unit_max = max_seq_len // seq_multiple
    if unit_max < unit_min:
        raise ValueError(
            "No valid dynamic sequence values after applying sequence multiple. "
            f"min_seq_len={min_seq_len}, max_seq_len={max_seq_len}, seq_multiple={seq_multiple}"
        )
    unit_dim = torch.export.Dim("seq_len_unit", min=unit_min, max=unit_max)
    seq_dim = seq_multiple * unit_dim
    return seq_dim, unit_min * seq_multiple, unit_max * seq_multiple


def apply_transformers_masking_workaround() -> bool:
    """
    Work around transformers 4.57.x + torch.export dynamic tracing failures in
    masking_utils by forcing the older SDPA mask path.
    """

    try:
        import transformers.masking_utils as masking_utils
    except Exception as exc:
        print(f"[build] warning: could not import transformers.masking_utils: {exc}")
        return False

    older_sdpa = getattr(masking_utils, "sdpa_mask_older_torch", None)
    all_masks = getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None)
    eager_mask = getattr(masking_utils, "eager_mask", None)
    if older_sdpa is None or all_masks is None or eager_mask is None:
        print("[build] warning: masking workaround unavailable in this transformers version.")
        return False

    try:
        masking_utils.sdpa_mask = older_sdpa
        all_masks.register("sdpa", older_sdpa)
        all_masks.register("eager", eager_mask)
        return True
    except Exception as exc:
        print(f"[build] warning: failed to apply masking workaround: {exc}")
        return False


def pad_inputs_to_multiple(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    seq_multiple: int,
    pad_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if seq_multiple <= 1:
        return input_ids, attention_mask, int(input_ids.shape[1])

    seq_len = int(input_ids.shape[1])
    padded_seq_len = ((seq_len + seq_multiple - 1) // seq_multiple) * seq_multiple
    if padded_seq_len == seq_len:
        return input_ids, attention_mask, seq_len

    pad_len = padded_seq_len - seq_len
    pad_value = int(pad_token_id) if pad_token_id is not None else 0
    ids_pad = torch.full(
        (input_ids.shape[0], pad_len),
        pad_value,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    mask_pad = torch.zeros(
        (attention_mask.shape[0], pad_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    padded_input_ids = torch.cat([input_ids, ids_pad], dim=1)
    padded_attention_mask = torch.cat([attention_mask, mask_pad], dim=1)
    return padded_input_ids, padded_attention_mask, seq_len
