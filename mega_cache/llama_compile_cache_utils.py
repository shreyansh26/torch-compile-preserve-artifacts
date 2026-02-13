from __future__ import annotations

import dataclasses
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
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
        raise RuntimeError("CUDA is required for this workflow, but no CUDA device is visible.")
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


def make_synthetic_inputs(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def to_jsonable_cache_info(cache_info: Any) -> Any:
    if cache_info is None:
        return None
    if dataclasses.is_dataclass(cache_info):
        return dataclasses.asdict(cache_info)
    if isinstance(cache_info, dict):
        return {k: to_jsonable_cache_info(v) for k, v in cache_info.items()}
    if isinstance(cache_info, (list, tuple)):
        return [to_jsonable_cache_info(v) for v in cache_info]
    return cache_info


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
