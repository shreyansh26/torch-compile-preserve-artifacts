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
