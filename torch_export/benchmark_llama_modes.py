from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from aot_export_utils import (
    CausalLMDecodeWrapper,
    CausalLMPrefillWrapper,
    active_positions_from_attention_mask,
    clone_flattened_kv_cache,
    compact_flattened_kv_cache,
    load_model_and_tokenizer,
    pad_inputs_to_multiple,
    parse_positive_int_csv,
    require_cuda,
    resolve_dynamic_seq_dim,
    resolve_dtype,
)

BENCH_PREFIX = "BENCH_JSON:"
MODES = ("eager", "compile_preload", "compile_no_preload")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Llama inference across compile+preload (.pt2 load), "
            "compile+no-preload (runtime torch.compile), and eager mode."
        )
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--package-path",
        default="artifacts/llama3b_aotinductor.pt2",
        help="Path to prebuilt AOTInductor prefill package for compile_preload mode.",
    )
    parser.add_argument(
        "--decode-package-path",
        default="",
        help=(
            "Path to prebuilt AOTInductor decode package for compile_preload mode. "
            "If empty, read from metadata or derive from --package-path."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        default="artifacts/llama3b_aotinductor_meta.json",
        help="Path to metadata JSON produced by build_llama_aotinductor.py.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a short explanation of portable torch.compile caches.",
        help="Prompt for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1,
        help="Decode steps per request.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=3,
        help="Sequential requests per mode, in a single process.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Independent subprocess repeats per mode.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Model dtype on GPU.",
    )
    parser.add_argument(
        "--cache-implementation",
        default="static",
        choices=["static", "hybrid", "sliding_window"],
        help="KV cache implementation used for eager generate().",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature in eager mode; 0.0 means greedy.",
    )
    parser.add_argument(
        "--bucket-pad",
        action="store_true",
        default=False,
        help="Pad prompt to nearest bucket.",
    )
    parser.add_argument(
        "--no-bucket-pad",
        dest="bucket_pad",
        action="store_false",
        help="Disable prompt bucket padding.",
    )
    parser.add_argument(
        "--prefill-buckets",
        default="",
        help="Comma-separated prefill buckets. If empty, infer from metadata example.seq_len.",
    )
    parser.add_argument(
        "--stop-on-eos",
        action="store_true",
        default=True,
        help="Stop autoregressive loop early on EOS token.",
    )
    parser.add_argument(
        "--no-stop-on-eos",
        dest="stop_on_eos",
        action="store_false",
        help="Disable EOS early stop.",
    )
    parser.add_argument(
        "--compile-dynamic-seq-len",
        action="store_true",
        default=False,
        help="For compile_no_preload mode, set torch.compile(dynamic=True).",
    )
    parser.add_argument(
        "--example-seq-len",
        type=int,
        default=128,
        help="Fallback prefill bucket length when metadata does not exist.",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=1,
        help="Reference min sequence length used for compile_no_preload reporting.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Reference max total sequence length used for compile_no_preload reporting.",
    )
    parser.add_argument(
        "--dynamic-seq-multiple",
        type=int,
        default=8,
        help="Pad prefill sequence length to this multiple for compile_no_preload mode.",
    )
    parser.add_argument(
        "--max-autotune",
        action="store_true",
        default=True,
        help="Use torch.compile(mode='max-autotune') in compile_no_preload mode.",
    )
    parser.add_argument(
        "--no-max-autotune",
        dest="max_autotune",
        action="store_false",
        help="Use torch.compile(mode='default') in compile_no_preload mode.",
    )
    parser.add_argument(
        "--isolate-compiler-caches",
        action="store_true",
        default=True,
        help="Use fresh TORCHINDUCTOR/TRITON cache dirs per subprocess run.",
    )
    parser.add_argument(
        "--no-isolate-compiler-caches",
        dest="isolate_compiler_caches",
        action="store_false",
        help="Reuse current compiler cache dirs.",
    )
    parser.add_argument(
        "--single-mode",
        choices=MODES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be >= 0.0")
    if args.example_seq_len <= 0:
        raise ValueError("--example-seq-len must be positive")
    if args.min_seq_len <= 0:
        raise ValueError("--min-seq-len must be positive")
    if args.max_seq_len < args.min_seq_len:
        raise ValueError("--max-seq-len must be >= --min-seq-len")
    if args.dynamic_seq_multiple <= 0:
        raise ValueError("--dynamic-seq-multiple must be positive")
    return args


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    idx = max(0, min(len(data) - 1, int(round((len(data) - 1) * p))))
    return float(data[idx])


def safe_tps(tokens: int, seconds: float) -> float:
    return float(tokens / max(1e-9, seconds))


def derive_decode_package_path(prefill_path: Path) -> Path:
    suffix = prefill_path.suffix or ".pt2"
    return prefill_path.with_name(f"{prefill_path.stem}_decode{suffix}")


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_package_paths(args: argparse.Namespace, metadata: dict[str, Any]) -> tuple[Path, Path]:
    prefill_path = Path(args.package_path)
    decode_path = None
    if args.decode_package_path.strip():
        decode_path = Path(args.decode_package_path)
    else:
        metadata_decode = metadata.get("decode_package_path")
        if isinstance(metadata_decode, str) and metadata_decode.strip():
            decode_path = Path(metadata_decode)
    if decode_path is None:
        decode_path = derive_decode_package_path(prefill_path)
    return prefill_path, decode_path


def resolve_prefill_buckets(args: argparse.Namespace, metadata: dict[str, Any]) -> list[int]:
    if args.prefill_buckets.strip():
        return parse_positive_int_csv(args.prefill_buckets)
    example_seq_len = metadata.get("example", {}).get("seq_len")
    if isinstance(example_seq_len, int) and example_seq_len > 0:
        return [example_seq_len]
    if args.example_seq_len > 0:
        return [args.example_seq_len]
    return []


def maybe_bucket_pad_inputs(
    tokenizer,
    prompt: str,
    prefill_buckets: list[int],
    enable_bucket_pad: bool,
):
    encoded = tokenizer(prompt, return_tensors="pt")
    original_prompt_tokens = int(encoded["input_ids"].shape[1])
    if not enable_bucket_pad:
        return encoded, original_prompt_tokens, original_prompt_tokens

    if not prefill_buckets:
        print("[bench] bucket pad requested but no prefill buckets found; using raw prompt.")
        return encoded, original_prompt_tokens, original_prompt_tokens

    target = next((b for b in sorted(prefill_buckets) if b >= original_prompt_tokens), None)
    if target is None or target == original_prompt_tokens:
        return encoded, original_prompt_tokens, original_prompt_tokens

    tokenizer.padding_side = "left"
    padded = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=target,
        truncation=False,
    )
    print(
        "[bench] bucket padding enabled: "
        f"prompt_tokens={original_prompt_tokens} -> padded_tokens={target}"
    )
    return padded, original_prompt_tokens, target


def split_prefill_outputs(outputs: Any) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    logits = outputs[0]
    cache = tuple(outputs[1:])
    return logits, cache


def ensure_total_seq_limit(
    *,
    current_total_seq_len: int,
    max_total_seq_len: int | None,
) -> None:
    if max_total_seq_len is None:
        return
    if current_total_seq_len > max_total_seq_len:
        raise ValueError(
            f"Total sequence length {current_total_seq_len} exceeded max {max_total_seq_len}. "
            "Reduce prompt length or max_new_tokens, or rebuild with a larger max sequence length."
        )


def run_kv_request(
    *,
    prefill_runner,
    decode_runner,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    stop_on_eos: bool,
    eos_token_id: int | None,
    prefill_seq_multiple: int,
    pad_token_id: int | None,
    max_total_seq_len: int | None,
) -> tuple[list[int], float]:
    generated_ids: list[int] = []
    prompt_input_ids = input_ids.clone()
    prompt_attention_mask = attention_mask.clone()
    prompt_seq_len = int(prompt_input_ids.shape[1])

    start = time.perf_counter()
    with torch.inference_mode():
        ensure_total_seq_limit(
            current_total_seq_len=prompt_seq_len,
            max_total_seq_len=max_total_seq_len,
        )
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        prefill_input_ids, prefill_attention_mask, _ = pad_inputs_to_multiple(
            prompt_input_ids,
            prompt_attention_mask,
            seq_multiple=prefill_seq_multiple,
            pad_token_id=pad_token_id,
        )
        prefill_outputs = prefill_runner(prefill_input_ids, prefill_attention_mask)
        logits, past_flat = split_prefill_outputs(prefill_outputs)
        active_positions = active_positions_from_attention_mask(prefill_attention_mask)
        last_active_pos = int(active_positions[-1].item())
        past_flat = compact_flattened_kv_cache(
            past_flat,
            active_positions=active_positions,
        )
        past_flat = clone_flattened_kv_cache(past_flat)

        next_token = logits[:, last_active_pos, :].argmax(dim=-1, keepdim=True)
        next_id = int(next_token.item())
        generated_ids.append(next_id)
        if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
            torch.cuda.synchronize()
            return generated_ids, time.perf_counter() - start

        active_seq_len = int(past_flat[0].shape[past_flat[0].dim() - 2])
        decode_attention_mask = torch.ones(
            (prompt_attention_mask.shape[0], active_seq_len + 1),
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )
        decode_input_ids = next_token

        for _ in range(1, max_new_tokens):
            ensure_total_seq_limit(
                current_total_seq_len=int(decode_attention_mask.shape[1]),
                max_total_seq_len=max_total_seq_len,
            )
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            decode_outputs = decode_runner(decode_input_ids, decode_attention_mask, *past_flat)
            logits = decode_outputs[0]
            past_flat = clone_flattened_kv_cache(tuple(decode_outputs[1:]))

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_id = int(next_token.item())
            generated_ids.append(next_id)
            decode_input_ids = next_token
            decode_attention_mask = torch.cat(
                [
                    decode_attention_mask,
                    torch.ones(
                        (decode_attention_mask.shape[0], 1),
                        dtype=decode_attention_mask.dtype,
                        device=decode_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return generated_ids, elapsed


def run_prefill_stage(
    *,
    prefill_runner,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    stop_on_eos: bool,
    eos_token_id: int | None,
    prefill_seq_multiple: int,
    pad_token_id: int | None,
    max_total_seq_len: int | None,
) -> tuple[dict[str, Any], float]:
    prompt_input_ids = input_ids.clone()
    prompt_attention_mask = attention_mask.clone()
    prompt_seq_len = int(prompt_input_ids.shape[1])

    start = time.perf_counter()
    with torch.inference_mode():
        ensure_total_seq_limit(
            current_total_seq_len=prompt_seq_len,
            max_total_seq_len=max_total_seq_len,
        )
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        prefill_input_ids, prefill_attention_mask, _ = pad_inputs_to_multiple(
            prompt_input_ids,
            prompt_attention_mask,
            seq_multiple=prefill_seq_multiple,
            pad_token_id=pad_token_id,
        )
        prefill_outputs = prefill_runner(prefill_input_ids, prefill_attention_mask)
        logits, past_flat = split_prefill_outputs(prefill_outputs)
        active_positions = active_positions_from_attention_mask(prefill_attention_mask)
        last_active_pos = int(active_positions[-1].item())
        past_flat = compact_flattened_kv_cache(
            past_flat,
            active_positions=active_positions,
        )
        past_flat = clone_flattened_kv_cache(past_flat)
        next_token = logits[:, last_active_pos, :].argmax(dim=-1, keepdim=True)
        next_id = int(next_token.item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    finished = bool(stop_on_eos and eos_token_id is not None and next_id == eos_token_id)
    active_seq_len = int(past_flat[0].shape[past_flat[0].dim() - 2])
    decode_attention_mask = torch.ones(
        (prompt_attention_mask.shape[0], active_seq_len + 1),
        dtype=prompt_attention_mask.dtype,
        device=prompt_attention_mask.device,
    )
    return (
        {
            "generated_ids": [next_id],
            "finished": finished,
            "past_flat": past_flat,
            "decode_input_ids": next_token,
            "decode_attention_mask": decode_attention_mask,
        },
        elapsed,
    )


def run_decode_stage(
    *,
    decode_runner,
    prefill_state: dict[str, Any],
    max_new_tokens: int,
    stop_on_eos: bool,
    eos_token_id: int | None,
    max_total_seq_len: int | None,
) -> tuple[list[int], float]:
    generated_ids: list[int] = []
    decode_input_ids = prefill_state["decode_input_ids"]
    decode_attention_mask = prefill_state["decode_attention_mask"]
    past_flat = clone_flattened_kv_cache(prefill_state["past_flat"])

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(1, max_new_tokens):
            ensure_total_seq_limit(
                current_total_seq_len=int(decode_attention_mask.shape[1]),
                max_total_seq_len=max_total_seq_len,
            )
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            decode_outputs = decode_runner(decode_input_ids, decode_attention_mask, *past_flat)
            logits = decode_outputs[0]
            past_flat = clone_flattened_kv_cache(tuple(decode_outputs[1:]))
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_id = int(next_token.item())
            generated_ids.append(next_id)
            decode_input_ids = next_token
            decode_attention_mask = torch.cat(
                [
                    decode_attention_mask,
                    torch.ones(
                        (decode_attention_mask.shape[0], 1),
                        dtype=decode_attention_mask.dtype,
                        device=decode_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return generated_ids, elapsed


def ensure_aot_runtime_compatible(
    *,
    mode: str,
    prompt_seq_len: int,
    dynamic_seq_len: bool,
    expected_seq_len: int | None,
    max_new_tokens: int,
) -> None:
    if dynamic_seq_len:
        return
    if expected_seq_len is not None and prompt_seq_len != expected_seq_len:
        raise ValueError(
            f"{mode} is static-sequence and expects seq_len={expected_seq_len}, "
            f"but runtime prompt uses seq_len={prompt_seq_len}. "
            "Enable --bucket-pad with a matching bucket, or use dynamic export."
        )
    if max_new_tokens > 1:
        raise ValueError(
            f"{mode} is static-sequence; --max-new-tokens must be 1, got {max_new_tokens}. "
            "Build and run with dynamic sequence support for multi-token generation."
        )


def make_single_mode_metrics(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    overall_start = time.perf_counter()
    device = require_cuda()
    dtype, dtype_name = resolve_dtype(args.dtype, device)
    metadata = load_metadata(Path(args.metadata_path))

    prefill_buckets = resolve_prefill_buckets(args, metadata)
    artifact_build_seconds = 0.0
    artifact_load_seconds = 0.0
    model_load_seconds = 0.0
    tokenizer_load_seconds = 0.0
    request_latencies: list[float] = []
    total_new_tokens = 0
    first_completion = ""
    mode_notes = ""

    if mode == "eager":
        print("[bench] mode=eager: no compile path.")
        model_load_start = time.perf_counter()
        model, tokenizer = load_model_and_tokenizer(args.model_id, device, dtype)
        model_load_seconds = time.perf_counter() - model_load_start

        encoded_cpu, original_prompt_tokens, effective_prompt_tokens = maybe_bucket_pad_inputs(
            tokenizer=tokenizer,
            prompt=args.prompt,
            prefill_buckets=prefill_buckets,
            enable_bucket_pad=args.bucket_pad,
        )
        encoded = encoded_cpu.to(device)
        prompt_len = int(encoded["input_ids"].shape[1])

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")

        do_sample = args.temperature > 0.0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": pad_token_id,
            "do_sample": do_sample,
            "cache_implementation": args.cache_implementation,
            "disable_compile": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = args.temperature

        for request_idx in range(args.num_requests):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(**encoded, **generation_kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            request_latencies.append(elapsed)
            generated = output_ids[0][prompt_len:]
            total_new_tokens += int(generated.shape[0])
            if request_idx == 0:
                first_completion = tokenizer.decode(generated, skip_special_tokens=True).strip()
            print(
                f"[bench] mode={mode} request={request_idx + 1}/{args.num_requests} "
                f"latency={elapsed:.2f}s"
            )

    else:
        dynamic_seq_len = False
        expected_seq_len: int | None = None
        prefill_seq_multiple = 1
        max_total_seq_len: int | None = None

        if mode == "compile_preload":
            tokenizer_load_start = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer_load_seconds = time.perf_counter() - tokenizer_load_start

            encoded_cpu, original_prompt_tokens, effective_prompt_tokens = maybe_bucket_pad_inputs(
                tokenizer=tokenizer,
                prompt=args.prompt,
                prefill_buckets=prefill_buckets,
                enable_bucket_pad=args.bucket_pad,
            )
            input_ids = encoded_cpu["input_ids"].to(device)
            attention_mask = encoded_cpu["attention_mask"].to(device)

            prefill_package_path, decode_package_path = resolve_package_paths(args, metadata)
            if not prefill_package_path.exists():
                raise FileNotFoundError(
                    f"Prebuilt prefill package not found: {prefill_package_path}. Run ./build.sh first."
                )
            if not decode_package_path.exists():
                raise FileNotFoundError(
                    f"Prebuilt decode package not found: {decode_package_path}. Run ./build.sh first."
                )
            dynamic_seq_len = bool(metadata.get("export", {}).get("dynamic_seq_len", False))
            maybe_dynamic_multiple = metadata.get("export", {}).get("dynamic_seq_multiple", 1)
            if isinstance(maybe_dynamic_multiple, int) and maybe_dynamic_multiple > 0:
                prefill_seq_multiple = maybe_dynamic_multiple
            maybe_expected = metadata.get("example", {}).get("seq_len")
            if isinstance(maybe_expected, int) and maybe_expected > 0:
                expected_seq_len = maybe_expected
            maybe_max_total = metadata.get("export", {}).get("effective_max_seq_len")
            if isinstance(maybe_max_total, int) and maybe_max_total > 0:
                max_total_seq_len = maybe_max_total
            ensure_aot_runtime_compatible(
                mode=mode,
                prompt_seq_len=int(input_ids.shape[1]),
                dynamic_seq_len=dynamic_seq_len,
                expected_seq_len=expected_seq_len,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"[bench] mode=compile_preload: loading prefill package {prefill_package_path}")
            load_start = time.perf_counter()
            compiled_prefill = torch._inductor.aoti_load_package(str(prefill_package_path))
            torch.cuda.synchronize()
            prefill_load_seconds = time.perf_counter() - load_start
            artifact_load_seconds = prefill_load_seconds
            decode_load_seconds = 0.0
            mode_notes = (
                "prebuilt .pt2 staged load (prefill then decode)"
                f" (dynamic_seq_len={dynamic_seq_len}, prefill_seq_multiple={prefill_seq_multiple})"
            )
        else:
            model_load_start = time.perf_counter()
            model, tokenizer = load_model_and_tokenizer(args.model_id, device, dtype)
            model_load_seconds = time.perf_counter() - model_load_start

            encoded_cpu, original_prompt_tokens, effective_prompt_tokens = maybe_bucket_pad_inputs(
                tokenizer=tokenizer,
                prompt=args.prompt,
                prefill_buckets=prefill_buckets,
                enable_bucket_pad=args.bucket_pad,
            )
            input_ids = encoded_cpu["input_ids"].to(device)
            attention_mask = encoded_cpu["attention_mask"].to(device)

            dynamic_seq_len = bool(args.compile_dynamic_seq_len)
            prefill_seq_multiple = args.dynamic_seq_multiple if dynamic_seq_len else 1
            if prefill_seq_multiple <= 1:
                prefill_seq_multiple = 1

            effective_min_seq_len = args.min_seq_len
            effective_max_seq_len = args.max_seq_len
            if dynamic_seq_len:
                _, effective_min_seq_len, effective_max_seq_len = resolve_dynamic_seq_dim(
                    min_seq_len=args.min_seq_len,
                    max_seq_len=args.max_seq_len,
                    seq_multiple=prefill_seq_multiple,
                )
            else:
                prefill_seq_multiple = 1
            max_total_seq_len = effective_max_seq_len

            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
            if pad_token_id is None:
                raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")

            compile_input_ids, compile_attention_mask, _ = pad_inputs_to_multiple(
                input_ids,
                attention_mask,
                seq_multiple=prefill_seq_multiple,
                pad_token_id=pad_token_id,
            )

            prefill_wrapper = CausalLMPrefillWrapper(model).eval()
            decode_wrapper = CausalLMDecodeWrapper(model).eval()
            compile_mode = "max-autotune" if args.max_autotune else "default"
            print(
                "[bench] mode=compile_no_preload: compiling prefill+decode wrappers with torch.compile "
                f"(dynamic_seq_len={dynamic_seq_len}, compile_mode={compile_mode}, fullgraph=True)"
            )
            build_start = time.perf_counter()
            compiled_prefill = torch.compile(
                prefill_wrapper,
                mode=compile_mode,
                fullgraph=True,
                dynamic=dynamic_seq_len,
            )
            compiled_decode = torch.compile(
                decode_wrapper,
                mode=compile_mode,
                fullgraph=True,
                dynamic=dynamic_seq_len,
            )
            with torch.inference_mode():
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                warm_prefill_outputs = compiled_prefill(compile_input_ids, compile_attention_mask)
                warm_logits, warm_past_flat = split_prefill_outputs(warm_prefill_outputs)
                warm_active_positions = active_positions_from_attention_mask(
                    compile_attention_mask
                )
                warm_last_pos = int(warm_active_positions[-1].item())
                warm_past_flat = compact_flattened_kv_cache(
                    warm_past_flat,
                    active_positions=warm_active_positions,
                )
                warm_past_flat = clone_flattened_kv_cache(warm_past_flat)
                warm_next = warm_logits[:, warm_last_pos, :].argmax(
                    dim=-1, keepdim=True
                )
                warm_active_len = int(warm_past_flat[0].shape[warm_past_flat[0].dim() - 2])
                warm_decode_mask = torch.ones(
                    (attention_mask.shape[0], warm_active_len + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                ensure_total_seq_limit(
                    current_total_seq_len=int(warm_decode_mask.shape[1]),
                    max_total_seq_len=max_total_seq_len,
                )
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                _ = compiled_decode(warm_next, warm_decode_mask, *warm_past_flat)
            torch.cuda.synchronize()
            artifact_build_seconds = time.perf_counter() - build_start
            artifact_load_seconds = 0.0
            mode_notes = (
                "runtime torch.compile prefill+decode"
                f" (dynamic_seq_len={dynamic_seq_len}, "
                f"prefill_seq_multiple={prefill_seq_multiple}, "
                f"reference_seq=[{effective_min_seq_len}, {effective_max_seq_len}], "
                f"compile_mode={compile_mode}, fullgraph=True)"
            )

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id
        if mode == "compile_preload":
            request_latencies = [0.0 for _ in range(args.num_requests)]
            request_outputs: list[list[int]] = [[] for _ in range(args.num_requests)]
            pending_decode: list[tuple[int, dict[str, Any]]] = []

            for request_idx in range(args.num_requests):
                prefill_state, prefill_elapsed = run_prefill_stage(
                    prefill_runner=compiled_prefill,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    stop_on_eos=args.stop_on_eos,
                    eos_token_id=eos_token_id,
                    prefill_seq_multiple=prefill_seq_multiple,
                    pad_token_id=pad_token_id,
                    max_total_seq_len=max_total_seq_len,
                )
                request_latencies[request_idx] = prefill_elapsed
                if args.max_new_tokens <= 1 or prefill_state["finished"]:
                    request_outputs[request_idx] = list(prefill_state["generated_ids"])
                    print(
                        f"[bench] mode={mode} request={request_idx + 1}/{args.num_requests} "
                        f"latency={prefill_elapsed:.2f}s"
                    )
                else:
                    pending_decode.append((request_idx, prefill_state))

            del compiled_prefill
            torch.cuda.empty_cache()

            if pending_decode:
                print(f"[bench] mode=compile_preload: loading decode package {decode_package_path}")
                decode_load_start = time.perf_counter()
                compiled_decode = torch._inductor.aoti_load_package(str(decode_package_path))
                torch.cuda.synchronize()
                decode_load_seconds = time.perf_counter() - decode_load_start
                artifact_load_seconds += decode_load_seconds

                for request_idx, prefill_state in pending_decode:
                    extra_ids, decode_elapsed = run_decode_stage(
                        decode_runner=compiled_decode,
                        prefill_state=prefill_state,
                        max_new_tokens=args.max_new_tokens,
                        stop_on_eos=args.stop_on_eos,
                        eos_token_id=eos_token_id,
                        max_total_seq_len=max_total_seq_len,
                    )
                    generated_ids = list(prefill_state["generated_ids"])
                    generated_ids.extend(extra_ids)
                    request_outputs[request_idx] = generated_ids
                    request_latencies[request_idx] += decode_elapsed
                    print(
                        f"[bench] mode={mode} request={request_idx + 1}/{args.num_requests} "
                        f"latency={request_latencies[request_idx]:.2f}s"
                    )
                del compiled_decode
                torch.cuda.empty_cache()

            for request_idx in range(args.num_requests):
                generated_ids = request_outputs[request_idx]
                total_new_tokens += len(generated_ids)
                if request_idx == 0:
                    first_completion = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    ).strip()
        else:
            for request_idx in range(args.num_requests):
                generated_ids, elapsed = run_kv_request(
                    prefill_runner=compiled_prefill,
                    decode_runner=compiled_decode,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    stop_on_eos=args.stop_on_eos,
                    eos_token_id=eos_token_id,
                    prefill_seq_multiple=prefill_seq_multiple,
                    pad_token_id=pad_token_id,
                    max_total_seq_len=max_total_seq_len,
                )
                request_latencies.append(elapsed)
                total_new_tokens += len(generated_ids)
                if request_idx == 0:
                    first_completion = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    ).strip()
                print(
                    f"[bench] mode={mode} request={request_idx + 1}/{args.num_requests} "
                    f"latency={elapsed:.2f}s"
                )

    first_latency = request_latencies[0]
    tail_latencies = request_latencies[1:] if len(request_latencies) > 1 else request_latencies
    total_generation_seconds = float(sum(request_latencies))
    mean_request_seconds = float(statistics.mean(request_latencies))
    total_script_seconds = time.perf_counter() - overall_start
    cold_start_first_request_seconds = float(
        tokenizer_load_seconds
        + model_load_seconds
        + artifact_build_seconds
        + artifact_load_seconds
        + first_latency
    )
    tail_tokens = max(0, total_new_tokens - 1)
    tail_generation_seconds = float(sum(tail_latencies))

    return {
        "mode": mode,
        "num_requests": args.num_requests,
        "request_latencies": request_latencies,
        "first_request_seconds": first_latency,
        "ttft_after_ready_seconds": first_latency,
        "tail_mean_seconds": float(statistics.mean(tail_latencies)),
        "mean_request_seconds": mean_request_seconds,
        "median_request_seconds": median(request_latencies),
        "p90_request_seconds": percentile(request_latencies, 0.90),
        "total_generation_seconds": total_generation_seconds,
        "total_script_seconds": total_script_seconds,
        "tokens_per_second": safe_tps(total_new_tokens, total_generation_seconds),
        "tokens_per_second_all": safe_tps(total_new_tokens, total_generation_seconds),
        "tail_tokens_per_second": safe_tps(tail_tokens, tail_generation_seconds),
        "steady_state_tokens_per_second": safe_tps(tail_tokens, tail_generation_seconds),
        "end_to_end_tokens_per_second": safe_tps(total_new_tokens, total_script_seconds),
        "ttft_cold_start_seconds": cold_start_first_request_seconds,
        "cold_start_first_request_seconds": cold_start_first_request_seconds,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_load_seconds": model_load_seconds,
        "artifact_build_seconds": artifact_build_seconds,
        "artifact_load_seconds": artifact_load_seconds,
        "prompt_tokens_original": original_prompt_tokens,
        "prompt_tokens_effective": effective_prompt_tokens,
        "total_new_tokens": total_new_tokens,
        "dtype": dtype_name,
        "bucket_pad": bool(args.bucket_pad),
        "cache_implementation": args.cache_implementation,
        "mode_notes": mode_notes,
        "first_completion": first_completion,
        "kv_cache": mode != "eager",
    }


def extract_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(BENCH_PREFIX):
            return json.loads(line[len(BENCH_PREFIX) :])
    raise RuntimeError("Subprocess did not emit benchmark JSON.")


def build_subprocess_cmd(args: argparse.Namespace, mode: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-mode",
        mode,
        "--emit-json",
        "--model-id",
        args.model_id,
        "--package-path",
        args.package_path,
        "--metadata-path",
        args.metadata_path,
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--num-requests",
        str(args.num_requests),
        "--repeats",
        "1",
        "--dtype",
        args.dtype,
        "--cache-implementation",
        args.cache_implementation,
        "--temperature",
        str(args.temperature),
        "--prefill-buckets",
        args.prefill_buckets,
        "--example-seq-len",
        str(args.example_seq_len),
        "--min-seq-len",
        str(args.min_seq_len),
        "--max-seq-len",
        str(args.max_seq_len),
        "--dynamic-seq-multiple",
        str(args.dynamic_seq_multiple),
    ]
    if args.decode_package_path.strip():
        cmd.extend(["--decode-package-path", args.decode_package_path])
    cmd.append("--bucket-pad" if args.bucket_pad else "--no-bucket-pad")
    cmd.append("--stop-on-eos" if args.stop_on_eos else "--no-stop-on-eos")
    if args.compile_dynamic_seq_len:
        cmd.append("--compile-dynamic-seq-len")
    if args.max_autotune:
        cmd.append("--max-autotune")
    else:
        cmd.append("--no-max-autotune")
    return cmd


def run_parent_benchmark(args: argparse.Namespace) -> None:
    print(
        "[bench] running benchmark modes="
        f"{MODES}, requests_per_mode={args.num_requests}, repeats={args.repeats}"
    )
    all_results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    tmp_root_dir = os.environ.get("TMPDIR")
    if tmp_root_dir:
        Path(tmp_root_dir).mkdir(parents=True, exist_ok=True)

    for mode in MODES:
        for repeat in range(1, args.repeats + 1):
            temp_root = Path(
                tempfile.mkdtemp(
                    prefix=f"aot_bench_{mode}_{repeat}_",
                    dir=tmp_root_dir,
                )
            )
            try:
                env = os.environ.copy()
                if args.isolate_compiler_caches:
                    inductor_dir = temp_root / "inductor_cache"
                    triton_dir = temp_root / "triton_cache"
                    inductor_dir.mkdir(parents=True, exist_ok=True)
                    triton_dir.mkdir(parents=True, exist_ok=True)
                    env["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
                    env["TRITON_CACHE_DIR"] = str(triton_dir)

                cmd = build_subprocess_cmd(args, mode)
                wall_start = time.perf_counter()
                proc = subprocess.run(
                    cmd,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                wall_elapsed = time.perf_counter() - wall_start
                if proc.returncode != 0:
                    print(f"[bench] subprocess failed for mode={mode} repeat={repeat}")
                    if proc.stdout:
                        print(proc.stdout)
                    if proc.stderr:
                        print(proc.stderr)
                    raise RuntimeError("Benchmark subprocess failed.")

                result = extract_json(proc.stdout)
                result["subprocess_wall_seconds"] = wall_elapsed
                all_results[mode].append(result)
                print(
                    f"[bench] mode={mode} repeat={repeat}: "
                    f"ttft_cold={result['ttft_cold_start_seconds']:.2f}s, "
                    f"ttft_ready={result['ttft_after_ready_seconds']:.2f}s, "
                    f"mean={result['mean_request_seconds']:.2f}s, "
                    f"tokens/s_all={result['tokens_per_second_all']:.2f}, "
                    f"tail_tokens/s={result['tail_tokens_per_second']:.2f}"
                )
            finally:
                shutil.rmtree(temp_root, ignore_errors=True)

    print("\n=== Benchmark Summary ===")
    for mode in MODES:
        mode_results = all_results[mode]
        cold_first_avg = statistics.mean(
            [r["cold_start_first_request_seconds"] for r in mode_results]
        )
        first_avg = statistics.mean([r["first_request_seconds"] for r in mode_results])
        tail_avg = statistics.mean([r["tail_mean_seconds"] for r in mode_results])
        mean_avg = statistics.mean([r["mean_request_seconds"] for r in mode_results])
        tps_avg = statistics.mean([r["tokens_per_second_all"] for r in mode_results])
        tail_tps_avg = statistics.mean([r["tail_tokens_per_second"] for r in mode_results])
        end_to_end_tps_avg = statistics.mean(
            [r["end_to_end_tokens_per_second"] for r in mode_results]
        )
        total_script_avg = statistics.mean([r["total_script_seconds"] for r in mode_results])
        print(
            f"{mode}: ttft_cold={cold_first_avg:.2f}s, ttft_ready={first_avg:.2f}s, "
            f"tail_mean={tail_avg:.2f}s, mean={mean_avg:.2f}s, "
            f"tokens/s_all={tps_avg:.2f}, tail_tokens/s={tail_tps_avg:.2f}, "
            f"end_to_end_tokens/s={end_to_end_tps_avg:.2f}, total_script={total_script_avg:.2f}s"
        )

    preload_cold_first = statistics.mean(
        [r["cold_start_first_request_seconds"] for r in all_results["compile_preload"]]
    )
    noload_cold_first = statistics.mean(
        [r["cold_start_first_request_seconds"] for r in all_results["compile_no_preload"]]
    )
    eager_cold_first = statistics.mean(
        [r["cold_start_first_request_seconds"] for r in all_results["eager"]]
    )
    print(
        "compile_preload cold-start speedup vs compile_no_preload: "
        f"{noload_cold_first / preload_cold_first:.2f}x"
        if preload_cold_first > 0
        else "compile_preload cold-start speedup vs compile_no_preload: n/a"
    )
    print(
        "compile_preload cold-start speedup vs eager: "
        f"{eager_cold_first / preload_cold_first:.2f}x"
        if preload_cold_first > 0
        else "compile_preload cold-start speedup vs eager: n/a"
    )


def main() -> None:
    args = parse_args()
    if args.single_mode:
        result = make_single_mode_metrics(args, args.single_mode)
        if args.emit_json:
            print(f"{BENCH_PREFIX}{json.dumps(result, sort_keys=True)}")
        return
    run_parent_benchmark(args)


if __name__ == "__main__":
    main()
