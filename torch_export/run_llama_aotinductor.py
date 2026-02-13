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
from torch._dynamo.utils import counters
from transformers import AutoTokenizer

from aot_export_utils import (
    CausalLMLogitsWrapper,
    load_model_and_tokenizer,
    pad_inputs_to_multiple,
    parse_positive_int_csv,
    require_cuda,
    resolve_dynamic_seq_dim,
    resolve_dtype,
)

METRICS_PREFIX = "RUN_METRICS_JSON:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Llama 3.2 3B Instruct inference, with optional "
            "compare mode for preload vs no-preload runtime paths."
        )
    )
    parser.add_argument(
        "--package-path",
        default="artifacts/llama3b_aotinductor.pt2",
        help="Path to AOTInductor .pt2 package.",
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Tokenizer source model id.",
    )
    parser.add_argument(
        "--metadata-path",
        default="artifacts/llama3b_aotinductor_meta.json",
        help="Metadata JSON path used for optional defaults.",
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
        help="Greedy decode steps (naive no-KV-cache loop).",
    )
    parser.add_argument(
        "--bucket-pad",
        action="store_true",
        default=True,
        help="Pad prompt to nearest prefill bucket.",
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
        help="Comma-separated prefill buckets for --bucket-pad.",
    )
    parser.add_argument(
        "--stop-on-eos",
        action="store_true",
        default=True,
        help="Stop generation early when EOS token is generated.",
    )
    parser.add_argument(
        "--no-stop-on-eos",
        dest="stop_on_eos",
        action="store_false",
        help="Disable EOS early stop.",
    )
    parser.add_argument(
        "--dynamic-seq-multiple",
        type=int,
        default=0,
        help=(
            "Override runtime sequence multiple padding for dynamic packages. "
            "Use 0 to read from metadata."
        ),
    )
    parser.add_argument(
        "--load-artifacts",
        action="store_true",
        default=True,
        help="Use prebuilt .pt2 package preload path (compile_preload).",
    )
    parser.add_argument(
        "--no-load-artifacts",
        dest="load_artifacts",
        action="store_false",
        help="Skip prebuilt package and use runtime torch.compile path (compile_no_preload).",
    )
    parser.add_argument(
        "--compare-load-vs-skip",
        action="store_true",
        default=False,
        help="Run isolated cold-start comparisons for both preload and no-preload modes.",
    )
    parser.add_argument(
        "--compare-runs",
        type=int,
        default=1,
        help="Number of isolated runs per mode for --compare-load-vs-skip.",
    )
    parser.add_argument(
        "--show-completions",
        action="store_true",
        default=False,
        help="Print generated completion text during compare mode.",
    )
    parser.add_argument(
        "--no-show-completions",
        dest="show_completions",
        action="store_false",
        help="Disable completion text printing during compare mode.",
    )
    parser.add_argument(
        "--isolate-compiler-caches",
        action="store_true",
        default=True,
        help="Use fresh TORCHINDUCTOR/TRITON cache dirs per compare subprocess.",
    )
    parser.add_argument(
        "--no-isolate-compiler-caches",
        dest="isolate_compiler_caches",
        action="store_false",
        help="Reuse current compiler cache dirs in compare mode.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Model dtype for runtime compile path (no-preload mode).",
    )
    parser.add_argument(
        "--compile-dynamic-seq-len",
        action="store_true",
        default=True,
        help="Set torch.compile(dynamic=True) in no-preload mode.",
    )
    parser.add_argument(
        "--no-compile-dynamic-seq-len",
        dest="compile_dynamic_seq_len",
        action="store_false",
        help="Set torch.compile(dynamic=False) in no-preload mode.",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=1,
        help="Reference min seq length for no-preload logging/validation.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Reference max seq length for no-preload logging/validation.",
    )
    parser.add_argument(
        "--compile-dynamic-seq-multiple",
        type=int,
        default=8,
        help="Pad per-step sequence length to this multiple in no-preload mode.",
    )
    parser.add_argument(
        "--max-autotune",
        action="store_true",
        default=True,
        help="Use torch.compile(mode='max-autotune') in no-preload mode.",
    )
    parser.add_argument(
        "--no-max-autotune",
        dest="max_autotune",
        action="store_false",
        help="Use torch.compile(mode='default') in no-preload mode.",
    )
    parser.add_argument(
        "--masking-workaround",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-masking-workaround",
        dest="masking_workaround",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--emit-json-metrics",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.dynamic_seq_multiple < 0:
        raise ValueError("--dynamic-seq-multiple must be >= 0")
    if args.compare_runs <= 0:
        raise ValueError("--compare-runs must be positive")
    if args.min_seq_len <= 0:
        raise ValueError("--min-seq-len must be positive")
    if args.max_seq_len < args.min_seq_len:
        raise ValueError("--max-seq-len must be >= --min-seq-len")
    if args.compile_dynamic_seq_multiple <= 0:
        raise ValueError("--compile-dynamic-seq-multiple must be positive")
    return args


def resolve_prefill_buckets(args: argparse.Namespace, metadata: dict[str, Any]) -> list[int]:
    if args.prefill_buckets.strip():
        return parse_positive_int_csv(args.prefill_buckets)
    example_seq_len = metadata.get("example", {}).get("seq_len")
    if isinstance(example_seq_len, int) and example_seq_len > 0:
        return [example_seq_len]
    return []


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
        print("[run] bucket pad requested, but no prefill buckets resolved; using raw prompt.")
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
        "[run] bucket padding enabled: "
        f"prompt_tokens={original_prompt_tokens} -> padded_tokens={target}"
    )
    return padded, original_prompt_tokens, target


def generate_with_compiled_model(
    *,
    compiled_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    stop_on_eos: bool,
    eos_token_id: int | None,
    dynamic_seq_multiple: int,
    pad_token_id: int | None,
    device: torch.device,
) -> tuple[list[int], float]:
    generated_ids: list[int] = []

    gen_start = time.perf_counter()
    with torch.inference_mode():
        req_input_ids = input_ids.clone()
        req_attention_mask = attention_mask.clone()
        for _ in range(max_new_tokens):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            model_input_ids, model_attention_mask, logit_seq_len = pad_inputs_to_multiple(
                req_input_ids,
                req_attention_mask,
                seq_multiple=dynamic_seq_multiple,
                pad_token_id=pad_token_id,
            )
            logits = compiled_model(model_input_ids, model_attention_mask)
            next_token = logits[:, logit_seq_len - 1, :].argmax(dim=-1, keepdim=True)
            next_id = int(next_token.item())
            generated_ids.append(next_id)

            req_input_ids = torch.cat([req_input_ids, next_token], dim=1)
            next_mask = torch.ones(
                (req_attention_mask.shape[0], 1),
                dtype=req_attention_mask.dtype,
                device=device,
            )
            req_attention_mask = torch.cat([req_attention_mask, next_mask], dim=1)

            if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                break
    torch.cuda.synchronize()
    gen_seconds = time.perf_counter() - gen_start
    return generated_ids, gen_seconds


def print_result(
    *,
    mode: str,
    prompt: str,
    completion: str,
    setup_seconds: float,
    generation_seconds: float,
    original_tokens: int,
    effective_tokens: int,
    new_tokens: int,
) -> None:
    print(f"\n=== Mode: {mode} ===")
    print("\n=== Prompt ===")
    print(prompt)
    print("\n=== Completion ===")
    print(completion)
    print(f"\n[run] setup latency: {setup_seconds:.2f}s")
    print(f"[run] generation latency: {generation_seconds:.2f}s")
    print(
        "[run] token info: "
        f"prompt_original={original_tokens}, prompt_effective={effective_tokens}, "
        f"new_tokens={new_tokens}"
    )


def collect_compile_counters() -> dict[str, int]:
    compile_stats = dict(counters.get("stats", {}))
    inductor_stats = dict(counters.get("inductor", {}))
    aot_stats = dict(counters.get("aot_autograd", {}))
    return {
        "unique_graphs": int(compile_stats.get("unique_graphs", 0)),
        "calls_captured": int(compile_stats.get("calls_captured", 0)),
        "async_compile_miss": int(inductor_stats.get("async_compile_cache_miss", 0)),
        "async_compile_hit": int(inductor_stats.get("async_compile_cache_hit", 0)),
        "fxgraph_cache_miss": int(inductor_stats.get("fxgraph_cache_miss", 0)),
        "fxgraph_cache_hit": int(inductor_stats.get("fxgraph_cache_hit", 0)),
        "aot_cache_miss": int(aot_stats.get("autograd_cache_miss", 0)),
        "aot_cache_hit": int(aot_stats.get("autograd_cache_hit", 0)),
    }


def safe_tps(tokens: int, seconds: float) -> float:
    if tokens <= 0 or seconds <= 0:
        return 0.0
    return float(tokens / max(1e-9, seconds))


def run_preload_mode(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    start_overall = time.perf_counter()
    metadata = load_metadata(Path(args.metadata_path))
    package_path = Path(args.package_path)
    if not package_path.exists():
        raise FileNotFoundError(
            f"AOTInductor package not found: {package_path}. "
            "Run build_llama_aotinductor.py first."
        )

    print(f"[run] mode=compile_preload loading tokenizer: {args.model_id}")
    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_load_seconds = time.perf_counter() - tokenizer_start

    prefill_buckets = resolve_prefill_buckets(args, metadata)
    encoded_cpu, original_tokens, effective_tokens = maybe_bucket_pad_inputs(
        tokenizer=tokenizer,
        prompt=args.prompt,
        prefill_buckets=prefill_buckets,
        enable_bucket_pad=args.bucket_pad,
    )
    input_ids = encoded_cpu["input_ids"].to(device)
    attention_mask = encoded_cpu["attention_mask"].to(device)

    print(f"[run] loading AOT package: {package_path}")
    load_start = time.perf_counter()
    compiled_model = torch._inductor.aoti_load_package(str(package_path))
    artifact_load_seconds = time.perf_counter() - load_start
    print(f"[run] package loaded in {artifact_load_seconds:.2f}s")

    export_meta = metadata.get("export", {})
    dynamic_seq_len = bool(export_meta.get("dynamic_seq_len", False))
    metadata_dynamic_multiple = export_meta.get("dynamic_seq_multiple", 1)
    if not isinstance(metadata_dynamic_multiple, int) or metadata_dynamic_multiple <= 0:
        metadata_dynamic_multiple = 1
    dynamic_seq_multiple = (
        args.dynamic_seq_multiple if args.dynamic_seq_multiple > 0 else metadata_dynamic_multiple
    )
    expected_seq_len = metadata.get("example", {}).get("seq_len")
    if not dynamic_seq_len:
        if isinstance(expected_seq_len, int) and expected_seq_len > 0 and input_ids.shape[1] != expected_seq_len:
            raise ValueError(
                "This package was built with static sequence length "
                f"{expected_seq_len}, but runtime prompt length is {input_ids.shape[1]}. "
                "Use --bucket-pad with matching buckets, or rebuild with a different --example-seq-len."
            )
        if args.max_new_tokens > 1:
            raise ValueError(
                "This package was built with static sequence length. "
                "Autoregressive loops that grow input length require dynamic export. "
                "Use --max-new-tokens 1, or rebuild with --dynamic-seq-len."
            )
        dynamic_seq_multiple = 1
    elif dynamic_seq_multiple > 1:
        print(
            "[run] dynamic sequence multiple padding enabled: "
            f"multiple={dynamic_seq_multiple}"
        )

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    print(
        "[run] generating with preloaded AOT package "
        f"(max_new_tokens={args.max_new_tokens})"
    )
    counters.clear()
    generated_ids, generation_seconds = generate_with_compiled_model(
        compiled_model=compiled_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        stop_on_eos=args.stop_on_eos,
        eos_token_id=eos_token_id,
        dynamic_seq_multiple=dynamic_seq_multiple,
        pad_token_id=pad_token_id,
        device=device,
    )
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    counter_metrics = collect_compile_counters()
    print(
        "[run] dynamo stats: "
        f"unique_graphs={counter_metrics['unique_graphs']}, "
        f"calls_captured={counter_metrics['calls_captured']}"
    )
    print(
        "[run] inductor cache stats: "
        f"async_miss={counter_metrics['async_compile_miss']}, "
        f"async_hit={counter_metrics['async_compile_hit']}, "
        f"fxgraph_miss={counter_metrics['fxgraph_cache_miss']}, "
        f"fxgraph_hit={counter_metrics['fxgraph_cache_hit']}, "
        f"aot_miss={counter_metrics['aot_cache_miss']}, "
        f"aot_hit={counter_metrics['aot_cache_hit']}"
    )
    setup_seconds = tokenizer_load_seconds + artifact_load_seconds
    print_result(
        mode="compile_preload",
        prompt=args.prompt,
        completion=completion,
        setup_seconds=setup_seconds,
        generation_seconds=generation_seconds,
        original_tokens=original_tokens,
        effective_tokens=effective_tokens,
        new_tokens=len(generated_ids),
    )
    return {
        "mode": "compile_preload",
        "load_artifacts": True,
        "setup_seconds": setup_seconds,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_load_seconds": 0.0,
        "artifact_build_seconds": 0.0,
        "artifact_load_seconds": artifact_load_seconds,
        "generation_seconds": generation_seconds,
        "total_script_seconds": time.perf_counter() - start_overall,
        "completion": completion,
        "new_tokens": len(generated_ids),
        "prompt_tokens_original": original_tokens,
        "prompt_tokens_effective": effective_tokens,
        **counter_metrics,
    }


def run_no_preload_mode(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    start_overall = time.perf_counter()
    metadata = load_metadata(Path(args.metadata_path))
    dtype, dtype_name = resolve_dtype(args.dtype, device)

    print(f"[run] mode=compile_no_preload loading model/tokenizer: {args.model_id}")
    model_load_start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(args.model_id, device, dtype)
    model_load_seconds = time.perf_counter() - model_load_start

    prefill_buckets = resolve_prefill_buckets(args, metadata)
    encoded_cpu, original_tokens, effective_tokens = maybe_bucket_pad_inputs(
        tokenizer=tokenizer,
        prompt=args.prompt,
        prefill_buckets=prefill_buckets,
        enable_bucket_pad=args.bucket_pad,
    )
    runtime_input_ids_cpu = encoded_cpu["input_ids"]
    runtime_attention_mask_cpu = encoded_cpu["attention_mask"]
    input_ids = runtime_input_ids_cpu.to(device)
    attention_mask = runtime_attention_mask_cpu.to(device)

    dynamic_seq_multiple = args.compile_dynamic_seq_multiple
    if dynamic_seq_multiple <= 1:
        dynamic_seq_multiple = 1

    effective_min_seq_len = args.min_seq_len
    effective_max_seq_len = args.max_seq_len
    if args.compile_dynamic_seq_len:
        _, effective_min_seq_len, effective_max_seq_len = resolve_dynamic_seq_dim(
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            seq_multiple=dynamic_seq_multiple,
        )
        print(
            "[run] no-preload dynamic compile enabled: "
            f"dynamic=True, reference_seq=[{effective_min_seq_len}, {effective_max_seq_len}], "
            f"seq_multiple={dynamic_seq_multiple}"
        )
    else:
        print(
            "[run] no-preload dynamic compile disabled: "
            f"dynamic=False, seq_multiple={dynamic_seq_multiple}"
        )

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    compile_input_ids, compile_attention_mask, _ = pad_inputs_to_multiple(
        input_ids,
        attention_mask,
        seq_multiple=dynamic_seq_multiple,
        pad_token_id=pad_token_id,
    )

    wrapped = CausalLMLogitsWrapper(model).eval()
    compile_mode = "max-autotune" if args.max_autotune else "default"
    print(
        "[run] compiling model at runtime with torch.compile "
        f"(dtype={dtype_name}, mode={compile_mode}, dynamic={args.compile_dynamic_seq_len}, "
        "fullgraph=True)"
    )
    counters.clear()
    build_start = time.perf_counter()
    compiled_model = torch.compile(
        wrapped,
        mode=compile_mode,
        fullgraph=True,
        dynamic=args.compile_dynamic_seq_len,
    )
    with torch.inference_mode():
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        _ = compiled_model(compile_input_ids, compile_attention_mask)
    torch.cuda.synchronize()
    artifact_build_seconds = time.perf_counter() - build_start
    artifact_load_seconds = 0.0
    print(f"[run] runtime torch.compile warmup done in {artifact_build_seconds:.2f}s")

    print(
        "[run] generating with runtime torch.compile model "
        f"(max_new_tokens={args.max_new_tokens})"
    )
    generated_ids, generation_seconds = generate_with_compiled_model(
        compiled_model=compiled_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        stop_on_eos=args.stop_on_eos,
        eos_token_id=eos_token_id,
        dynamic_seq_multiple=dynamic_seq_multiple,
        pad_token_id=pad_token_id,
        device=device,
    )
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    counter_metrics = collect_compile_counters()
    print(
        "[run] dynamo stats: "
        f"unique_graphs={counter_metrics['unique_graphs']}, "
        f"calls_captured={counter_metrics['calls_captured']}"
    )
    print(
        "[run] inductor cache stats: "
        f"async_miss={counter_metrics['async_compile_miss']}, "
        f"async_hit={counter_metrics['async_compile_hit']}, "
        f"fxgraph_miss={counter_metrics['fxgraph_cache_miss']}, "
        f"fxgraph_hit={counter_metrics['fxgraph_cache_hit']}, "
        f"aot_miss={counter_metrics['aot_cache_miss']}, "
        f"aot_hit={counter_metrics['aot_cache_hit']}"
    )
    setup_seconds = model_load_seconds + artifact_build_seconds + artifact_load_seconds
    print_result(
        mode="compile_no_preload",
        prompt=args.prompt,
        completion=completion,
        setup_seconds=setup_seconds,
        generation_seconds=generation_seconds,
        original_tokens=original_tokens,
        effective_tokens=effective_tokens,
        new_tokens=len(generated_ids),
    )
    return {
        "mode": "compile_no_preload",
        "load_artifacts": False,
        "setup_seconds": setup_seconds,
        "tokenizer_load_seconds": 0.0,
        "model_load_seconds": model_load_seconds,
        "artifact_build_seconds": artifact_build_seconds,
        "artifact_load_seconds": artifact_load_seconds,
        "generation_seconds": generation_seconds,
        "total_script_seconds": time.perf_counter() - start_overall,
        "completion": completion,
        "new_tokens": len(generated_ids),
        "prompt_tokens_original": original_tokens,
        "prompt_tokens_effective": effective_tokens,
        "fullgraph": True,
        **counter_metrics,
    }


def extract_json_metrics(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(METRICS_PREFIX):
            return json.loads(line[len(METRICS_PREFIX) :])
    raise RuntimeError("Subprocess did not emit metrics JSON.")


def build_compare_subprocess_cmd(args: argparse.Namespace, load_artifacts: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--emit-json-metrics",
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
        "--prefill-buckets",
        args.prefill_buckets,
        "--dynamic-seq-multiple",
        str(args.dynamic_seq_multiple),
        "--dtype",
        args.dtype,
        "--min-seq-len",
        str(args.min_seq_len),
        "--max-seq-len",
        str(args.max_seq_len),
        "--compile-dynamic-seq-multiple",
        str(args.compile_dynamic_seq_multiple),
    ]
    cmd.append("--bucket-pad" if args.bucket_pad else "--no-bucket-pad")
    cmd.append("--stop-on-eos" if args.stop_on_eos else "--no-stop-on-eos")
    cmd.append("--compile-dynamic-seq-len" if args.compile_dynamic_seq_len else "--no-compile-dynamic-seq-len")
    cmd.append("--max-autotune" if args.max_autotune else "--no-max-autotune")
    cmd.append("--masking-workaround" if args.masking_workaround else "--no-masking-workaround")
    cmd.append("--load-artifacts" if load_artifacts else "--no-load-artifacts")
    return cmd


def run_compare(args: argparse.Namespace) -> None:
    if args.isolate_compiler_caches:
        print(
            "[compare] running cold-start comparison with isolated compiler caches "
            f"(trials per mode={args.compare_runs})"
        )
    else:
        print(
            "[compare] running comparison with shared compiler caches "
            f"(trials per mode={args.compare_runs})"
        )

    results_by_mode: dict[str, list[dict[str, Any]]] = {"load": [], "skip": []}
    mode_plan = [("load", True, "compile_preload"), ("skip", False, "compile_no_preload")]
    tmp_root_dir = os.environ.get("TMPDIR")
    if tmp_root_dir:
        Path(tmp_root_dir).mkdir(parents=True, exist_ok=True)

    for mode_name, should_load, mode_label in mode_plan:
        for trial in range(1, args.compare_runs + 1):
            temp_root = Path(
                tempfile.mkdtemp(
                    prefix=f"run_compare_{mode_label}_{trial}_",
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
                cmd = build_compare_subprocess_cmd(args, load_artifacts=should_load)
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
                    print(f"[compare] subprocess failed for mode={mode_label} run={trial}")
                    if proc.stdout:
                        print(proc.stdout)
                    if proc.stderr:
                        print(proc.stderr)
                    raise RuntimeError("Compare subprocess failed.")
                metrics = extract_json_metrics(proc.stdout)
                metrics["subprocess_wall_seconds"] = wall_elapsed
                results_by_mode[mode_name].append(metrics)
                trial_new_tokens = int(metrics.get("new_tokens", 0))
                trial_tps = safe_tps(trial_new_tokens, float(metrics["generation_seconds"]))
                print(
                    f"[compare] mode={mode_name} trial={trial}: "
                    f"setup={metrics['setup_seconds']:.2f}s, "
                    f"generation={metrics['generation_seconds']:.2f}s, "
                    f"script_total={metrics['total_script_seconds']:.2f}s, "
                    f"wall={wall_elapsed:.2f}s, "
                    f"new_tokens={trial_new_tokens}, "
                    f"gen_tps={trial_tps:.2f}"
                )
                if args.show_completions:
                    print(f"[compare] completion({mode_name}): {metrics.get('completion', '').strip()}")
            finally:
                shutil.rmtree(temp_root, ignore_errors=True)

    load_setup = [m["setup_seconds"] for m in results_by_mode["load"]]
    skip_setup = [m["setup_seconds"] for m in results_by_mode["skip"]]
    load_gen = [m["generation_seconds"] for m in results_by_mode["load"]]
    skip_gen = [m["generation_seconds"] for m in results_by_mode["skip"]]
    load_total = [m["total_script_seconds"] for m in results_by_mode["load"]]
    skip_total = [m["total_script_seconds"] for m in results_by_mode["skip"]]
    load_graphs = [m.get("unique_graphs", 0) for m in results_by_mode["load"]]
    skip_graphs = [m.get("unique_graphs", 0) for m in results_by_mode["skip"]]
    load_async_miss = [m.get("async_compile_miss", 0) for m in results_by_mode["load"]]
    skip_async_miss = [m.get("async_compile_miss", 0) for m in results_by_mode["skip"]]
    load_tokens = [int(m.get("new_tokens", 0)) for m in results_by_mode["load"]]
    skip_tokens = [int(m.get("new_tokens", 0)) for m in results_by_mode["skip"]]
    load_tps = [
        safe_tps(int(m.get("new_tokens", 0)), float(m.get("generation_seconds", 0.0)))
        for m in results_by_mode["load"]
    ]
    skip_tps = [
        safe_tps(int(m.get("new_tokens", 0)), float(m.get("generation_seconds", 0.0)))
        for m in results_by_mode["skip"]
    ]

    mean_load_setup = statistics.mean(load_setup)
    mean_skip_setup = statistics.mean(skip_setup)
    mean_load_gen = statistics.mean(load_gen)
    mean_skip_gen = statistics.mean(skip_gen)
    mean_load_total = statistics.mean(load_total)
    mean_skip_total = statistics.mean(skip_total)
    mean_load_tps = statistics.mean(load_tps)
    mean_skip_tps = statistics.mean(skip_tps)

    print("\n=== Comparison Summary ===")
    print(f"setup (load artifacts): {mean_load_setup:.2f}s avg over {len(load_setup)} run(s)")
    print(f"setup (skip artifacts): {mean_skip_setup:.2f}s avg over {len(skip_setup)} run(s)")
    print(
        "setup speedup from load artifacts: "
        f"{mean_skip_setup / mean_load_setup:.2f}x"
        if mean_load_setup > 0
        else "setup speedup from load artifacts: n/a"
    )
    print(f"generation (load artifacts): {mean_load_gen:.2f}s avg over {len(load_gen)} run(s)")
    print(f"generation (skip artifacts): {mean_skip_gen:.2f}s avg over {len(skip_gen)} run(s)")
    print(f"new_tokens (load artifacts): {statistics.mean(load_tokens):.2f} avg")
    print(f"new_tokens (skip artifacts): {statistics.mean(skip_tokens):.2f} avg")
    print(f"generation token/s (load artifacts): {mean_load_tps:.2f}")
    print(f"generation token/s (skip artifacts): {mean_skip_tps:.2f}")
    print(
        "generation speedup from load artifacts: "
        f"{mean_skip_gen / mean_load_gen:.2f}x"
        if mean_load_gen > 0
        else "generation speedup from load artifacts: n/a"
    )
    print(f"total script (load artifacts): {mean_load_total:.2f}s avg")
    print(f"total script (skip artifacts): {mean_skip_total:.2f}s avg")
    print(
        "total script speedup from load artifacts: "
        f"{mean_skip_total / mean_load_total:.2f}x"
        if mean_load_total > 0
        else "total script speedup from load artifacts: n/a"
    )
    print(f"avg unique_graphs (load artifacts): {statistics.mean(load_graphs):.2f}")
    print(f"avg unique_graphs (skip artifacts): {statistics.mean(skip_graphs):.2f}")
    print(f"avg async_compile_miss (load artifacts): {statistics.mean(load_async_miss):.2f}")
    print(f"avg async_compile_miss (skip artifacts): {statistics.mean(skip_async_miss):.2f}")


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    device = require_cuda()
    if args.load_artifacts:
        return run_preload_mode(args, device)
    return run_no_preload_mode(args, device)


def main() -> None:
    args = parse_args()
    if args.compare_load_vs_skip:
        run_compare(args)
        return
    metrics = run_single(args)
    if args.emit_json_metrics:
        print(f"{METRICS_PREFIX}{json.dumps(metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
