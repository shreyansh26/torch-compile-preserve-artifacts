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
from transformers.generation.configuration_utils import CompileConfig

from llama_compile_cache_utils import (
    load_model_and_tokenizer,
    parse_positive_int_csv,
    require_cuda,
    resolve_dtype,
    runtime_fingerprint,
)

METRICS_PREFIX = "METRICS_JSON:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load portable torch.compile cache artifacts, then run "
            "Llama 3.2 3B Instruct inference with torch.compile."
        )
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--cache-path",
        default="artifacts/llama3b_torchcompile_cache.bin",
        help="Path to serialized portable cache artifacts.",
    )
    parser.add_argument(
        "--metadata-path",
        default="artifacts/llama3b_torchcompile_cache_meta.json",
        help="Path to metadata JSON from the build stage.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain how torch.compile cache artifacts reduce startup latency.",
        help="Prompt for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation length.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Model dtype on GPU.",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        help="torch.compile mode.",
    )
    parser.add_argument(
        "--cache-implementation",
        default="static",
        choices=["static", "hybrid", "sliding_window"],
        help="KV cache implementation used by generate().",
    )
    parser.add_argument(
        "--compile-dynamic",
        choices=["auto", "true", "false"],
        default="auto",
        help="CompileConfig.dynamic value for generation auto-compile.",
    )
    parser.add_argument(
        "--bucket-pad",
        action="store_true",
        help="Pad prompt length to nearest warmed prefill bucket (recommended for cache hit rate).",
    )
    parser.add_argument(
        "--prefill-buckets",
        default="",
        help=(
            "Comma-separated prefill buckets for --bucket-pad. "
            "If empty, tries metadata warmup.prefill_lengths."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0.0 uses greedy decoding.",
    )
    parser.add_argument(
        "--load-artifacts",
        dest="load_artifacts",
        action="store_true",
        default=True,
        help="Load portable cache artifacts before model construction.",
    )
    parser.add_argument(
        "--no-load-artifacts",
        dest="load_artifacts",
        action="store_false",
        help="Skip loading portable cache artifacts.",
    )
    parser.add_argument(
        "--compare-load-vs-skip",
        action="store_true",
        help=(
            "Run two isolated cold-start trials (artifact load ON and OFF) "
            "and print latency comparison."
        ),
    )
    parser.add_argument(
        "--compare-runs",
        type=int,
        default=1,
        help="Number of cold-start trials per mode for --compare-load-vs-skip.",
    )
    parser.add_argument(
        "--fullgraph",
        dest="fullgraph",
        action="store_true",
        default=True,
        help="Enable fullgraph=True for torch.compile.",
    )
    parser.add_argument(
        "--no-fullgraph",
        dest="fullgraph",
        action="store_false",
        help="Disable fullgraph=True for torch.compile.",
    )
    parser.add_argument(
        "--emit-json-metrics",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be a positive integer")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be >= 0.0")
    if args.compare_runs <= 0:
        raise ValueError("--compare-runs must be a positive integer")
    return args


def maybe_report_mismatch(args: argparse.Namespace, dtype_name: str, device: torch.device) -> None:
    metadata_path = Path(args.metadata_path)
    if not metadata_path.exists():
        print(f"[run] metadata file not found, skipping compatibility checks: {metadata_path}")
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    warnings: list[str] = []
    if metadata.get("model_id") != args.model_id:
        warnings.append(f"model_id differs (build={metadata.get('model_id')}, run={args.model_id})")

    compile_meta = metadata.get("compile", {})
    if compile_meta.get("mode") != args.compile_mode:
        warnings.append(
            f"compile mode differs (build={compile_meta.get('mode')}, run={args.compile_mode})"
        )
    if bool(compile_meta.get("fullgraph")) != bool(args.fullgraph):
        warnings.append(
            f"fullgraph differs (build={compile_meta.get('fullgraph')}, run={args.fullgraph})"
        )
    if compile_meta.get("dtype") != dtype_name:
        warnings.append(f"dtype differs (build={compile_meta.get('dtype')}, run={dtype_name})")
    if compile_meta.get("cache_implementation") != args.cache_implementation:
        warnings.append(
            "cache_implementation differs "
            f"(build={compile_meta.get('cache_implementation')}, run={args.cache_implementation})"
        )
    build_dynamic = compile_meta.get("dynamic")
    dynamic_map = {"auto": None, "true": True, "false": False}
    run_dynamic = dynamic_map[args.compile_dynamic]
    if build_dynamic != run_dynamic:
        warnings.append(f"compile dynamic differs (build={build_dynamic}, run={run_dynamic})")
    build_decode_length = metadata.get("warmup", {}).get("decode_length")
    if isinstance(build_decode_length, int) and args.max_new_tokens > build_decode_length:
        warnings.append(
            "max_new_tokens exceeds build decode_length "
            f"(build={build_decode_length}, run={args.max_new_tokens})"
        )

    build_fingerprint = metadata.get("runtime_fingerprint", {})
    run_fingerprint = runtime_fingerprint(device)
    if build_fingerprint.get("torch_version") != run_fingerprint.get("torch_version"):
        warnings.append(
            "torch_version differs "
            f"(build={build_fingerprint.get('torch_version')}, run={run_fingerprint.get('torch_version')})"
        )
    build_gpu = build_fingerprint.get("gpu", {}).get("name")
    run_gpu = run_fingerprint.get("gpu", {}).get("name")
    if build_gpu and run_gpu and build_gpu != run_gpu:
        warnings.append(f"GPU differs (build={build_gpu}, run={run_gpu})")

    if warnings:
        print("[run] WARNING: Cache key mismatch is likely. Recompilation may happen.")
        for warning in warnings:
            print(f"[run]  - {warning}")
    else:
        print("[run] metadata check passed: compile/runtime settings look compatible.")


def resolve_prefill_buckets(args: argparse.Namespace) -> list[int]:
    if args.prefill_buckets.strip():
        return parse_positive_int_csv(args.prefill_buckets)

    metadata_path = Path(args.metadata_path)
    if not metadata_path.exists():
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lengths = metadata.get("warmup", {}).get("prefill_lengths", [])
    if not isinstance(lengths, list):
        return []
    resolved: list[int] = []
    for value in lengths:
        if isinstance(value, int) and value > 0:
            resolved.append(value)
    return resolved


def maybe_bucket_pad_inputs(
    *,
    tokenizer,
    prompt: str,
    prefill_buckets: list[int],
    enable_bucket_pad: bool,
) -> tuple[dict[str, torch.Tensor], int, int]:
    encoded = tokenizer(prompt, return_tensors="pt")
    original_prompt_tokens = int(encoded["input_ids"].shape[1])
    if not enable_bucket_pad:
        return encoded, original_prompt_tokens, original_prompt_tokens

    if not prefill_buckets:
        print(
            "[run] --bucket-pad requested, but no prefill buckets were resolved; "
            "using unpadded prompt."
        )
        return encoded, original_prompt_tokens, original_prompt_tokens

    buckets = sorted(prefill_buckets)
    target = next((bucket for bucket in buckets if bucket >= original_prompt_tokens), None)
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


def run_single_inference(args: argparse.Namespace) -> dict[str, Any]:
    overall_start = time.perf_counter()
    device = require_cuda()
    dtype, dtype_name = resolve_dtype(args.dtype, device)

    artifact_bytes = 0
    artifact_load_seconds = 0.0
    if args.load_artifacts:
        cache_path = Path(args.cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Portable cache file was not found: {cache_path}. "
                "Run build_llama_portable_cache.py first."
            )
        print(f"[run] loading portable cache artifacts: {cache_path}")
        load_start = time.perf_counter()
        artifact_payload = cache_path.read_bytes()
        cache_info = torch.compiler.load_cache_artifacts(artifact_payload)
        artifact_load_seconds = time.perf_counter() - load_start
        artifact_bytes = len(artifact_payload)
        print(f"[run] loaded cache bytes: {artifact_bytes}")
        print(f"[run] cache preload latency: {artifact_load_seconds:.2f}s")
        if cache_info is not None:
            print("[run] cache artifacts were registered in torch compiler caches.")
        maybe_report_mismatch(args, dtype_name, device)
    else:
        print("[run] skipping portable cache preload (--no-load-artifacts)")

    print(f"[run] loading model/tokenizer: {args.model_id}")
    model_load_start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(args.model_id, device, dtype)
    model_load_seconds = time.perf_counter() - model_load_start

    dynamic_map = {"auto": None, "true": True, "false": False}
    compile_dynamic = dynamic_map[args.compile_dynamic]
    compile_config = CompileConfig(
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
        dynamic=compile_dynamic,
    )
    print(
        "[run] generate auto-compile settings: "
        f"mode={args.compile_mode}, fullgraph={args.fullgraph}, "
        f"dynamic={compile_dynamic}, cache_implementation={args.cache_implementation}, "
        f"dtype={dtype_name}"
    )

    prefill_buckets = resolve_prefill_buckets(args)
    encoded_cpu, original_prompt_tokens, effective_prompt_tokens = maybe_bucket_pad_inputs(
        tokenizer=tokenizer,
        prompt=args.prompt,
        prefill_buckets=prefill_buckets,
        enable_bucket_pad=args.bucket_pad,
    )
    encoded = encoded_cpu.to(device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")

    do_sample = args.temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": pad_token_id,
        "do_sample": do_sample,
        "cache_implementation": args.cache_implementation,
        "compile_config": compile_config,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature

    print("[run] running generation")
    counters.clear()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    prompt_len = encoded["input_ids"].shape[1]
    completion = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Completion ===")
    print(completion.strip())
    print(f"\n[run] end-to-end generation latency: {elapsed:.2f}s")
    compile_stats = dict(counters.get("stats", {}))
    inductor_stats = dict(counters.get("inductor", {}))
    aot_stats = dict(counters.get("aot_autograd", {}))
    unique_graphs = int(compile_stats.get("unique_graphs", 0))
    calls_captured = int(compile_stats.get("calls_captured", 0))
    async_compile_miss = int(inductor_stats.get("async_compile_cache_miss", 0))
    async_compile_hit = int(inductor_stats.get("async_compile_cache_hit", 0))
    fxgraph_cache_miss = int(inductor_stats.get("fxgraph_cache_miss", 0))
    fxgraph_cache_hit = int(inductor_stats.get("fxgraph_cache_hit", 0))
    aot_cache_miss = int(aot_stats.get("autograd_cache_miss", 0))
    aot_cache_hit = int(aot_stats.get("autograd_cache_hit", 0))
    print(
        f"[run] dynamo stats: unique_graphs={unique_graphs}, "
        f"calls_captured={calls_captured}"
    )
    print(
        "[run] inductor cache stats: "
        f"async_miss={async_compile_miss}, async_hit={async_compile_hit}, "
        f"fxgraph_miss={fxgraph_cache_miss}, fxgraph_hit={fxgraph_cache_hit}, "
        f"aot_miss={aot_cache_miss}, aot_hit={aot_cache_hit}"
    )

    metrics = {
        "load_artifacts": bool(args.load_artifacts),
        "artifact_bytes": artifact_bytes,
        "artifact_load_seconds": artifact_load_seconds,
        "model_load_seconds": model_load_seconds,
        "generation_seconds": elapsed,
        "prompt_tokens": int(prompt_len),
        "prompt_tokens_original": int(original_prompt_tokens),
        "prompt_tokens_effective": int(effective_prompt_tokens),
        "new_tokens": int(output_ids.shape[1] - prompt_len),
        "unique_graphs": unique_graphs,
        "calls_captured": calls_captured,
        "async_compile_miss": async_compile_miss,
        "async_compile_hit": async_compile_hit,
        "fxgraph_cache_miss": fxgraph_cache_miss,
        "fxgraph_cache_hit": fxgraph_cache_hit,
        "aot_cache_miss": aot_cache_miss,
        "aot_cache_hit": aot_cache_hit,
        "cache_implementation": args.cache_implementation,
        "compile_dynamic": compile_dynamic,
        "bucket_pad": bool(args.bucket_pad),
        "prefill_buckets": prefill_buckets,
        "total_script_seconds": time.perf_counter() - overall_start,
        "compile_mode": args.compile_mode,
        "fullgraph": bool(args.fullgraph),
        "dtype": dtype_name,
        "model_id": args.model_id,
    }
    if args.emit_json_metrics:
        print(f"{METRICS_PREFIX}{json.dumps(metrics, sort_keys=True)}")
    return metrics


def build_compare_subprocess_cmd(args: argparse.Namespace, load_artifacts: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-id",
        args.model_id,
        "--cache-path",
        args.cache_path,
        "--metadata-path",
        args.metadata_path,
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--compile-mode",
        args.compile_mode,
        "--cache-implementation",
        args.cache_implementation,
        "--compile-dynamic",
        args.compile_dynamic,
        "--prefill-buckets",
        args.prefill_buckets,
        "--temperature",
        str(args.temperature),
        "--emit-json-metrics",
    ]
    if args.bucket_pad:
        cmd.append("--bucket-pad")
    cmd.append("--fullgraph" if args.fullgraph else "--no-fullgraph")
    cmd.append("--load-artifacts" if load_artifacts else "--no-load-artifacts")
    return cmd


def extract_metrics_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(METRICS_PREFIX):
            return json.loads(line[len(METRICS_PREFIX) :])
    raise RuntimeError("Subprocess did not emit metrics JSON. Cannot compare runs.")


def run_compare_mode(args: argparse.Namespace) -> None:
    print(
        "[compare] running cold-start comparison with isolated compiler caches "
        f"(trials per mode={args.compare_runs})"
    )
    results_by_mode: dict[str, list[dict[str, Any]]] = {"load": [], "skip": []}
    mode_plan = [("load", True), ("skip", False)]

    for mode_name, should_load in mode_plan:
        for trial in range(1, args.compare_runs + 1):
            temp_root = Path(tempfile.mkdtemp(prefix=f"portable_cache_{mode_name}_{trial}_"))
            try:
                inductor_dir = temp_root / "inductor_cache"
                triton_dir = temp_root / "triton_cache"
                inductor_dir.mkdir(parents=True, exist_ok=True)
                triton_dir.mkdir(parents=True, exist_ok=True)

                cmd = build_compare_subprocess_cmd(args, should_load)
                env = os.environ.copy()
                env["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
                env["TRITON_CACHE_DIR"] = str(triton_dir)

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
                    print(f"[compare] subprocess failed (mode={mode_name}, trial={trial})")
                    if proc.stdout:
                        print(proc.stdout)
                    if proc.stderr:
                        print(proc.stderr)
                    raise RuntimeError("Comparison subprocess failed")

                metrics = extract_metrics_from_stdout(proc.stdout)
                metrics["subprocess_wall_seconds"] = wall_elapsed
                results_by_mode[mode_name].append(metrics)
                print(
                    f"[compare] mode={mode_name} trial={trial}: "
                    f"generation={metrics['generation_seconds']:.2f}s, "
                    f"script_total={metrics['total_script_seconds']:.2f}s, "
                    f"wall={wall_elapsed:.2f}s"
                )
            finally:
                shutil.rmtree(temp_root, ignore_errors=True)

    load_gen = [m["generation_seconds"] for m in results_by_mode["load"]]
    skip_gen = [m["generation_seconds"] for m in results_by_mode["skip"]]
    load_total = [m["total_script_seconds"] for m in results_by_mode["load"]]
    skip_total = [m["total_script_seconds"] for m in results_by_mode["skip"]]
    load_graphs = [m.get("unique_graphs", 0) for m in results_by_mode["load"]]
    skip_graphs = [m.get("unique_graphs", 0) for m in results_by_mode["skip"]]
    load_async_miss = [m.get("async_compile_miss", 0) for m in results_by_mode["load"]]
    skip_async_miss = [m.get("async_compile_miss", 0) for m in results_by_mode["skip"]]

    mean_load_gen = statistics.mean(load_gen)
    mean_skip_gen = statistics.mean(skip_gen)
    mean_load_total = statistics.mean(load_total)
    mean_skip_total = statistics.mean(skip_total)

    print("\n=== Comparison Summary ===")
    print(f"generation (load artifacts): {mean_load_gen:.2f}s avg over {len(load_gen)} run(s)")
    print(f"generation (skip artifacts): {mean_skip_gen:.2f}s avg over {len(skip_gen)} run(s)")
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
    print(
        f"avg unique_graphs (load artifacts): {statistics.mean(load_graphs):.2f}"
    )
    print(
        f"avg unique_graphs (skip artifacts): {statistics.mean(skip_graphs):.2f}"
    )
    print(
        f"avg async_compile_miss (load artifacts): {statistics.mean(load_async_miss):.2f}"
    )
    print(
        f"avg async_compile_miss (skip artifacts): {statistics.mean(skip_async_miss):.2f}"
    )


def main() -> None:
    args = parse_args()
    if args.compare_load_vs_skip:
        run_compare_mode(args)
        return
    run_single_inference(args)


if __name__ == "__main__":
    main()
