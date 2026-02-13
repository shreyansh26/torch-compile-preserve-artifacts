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
)

BENCH_PREFIX = "BENCH_JSON:"
MODES = ("eager", "compile_preload", "compile_no_preload")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Llama inference across eager, compile+preload, and compile+no-preload "
            "with sequential requests in isolated subprocesses."
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
        help="Path to metadata JSON from build stage.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a short explanation of portable torch.compile caches.",
        help="Prompt for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation length per request.",
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
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0.0 uses greedy decoding.",
    )
    parser.add_argument(
        "--fixed-new-tokens",
        action="store_true",
        default=True,
        help=(
            "Set min_new_tokens=max_new_tokens so each request decodes fixed length. "
            "Recommended for stable compile benchmarking."
        ),
    )
    parser.add_argument(
        "--no-fixed-new-tokens",
        dest="fixed_new_tokens",
        action="store_false",
        help="Disable fixed-length decode; generation may stop early on EOS.",
    )
    parser.add_argument(
        "--cache-implementation",
        default="static",
        choices=["static", "hybrid", "sliding_window"],
        help="KV cache implementation used by generate().",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        help="CompileConfig.mode for compile modes.",
    )
    parser.add_argument(
        "--compile-dynamic",
        choices=["auto", "true", "false"],
        default="auto",
        help="CompileConfig.dynamic value for compile modes.",
    )
    parser.add_argument(
        "--fullgraph",
        dest="fullgraph",
        action="store_true",
        default=True,
        help="Use fullgraph=True for compile modes.",
    )
    parser.add_argument(
        "--no-fullgraph",
        dest="fullgraph",
        action="store_false",
        help="Use fullgraph=False for compile modes.",
    )
    parser.add_argument(
        "--bucket-pad",
        action="store_true",
        default=True,
        help="Pad prompt length to nearest warmed prefill bucket.",
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
    return args


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
    return [v for v in lengths if isinstance(v, int) and v > 0]


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


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def make_single_mode_metrics(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    overall_start = time.perf_counter()
    device = require_cuda()
    dtype, dtype_name = resolve_dtype(args.dtype, device)
    prefill_buckets = resolve_prefill_buckets(args)
    metadata = load_metadata(Path(args.metadata_path))
    build_decode_length = metadata.get("warmup", {}).get("decode_length")

    artifact_bytes = 0
    artifact_load_seconds = 0.0
    artifact_build_seconds = 0.0
    tokenizer_load_seconds = 0.0
    if mode == "compile_preload":
        cache_path = Path(args.cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache artifact file not found: {cache_path}")
        load_start = time.perf_counter()
        payload = cache_path.read_bytes()
        torch.compiler.load_cache_artifacts(payload)
        torch.cuda.synchronize()
        artifact_load_seconds = time.perf_counter() - load_start
        artifact_bytes = len(payload)
        print(
            f"[bench] mode={mode}: loaded portable artifacts "
            f"({artifact_bytes} bytes) in {artifact_load_seconds:.2f}s"
        )
    elif mode == "compile_no_preload":
        print("[bench] mode=compile_no_preload: skipping portable artifact load.")
    elif mode == "eager":
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
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
    if args.fixed_new_tokens:
        generation_kwargs["min_new_tokens"] = args.max_new_tokens

    if mode == "eager":
        generation_kwargs["disable_compile"] = True
    else:
        dynamic_map = {"auto": None, "true": True, "false": False}
        generation_kwargs["compile_config"] = CompileConfig(
            mode=args.compile_mode,
            fullgraph=args.fullgraph,
            dynamic=dynamic_map[args.compile_dynamic],
        )

    request_latencies: list[float] = []
    request_new_tokens: list[int] = []
    request_compile_stats: list[dict[str, int]] = []
    first_completion = ""

    for request_idx in range(args.num_requests):
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        counters.clear()
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generation_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        request_latencies.append(elapsed)
        new_tokens = int(output_ids.shape[1] - prompt_len)
        request_new_tokens.append(new_tokens)

        if request_idx == 0:
            first_completion = tokenizer.decode(
                output_ids[0][prompt_len:],
                skip_special_tokens=True,
            ).strip()

        stats = dict(counters.get("stats", {}))
        inductor_stats = dict(counters.get("inductor", {}))
        request_compile_stats.append(
            {
                "unique_graphs": int(stats.get("unique_graphs", 0)),
                "calls_captured": int(stats.get("calls_captured", 0)),
                "async_compile_miss": int(inductor_stats.get("async_compile_cache_miss", 0)),
                "async_compile_hit": int(inductor_stats.get("async_compile_cache_hit", 0)),
            }
        )
        print(
            f"[bench] mode={mode} request={request_idx + 1}/{args.num_requests} "
            f"latency={elapsed:.2f}s"
        )

    compile_activity_detected = any(
        (
            entry["unique_graphs"] > 0
            or entry["calls_captured"] > 0
            or entry["async_compile_miss"] > 0
            or entry["async_compile_hit"] > 0
        )
        for entry in request_compile_stats
    )
    if mode != "eager" and not compile_activity_detected:
        print(
            "[bench] warning: compile mode requested but no compile activity was observed "
            "(request_compile_stats were all zero)."
        )
    if (
        mode != "eager"
        and isinstance(build_decode_length, int)
        and build_decode_length > 0
        and args.max_new_tokens != build_decode_length
    ):
        print(
            "[bench] warning: max_new_tokens does not match build warmup decode_length "
            f"(runtime={args.max_new_tokens}, build={build_decode_length}); "
            "portable cache hit rate may be lower."
        )

    first_latency = request_latencies[0]
    tail = request_latencies[1:] if len(request_latencies) > 1 else request_latencies
    tail_new_tokens = request_new_tokens[1:] if len(request_new_tokens) > 1 else request_new_tokens
    total_generation_seconds = float(sum(request_latencies))
    total_script_seconds = time.perf_counter() - overall_start
    total_new_tokens = int(sum(request_new_tokens))
    tail_generation_seconds = float(sum(tail))
    tail_total_new_tokens = int(sum(tail_new_tokens))
    cold_start_first_request_seconds = float(
        tokenizer_load_seconds
        + model_load_seconds
        + artifact_build_seconds
        + artifact_load_seconds
        + first_latency
    )
    summary = {
        "mode": mode,
        "num_requests": args.num_requests,
        "mode_notes": (
            "portable cache preload" if mode == "compile_preload"
            else "portable cache not preloaded" if mode == "compile_no_preload"
            else "hf eager"
        ),
        "artifact_bytes": artifact_bytes,
        "artifact_build_seconds": artifact_build_seconds,
        "artifact_load_seconds": artifact_load_seconds,
        # load_model_and_tokenizer loads both together in this benchmark.
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_load_seconds": model_load_seconds,
        "request_latencies": request_latencies,
        "request_new_tokens": request_new_tokens,
        "first_request_seconds": first_latency,
        "ttft_after_ready_seconds": first_latency,
        "tail_mean_seconds": statistics.mean(tail),
        "mean_request_seconds": statistics.mean(request_latencies),
        "median_request_seconds": median(request_latencies),
        "p90_request_seconds": percentile(request_latencies, 0.90),
        "total_generation_seconds": total_generation_seconds,
        "total_script_seconds": total_script_seconds,
        "tokens_per_second": safe_tps(total_new_tokens, total_generation_seconds),
        "tokens_per_second_all": safe_tps(total_new_tokens, total_generation_seconds),
        "tail_tokens_per_second": safe_tps(tail_total_new_tokens, tail_generation_seconds),
        "steady_state_tokens_per_second": safe_tps(
            tail_total_new_tokens,
            tail_generation_seconds,
        ),
        "end_to_end_tokens_per_second": safe_tps(total_new_tokens, total_script_seconds),
        "ttft_cold_start_seconds": cold_start_first_request_seconds,
        "cold_start_first_request_seconds": cold_start_first_request_seconds,
        "prompt_tokens_original": original_prompt_tokens,
        "prompt_tokens_effective": effective_prompt_tokens,
        "total_new_tokens": total_new_tokens,
        "fixed_new_tokens": bool(args.fixed_new_tokens),
        "build_decode_length": build_decode_length,
        "compile_activity_detected": bool(compile_activity_detected),
        "dtype": dtype_name,
        "compile_mode": args.compile_mode,
        "fullgraph": bool(args.fullgraph),
        "cache_implementation": args.cache_implementation,
        "bucket_pad": bool(args.bucket_pad),
        "request_compile_stats": request_compile_stats,
        "first_completion": first_completion,
    }
    return summary


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
        "--cache-path",
        args.cache_path,
        "--metadata-path",
        args.metadata_path,
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--num-requests",
        str(args.num_requests),
        "--dtype",
        args.dtype,
        "--temperature",
        str(args.temperature),
        "--cache-implementation",
        args.cache_implementation,
        "--compile-mode",
        args.compile_mode,
        "--compile-dynamic",
        args.compile_dynamic,
        "--prefill-buckets",
        args.prefill_buckets,
    ]
    cmd.append("--bucket-pad" if args.bucket_pad else "--no-bucket-pad")
    cmd.append("--fullgraph" if args.fullgraph else "--no-fullgraph")
    cmd.append("--fixed-new-tokens" if args.fixed_new_tokens else "--no-fixed-new-tokens")
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
                    prefix=f"bench_{mode}_{repeat}_",
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
        first_avg = statistics.mean([r["ttft_after_ready_seconds"] for r in mode_results])
        cold_first_avg = statistics.mean([r["ttft_cold_start_seconds"] for r in mode_results])
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

    compile_preload_cold = statistics.mean(
        [r["ttft_cold_start_seconds"] for r in all_results["compile_preload"]]
    )
    compile_noload_cold = statistics.mean(
        [r["ttft_cold_start_seconds"] for r in all_results["compile_no_preload"]]
    )
    eager_cold = statistics.mean([r["ttft_cold_start_seconds"] for r in all_results["eager"]])
    print(
        "compile_preload cold-start speedup vs compile_no_preload: "
        f"{compile_noload_cold / compile_preload_cold:.2f}x"
        if compile_preload_cold > 0
        else "compile_preload cold-start speedup vs compile_no_preload: n/a"
    )
    print(
        "compile_preload cold-start speedup vs eager: "
        f"{eager_cold / compile_preload_cold:.2f}x"
        if compile_preload_cold > 0
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
