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
    CausalLMLogitsWrapper,
    load_model_and_tokenizer,
    parse_positive_int_csv,
    require_cuda,
    resolve_dtype,
)

BENCH_PREFIX = "BENCH_JSON:"
MODES = ("compile_preload", "compile_no_preload", "eager")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Llama inference across compile+preload (.pt2 load), "
            "compile+no-preload (build .pt2 at runtime), and eager mode."
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
        help="Path to prebuilt AOTInductor package for compile_preload mode.",
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
        default=True,
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
        help=(
            "For compile_no_preload mode, export with dynamic sequence length "
            "instead of static sequence length."
        ),
    )
    parser.add_argument(
        "--example-seq-len",
        type=int,
        default=128,
        help="Fallback static export sequence length when metadata does not exist.",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=1,
        help="Minimum sequence length guard for dynamic export in compile_no_preload mode.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length guard for dynamic export in compile_no_preload mode.",
    )
    parser.add_argument(
        "--max-autotune",
        action="store_true",
        default=True,
        help="Set inductor_configs['max_autotune']=True in compile_no_preload mode.",
    )
    parser.add_argument(
        "--no-max-autotune",
        dest="max_autotune",
        action="store_false",
        help="Disable max_autotune in compile_no_preload mode.",
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


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def run_aot_request(
    compiled_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    stop_on_eos: bool,
    eos_token_id: int | None,
) -> tuple[list[int], float]:
    generated_ids: list[int] = []
    req_input_ids = input_ids.clone()
    req_attention_mask = attention_mask.clone()

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            logits = compiled_model(req_input_ids, req_attention_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_id = int(next_token.item())
            generated_ids.append(next_id)

            req_input_ids = torch.cat([req_input_ids, next_token], dim=1)
            next_mask = torch.ones(
                (req_attention_mask.shape[0], 1),
                dtype=req_attention_mask.dtype,
                device=req_attention_mask.device,
            )
            req_attention_mask = torch.cat([req_attention_mask, next_mask], dim=1)

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
            "Use --compile-dynamic-seq-len (compile_no_preload) or prebuild a dynamic package."
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
        temp_dir_handle: tempfile.TemporaryDirectory[str] | None = None
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

        dynamic_seq_len = False
        expected_seq_len: int | None = None
        if mode == "compile_preload":
            package_path = Path(args.package_path)
            if not package_path.exists():
                raise FileNotFoundError(
                    f"Prebuilt package not found: {package_path}. Run ./build.sh first."
                )
            dynamic_seq_len = bool(metadata.get("export", {}).get("dynamic_seq_len", False))
            maybe_expected = metadata.get("example", {}).get("seq_len")
            if isinstance(maybe_expected, int) and maybe_expected > 0:
                expected_seq_len = maybe_expected
            ensure_aot_runtime_compatible(
                mode=mode,
                prompt_seq_len=int(input_ids.shape[1]),
                dynamic_seq_len=dynamic_seq_len,
                expected_seq_len=expected_seq_len,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"[bench] mode=compile_preload: loading package {package_path}")
            load_start = time.perf_counter()
            compiled_model = torch._inductor.aoti_load_package(str(package_path))
            torch.cuda.synchronize()
            artifact_load_seconds = time.perf_counter() - load_start
            mode_notes = "prebuilt .pt2 load"
        else:
            dynamic_seq_len = bool(args.compile_dynamic_seq_len)
            compile_seq_len = int(input_ids.shape[1]) if not dynamic_seq_len else max(
                args.min_seq_len,
                min(args.max_seq_len, int(input_ids.shape[1])),
            )
            expected_seq_len = None if dynamic_seq_len else compile_seq_len
            ensure_aot_runtime_compatible(
                mode=mode,
                prompt_seq_len=int(input_ids.shape[1]),
                dynamic_seq_len=dynamic_seq_len,
                expected_seq_len=expected_seq_len,
                max_new_tokens=args.max_new_tokens,
            )
            model_load_start = time.perf_counter()
            model, compile_tokenizer = load_model_and_tokenizer(args.model_id, device, dtype)
            model_load_seconds = time.perf_counter() - model_load_start
            wrapped = CausalLMLogitsWrapper(model).eval()

            compile_encoded = compile_tokenizer(
                args.prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=compile_seq_len,
            )
            compile_input_ids = compile_encoded["input_ids"].to(device)
            compile_attention_mask = compile_encoded["attention_mask"].to(device)

            dynamic_shapes = None
            if dynamic_seq_len:
                seq_dim = torch.export.Dim(
                    "seq_len",
                    min=args.min_seq_len,
                    max=args.max_seq_len,
                )
                dynamic_shapes = ({1: seq_dim}, {1: seq_dim})

            temp_dir_handle = tempfile.TemporaryDirectory(prefix="aot_compile_no_preload_")
            package_path = Path(temp_dir_handle.name) / "llama3b_aotinductor.pt2"
            inductor_configs = {"max_autotune": args.max_autotune}
            print(
                "[bench] mode=compile_no_preload: exporting+compiling package "
                f"(dynamic_seq_len={dynamic_seq_len})"
            )
            build_start = time.perf_counter()
            exported_program = torch.export.export(
                wrapped,
                (compile_input_ids, compile_attention_mask),
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
            torch._inductor.aoti_compile_and_package(
                exported_program,
                package_path=str(package_path),
                inductor_configs=inductor_configs,
            )
            torch.cuda.synchronize()
            artifact_build_seconds = time.perf_counter() - build_start
            mode_notes = (
                "runtime export+compile"
                f" (dynamic_seq_len={dynamic_seq_len}, max_autotune={args.max_autotune})"
            )
            del exported_program
            del wrapped
            del model
            torch.cuda.empty_cache()

            load_start = time.perf_counter()
            compiled_model = torch._inductor.aoti_load_package(str(package_path))
            torch.cuda.synchronize()
            artifact_load_seconds = time.perf_counter() - load_start

        try:
            eos_token_id = tokenizer.eos_token_id
            for request_idx in range(args.num_requests):
                generated_ids, elapsed = run_aot_request(
                    compiled_model=compiled_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    stop_on_eos=args.stop_on_eos,
                    eos_token_id=eos_token_id,
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
        finally:
            if temp_dir_handle is not None:
                temp_dir_handle.cleanup()

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
        # TTFT measured after model/package is already available in-process.
        "ttft_after_ready_seconds": first_latency,
        "tail_mean_seconds": float(statistics.mean(tail_latencies)),
        "mean_request_seconds": mean_request_seconds,
        "median_request_seconds": median(request_latencies),
        "p90_request_seconds": percentile(request_latencies, 0.90),
        "total_generation_seconds": total_generation_seconds,
        "total_script_seconds": total_script_seconds,
        # Includes first request (captures first-token startup effects).
        "tokens_per_second": safe_tps(total_new_tokens, total_generation_seconds),
        "tokens_per_second_all": safe_tps(total_new_tokens, total_generation_seconds),
        # Excludes first request to show steady-state throughput.
        "tail_tokens_per_second": safe_tps(tail_tokens, tail_generation_seconds),
        "steady_state_tokens_per_second": safe_tps(tail_tokens, tail_generation_seconds),
        # Includes compile/load/model startup and all requests.
        "end_to_end_tokens_per_second": safe_tps(total_new_tokens, total_script_seconds),
        # TTFT from process start (includes load/compile work).
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
    ]
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

    for mode in MODES:
        for repeat in range(1, args.repeats + 1):
            temp_root = Path(tempfile.mkdtemp(prefix=f"aot_bench_{mode}_{repeat}_"))
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
