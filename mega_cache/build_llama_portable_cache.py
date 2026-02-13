from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers.generation.configuration_utils import CompileConfig

from llama_compile_cache_utils import (
    load_model_and_tokenizer,
    make_synthetic_inputs,
    parse_positive_int_csv,
    require_cuda,
    resolve_dtype,
    runtime_fingerprint,
    to_jsonable_cache_info,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline compile/warmup job for Llama 3.2 3B Instruct that exports "
            "portable torch.compile cache artifacts."
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
        help="Output path for serialized portable cache artifacts.",
    )
    parser.add_argument(
        "--metadata-path",
        default="artifacts/llama3b_torchcompile_cache_meta.json",
        help="Output path for metadata JSON.",
    )
    parser.add_argument(
        "--prefill-lengths",
        default="128,256,512,1024",
        help="Comma-separated input token lengths to compile/warm.",
    )
    parser.add_argument(
        "--decode-length",
        type=int,
        default=128,
        help="Number of generated tokens per warmup bucket.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used during warmup.",
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
        help="KV cache implementation used during warmup generation.",
    )
    parser.add_argument(
        "--compile-dynamic",
        choices=["auto", "true", "false"],
        default="auto",
        help="CompileConfig.dynamic value for generation auto-compile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic warmup inputs.",
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
    args = parser.parse_args()
    if args.decode_length <= 0:
        raise ValueError("--decode-length must be a positive integer")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    prefill_lengths = parse_positive_int_csv(args.prefill_lengths)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = require_cuda()
    dtype, dtype_name = resolve_dtype(args.dtype, device)

    print(f"[build] loading model/tokenizer: {args.model_id}")
    model, tokenizer = load_model_and_tokenizer(args.model_id, device, dtype)

    compile_dynamic_map = {"auto": None, "true": True, "false": False}
    compile_dynamic = compile_dynamic_map[args.compile_dynamic]
    compile_config = CompileConfig(
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
        dynamic=compile_dynamic,
    )
    print(
        "[build] generate auto-compile settings: "
        f"mode={args.compile_mode}, fullgraph={args.fullgraph}, "
        f"dynamic={compile_dynamic}, cache_implementation={args.cache_implementation}, "
        f"dtype={dtype_name}"
    )

    if tokenizer.pad_token_id is None:
        # Fallback to eos token if pad token is still missing.
        pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")

    print(
        f"[build] warming compile cache for prefill buckets={prefill_lengths}, "
        f"decode_length={args.decode_length}, batch_size={args.batch_size}, "
        f"cache_implementation={args.cache_implementation}"
    )
    warmup_timings: list[dict[str, Any]] = []
    with torch.inference_mode():
        for seq_len in prefill_lengths:
            bucket_inputs = make_synthetic_inputs(
                batch_size=args.batch_size,
                seq_len=seq_len,
                vocab_size=model.config.vocab_size,
                device=device,
            )

            start = time.perf_counter()
            _ = model.generate(
                **bucket_inputs,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                cache_implementation=args.cache_implementation,
                compile_config=compile_config,
                max_new_tokens=args.decode_length,
                min_new_tokens=args.decode_length,
                pad_token_id=pad_token_id,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            warmup_timings.append(
                {
                    "stage": "generate_auto_compile",
                    "prefill_length": seq_len,
                    "seconds": elapsed,
                }
            )
            print(
                f"[build] generate warmup seq_len={seq_len} done in {elapsed:.2f}s"
            )

    saved = torch.compiler.save_cache_artifacts()
    if saved is None:
        raise RuntimeError(
            "No cache artifacts were produced. "
            "Ensure warmup ran through compiled code paths."
        )

    artifact_bytes, cache_info = saved
    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(artifact_bytes)

    metadata_path = Path(args.metadata_path)
    metadata = {
        "model_id": args.model_id,
        "compile": {
            "mode": args.compile_mode,
            "fullgraph": args.fullgraph,
            "dynamic": compile_dynamic,
            "dtype": dtype_name,
            "cache_implementation": args.cache_implementation,
        },
        "warmup": {
            "prefill_lengths": prefill_lengths,
            "decode_length": args.decode_length,
            "batch_size": args.batch_size,
            "timings": warmup_timings,
        },
        "runtime_fingerprint": runtime_fingerprint(device),
        "cache_info": to_jsonable_cache_info(cache_info),
        "cache_bytes": len(artifact_bytes),
    }
    write_json(metadata_path, metadata)

    print(f"[build] wrote cache artifacts: {cache_path} ({len(artifact_bytes)} bytes)")
    print(f"[build] wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
