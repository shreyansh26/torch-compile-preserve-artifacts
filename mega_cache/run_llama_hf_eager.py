from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from llama_compile_cache_utils import (
    load_model_and_tokenizer,
    parse_positive_int_csv,
    require_cuda,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Llama HF inference in eager mode (no torch.compile, no compile_config)."
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model id.",
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
        help="Generation length.",
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
        "--cache-implementation",
        default="static",
        choices=["static", "hybrid", "sliding_window"],
        help="KV cache implementation used by generate().",
    )
    parser.add_argument(
        "--bucket-pad",
        action="store_true",
        help="Pad prompt to nearest prefill bucket.",
    )
    parser.add_argument(
        "--prefill-buckets",
        default="",
        help="Comma-separated prefill buckets for --bucket-pad.",
    )
    parser.add_argument(
        "--metadata-path",
        default="artifacts/llama3b_torchcompile_cache_meta.json",
        help="Metadata file to auto-load prefill buckets if --prefill-buckets is empty.",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be a positive integer")
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
        print(
            "[eager] --bucket-pad requested, but no prefill buckets were resolved; "
            "using unpadded prompt."
        )
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
        "[eager] bucket padding enabled: "
        f"prompt_tokens={original_prompt_tokens} -> padded_tokens={target}"
    )
    return padded, original_prompt_tokens, target


def main() -> None:
    args = parse_args()
    overall_start = time.perf_counter()

    device = require_cuda()
    dtype, dtype_name = resolve_dtype(args.dtype, device)
    print(f"[eager] loading model/tokenizer: {args.model_id} (dtype={dtype_name})")
    model_load_start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(args.model_id, device, dtype)
    model_load_seconds = time.perf_counter() - model_load_start

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
        "disable_compile": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature

    print("[eager] running generation")
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
    print(f"\n[eager] end-to-end generation latency: {elapsed:.2f}s")
    print(f"[eager] model load latency: {model_load_seconds:.2f}s")
    print(f"[eager] total script latency: {time.perf_counter() - overall_start:.2f}s")
    print(
        "[eager] token info: "
        f"prompt_original={original_prompt_tokens}, "
        f"prompt_effective={effective_prompt_tokens}, "
        f"new_tokens={int(output_ids.shape[1] - prompt_len)}"
    )


if __name__ == "__main__":
    main()
