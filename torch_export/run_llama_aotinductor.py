from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from aot_export_utils import parse_positive_int_csv, require_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Llama 3.2 3B Instruct inference from AOTInductor .pt2 package "
            "(torch._inductor.aoti_load_package)."
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
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    return args


def resolve_prefill_buckets(args: argparse.Namespace) -> list[int]:
    if args.prefill_buckets.strip():
        return parse_positive_int_csv(args.prefill_buckets)

    metadata_path = Path(args.metadata_path)
    if not metadata_path.exists():
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    example_seq_len = metadata.get("example", {}).get("seq_len")
    if isinstance(example_seq_len, int) and example_seq_len > 0:
        return [example_seq_len]
    return []


def load_metadata(path: Path) -> dict:
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


def main() -> None:
    args = parse_args()
    device = require_cuda()
    package_path = Path(args.package_path)
    if not package_path.exists():
        raise FileNotFoundError(
            f"AOTInductor package not found: {package_path}. "
            "Run build_llama_aotinductor.py first."
        )

    print(f"[run] loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    metadata_path = Path(args.metadata_path)
    metadata = load_metadata(metadata_path)
    prefill_buckets = resolve_prefill_buckets(args)
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
    load_seconds = time.perf_counter() - load_start
    print(f"[run] package loaded in {load_seconds:.2f}s")

    export_meta = metadata.get("export", {})
    dynamic_seq_len = bool(export_meta.get("dynamic_seq_len", False))
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

    eos_token_id = tokenizer.eos_token_id
    generated_ids: list[int] = []

    print(
        "[run] generating with AOT package "
        f"(naive autoregressive loop, max_new_tokens={args.max_new_tokens})"
    )
    gen_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            logits = compiled_model(input_ids, attention_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_id = int(next_token.item())
            generated_ids.append(next_id)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            next_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, next_mask], dim=1)

            if args.stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                break
    torch.cuda.synchronize()
    gen_seconds = time.perf_counter() - gen_start

    completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Completion ===")
    print(completion)
    print(f"\n[run] package load latency: {load_seconds:.2f}s")
    print(f"[run] generation latency: {gen_seconds:.2f}s")
    print(
        "[run] token info: "
        f"prompt_original={original_tokens}, "
        f"prompt_effective={effective_tokens}, "
        f"new_tokens={len(generated_ids)}"
    )


if __name__ == "__main__":
    main()
