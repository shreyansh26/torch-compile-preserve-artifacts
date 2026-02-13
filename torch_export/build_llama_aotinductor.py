from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from aot_export_utils import (
    CausalLMDecodeWrapper,
    CausalLMPrefillWrapper,
    apply_transformers_masking_workaround,
    load_model_and_tokenizer,
    resolve_dynamic_seq_dim,
    resolve_dtype,
    resolve_num_hidden_layers,
    require_cuda,
    runtime_fingerprint,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build AOTInductor deployment artifacts for KV-cache inference "
            "(torch.export -> aoti_compile_and_package -> .pt2)."
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
        help="Output prefill .pt2 package path.",
    )
    parser.add_argument(
        "--decode-package-path",
        default="",
        help=(
            "Output decode .pt2 package path. "
            "Default derives from --package-path by appending _decode."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        default="artifacts/llama3b_aotinductor_meta.json",
        help="Output metadata JSON path.",
    )
    parser.add_argument(
        "--example-prompt",
        default="Write a short explanation of portable torch.compile caches.",
        help="Prompt used to create example export inputs.",
    )
    parser.add_argument(
        "--example-seq-len",
        type=int,
        default=128,
        help="Example sequence length for export (tokenizer pads/truncates to this).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Model dtype on GPU.",
    )
    parser.add_argument(
        "--dynamic-seq-len",
        action="store_true",
        default=False,
        help="Export with dynamic prompt and cache sequence dimensions.",
    )
    parser.add_argument(
        "--no-dynamic-seq-len",
        dest="dynamic_seq_len",
        action="store_false",
        help="Export with static sequence lengths.",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=1,
        help="Minimum sequence length guard for dynamic export.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length guard for dynamic export.",
    )
    parser.add_argument(
        "--dynamic-seq-multiple",
        type=int,
        default=8,
        help=(
            "Dynamic sequence lengths must be multiples of this value. "
            "Use 1 to disable."
        ),
    )
    parser.add_argument(
        "--masking-workaround",
        action="store_true",
        default=True,
        help=(
            "Apply transformers masking workaround for dynamic export "
            "(recommended on torch==2.9.1 / transformers==4.57.x)."
        ),
    )
    parser.add_argument(
        "--no-masking-workaround",
        dest="masking_workaround",
        action="store_false",
        help="Disable transformers masking workaround.",
    )
    parser.add_argument(
        "--max-autotune",
        action="store_true",
        default=True,
        help="Set inductor_configs['max_autotune']=True.",
    )
    parser.add_argument(
        "--no-max-autotune",
        dest="max_autotune",
        action="store_false",
        help="Disable max_autotune.",
    )
    parser.add_argument(
        "--strict-export",
        action="store_true",
        default=False,
        help="Use strict=True for torch.export.export (default strict=False).",
    )
    parser.add_argument(
        "--verify-load",
        action="store_true",
        default=True,
        help="Load .pt2 files after build and run one correctness check.",
    )
    parser.add_argument(
        "--no-verify-load",
        dest="verify_load",
        action="store_false",
        help="Skip post-build load/correctness check.",
    )
    args = parser.parse_args()
    if args.example_seq_len <= 0:
        raise ValueError("--example-seq-len must be positive")
    if args.min_seq_len <= 0:
        raise ValueError("--min-seq-len must be positive")
    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive")
    if args.max_seq_len < args.min_seq_len:
        raise ValueError("--max-seq-len must be >= --min-seq-len")
    if args.dynamic_seq_multiple <= 0:
        raise ValueError("--dynamic-seq-multiple must be positive")
    return args


def derive_decode_package_path(prefill_path: Path) -> Path:
    suffix = prefill_path.suffix or ".pt2"
    stem = prefill_path.stem
    return prefill_path.with_name(f"{stem}_decode{suffix}")


def split_prefill_outputs(outputs: Any) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    logits = outputs[0]
    cache = tuple(outputs[1:])
    return logits, cache


def main() -> None:
    args = parse_args()
    device = require_cuda()
    dtype, dtype_name = resolve_dtype(args.dtype, device)

    print(f"[build] loading model/tokenizer: {args.model_id}")
    model, tokenizer = load_model_and_tokenizer(args.model_id, device, dtype)
    num_hidden_layers = resolve_num_hidden_layers(model)
    prefill_wrapper = CausalLMPrefillWrapper(model).eval()
    decode_wrapper = CausalLMDecodeWrapper(model, num_hidden_layers=num_hidden_layers).eval()

    encoded = tokenizer(
        args.example_prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.example_seq_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    dynamic_shapes_prefill: Any = None
    dynamic_shapes_decode: Any = None
    effective_min_seq_len = args.min_seq_len
    effective_max_seq_len = args.max_seq_len
    masking_workaround_applied = False
    seq_dim: Any = None

    if args.dynamic_seq_len:
        if args.masking_workaround:
            masking_workaround_applied = apply_transformers_masking_workaround()
            print(
                "[build] masking workaround "
                f"{'applied' if masking_workaround_applied else 'not applied'}"
            )
        seq_dim, effective_min_seq_len, effective_max_seq_len = resolve_dynamic_seq_dim(
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            seq_multiple=args.dynamic_seq_multiple,
        )
        dynamic_shapes_prefill = ({1: seq_dim}, {1: seq_dim})
        print(
            "[build] dynamic export enabled: "
            f"requested=[{args.min_seq_len}, {args.max_seq_len}], "
            f"effective=[{effective_min_seq_len}, {effective_max_seq_len}], "
            f"multiple={args.dynamic_seq_multiple}"
        )
    else:
        print("[build] static export enabled.")

    # Build decode example from eager prefill output so export sees realistic KV tensors.
    with torch.inference_mode():
        eager_prefill_outputs = prefill_wrapper(input_ids, attention_mask)
    eager_prefill_logits, eager_prefill_cache = split_prefill_outputs(eager_prefill_outputs)
    expected_cache_tensors = num_hidden_layers * 2
    if len(eager_prefill_cache) != expected_cache_tensors:
        raise RuntimeError(
            "Unexpected KV tensor count from prefill wrapper: "
            f"expected={expected_cache_tensors}, got={len(eager_prefill_cache)}"
        )
    decode_input_ids = eager_prefill_logits[:, input_ids.shape[1] - 1, :].argmax(
        dim=-1, keepdim=True
    )
    decode_attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ],
        dim=1,
    )
    with torch.inference_mode():
        eager_decode_outputs = decode_wrapper(
            decode_input_ids, decode_attention_mask, *eager_prefill_cache
        )
    eager_decode_logits = eager_decode_outputs[0]
    verify_ref_prefill_logits_cpu = (
        eager_prefill_logits.detach().float().cpu() if args.verify_load else None
    )
    verify_ref_decode_logits_cpu = (
        eager_decode_logits.detach().float().cpu() if args.verify_load else None
    )
    if args.dynamic_seq_len:
        decode_seq_dim = torch.export.Dim(
            "decode_seq_len",
            min=args.min_seq_len,
            max=args.max_seq_len,
        )
        decode_attention_dim = decode_seq_dim + 1
        dynamic_shapes_decode = (
            None,
            {1: decode_attention_dim},
            tuple({2: decode_seq_dim} for _ in eager_prefill_cache),
        )

    print(
        f"[build] exporting prefill wrapper (strict={args.strict_export}) "
        f"with example_shape={tuple(input_ids.shape)}"
    )
    prefill_export_start = time.perf_counter()
    exported_prefill = torch.export.export(
        prefill_wrapper,
        (input_ids, attention_mask),
        dynamic_shapes=dynamic_shapes_prefill,
        strict=args.strict_export,
    )
    prefill_export_seconds = time.perf_counter() - prefill_export_start
    print(f"[build] prefill export completed in {prefill_export_seconds:.2f}s")

    print(
        f"[build] exporting decode wrapper (strict={args.strict_export}) "
        f"with example_shape={(tuple(decode_input_ids.shape), tuple(decode_attention_mask.shape))}"
    )
    decode_export_start = time.perf_counter()
    exported_decode = torch.export.export(
        decode_wrapper,
        (decode_input_ids, decode_attention_mask, *eager_prefill_cache),
        dynamic_shapes=dynamic_shapes_decode,
        strict=args.strict_export,
    )
    decode_export_seconds = time.perf_counter() - decode_export_start
    print(f"[build] decode export completed in {decode_export_seconds:.2f}s")

    inductor_configs = {"max_autotune": args.max_autotune}
    prefill_package_path = Path(args.package_path)
    decode_package_path = (
        Path(args.decode_package_path)
        if args.decode_package_path.strip()
        else derive_decode_package_path(prefill_package_path)
    )
    prefill_package_path.parent.mkdir(parents=True, exist_ok=True)
    decode_package_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[build] compiling prefill package -> {prefill_package_path} "
        f"with inductor_configs={inductor_configs}"
    )
    prefill_compile_start = time.perf_counter()
    built_prefill_package = torch._inductor.aoti_compile_and_package(
        exported_prefill,
        package_path=str(prefill_package_path),
        inductor_configs=inductor_configs,
    )
    prefill_compile_seconds = time.perf_counter() - prefill_compile_start
    prefill_package_size = prefill_package_path.stat().st_size
    print(f"[build] prefill compile+package completed in {prefill_compile_seconds:.2f}s")
    print(f"[build] prefill package created: {built_prefill_package} ({prefill_package_size} bytes)")
    del exported_prefill
    torch.cuda.empty_cache()

    print(
        f"[build] compiling decode package -> {decode_package_path} "
        f"with inductor_configs={inductor_configs}"
    )
    decode_compile_start = time.perf_counter()
    built_decode_package = torch._inductor.aoti_compile_and_package(
        exported_decode,
        package_path=str(decode_package_path),
        inductor_configs=inductor_configs,
    )
    decode_compile_seconds = time.perf_counter() - decode_compile_start
    decode_package_size = decode_package_path.stat().st_size
    print(f"[build] decode compile+package completed in {decode_compile_seconds:.2f}s")
    print(f"[build] decode package created: {built_decode_package} ({decode_package_size} bytes)")
    del exported_decode
    torch.cuda.empty_cache()

    verify_metrics: dict[str, Any] = {"enabled": args.verify_load}
    if args.verify_load:
        print("[build] verifying package load + numerical parity")
        verify_start = time.perf_counter()
        # Reduce peak memory during verification: free eager model state first.
        model.to("cpu")
        del model
        del prefill_wrapper
        del decode_wrapper
        del eager_prefill_outputs
        del eager_decode_outputs
        torch.cuda.empty_cache()

        compiled_prefill = torch._inductor.aoti_load_package(str(prefill_package_path))
        with torch.inference_mode():
            got_prefill = compiled_prefill(input_ids, attention_mask)
            got_prefill_logits, got_prefill_cache = split_prefill_outputs(got_prefill)
        del compiled_prefill
        torch.cuda.empty_cache()

        compiled_decode = torch._inductor.aoti_load_package(str(decode_package_path))
        with torch.inference_mode():
            got_decode = compiled_decode(
                decode_input_ids, decode_attention_mask, *got_prefill_cache
            )
            got_decode_logits = got_decode[0]
        torch.cuda.synchronize()
        verify_seconds = time.perf_counter() - verify_start

        prefill_max_abs_diff = (
            verify_ref_prefill_logits_cpu - got_prefill_logits.detach().float().cpu()
        ).abs().max().item()
        decode_max_abs_diff = (
            verify_ref_decode_logits_cpu - got_decode_logits.detach().float().cpu()
        ).abs().max().item()
        verify_metrics.update(
            {
                "seconds": verify_seconds,
                "prefill_max_abs_diff": float(prefill_max_abs_diff),
                "decode_max_abs_diff": float(decode_max_abs_diff),
                "prefill_output_shape": list(got_prefill_logits.shape),
                "decode_output_shape": list(got_decode_logits.shape),
            }
        )
        print(
            f"[build] verify completed in {verify_seconds:.2f}s, "
            f"prefill_max_abs_diff={prefill_max_abs_diff:.6f}, "
            f"decode_max_abs_diff={decode_max_abs_diff:.6f}"
        )

    metadata_path = Path(args.metadata_path)
    metadata = {
        "model_id": args.model_id,
        "package_path": str(prefill_package_path),
        "prefill_package_path": str(prefill_package_path),
        "decode_package_path": str(decode_package_path),
        "prefill_package_size_bytes": prefill_package_size,
        "decode_package_size_bytes": decode_package_size,
        "package_size_bytes": prefill_package_size + decode_package_size,
        "dtype": dtype_name,
        "example": {
            "prompt": args.example_prompt,
            "seq_len": args.example_seq_len,
            "shape": list(input_ids.shape),
            "decode_shape": {
                "input_ids": list(decode_input_ids.shape),
                "attention_mask": list(decode_attention_mask.shape),
            },
        },
        "export": {
            "strict": args.strict_export,
            "dynamic_seq_len": args.dynamic_seq_len,
            "min_seq_len": args.min_seq_len,
            "max_seq_len": args.max_seq_len,
            "effective_min_seq_len": effective_min_seq_len,
            "effective_max_seq_len": effective_max_seq_len,
            "dynamic_seq_multiple": args.dynamic_seq_multiple,
            "masking_workaround_requested": args.masking_workaround,
            "masking_workaround_applied": masking_workaround_applied,
            "prefill_seconds": prefill_export_seconds,
            "decode_seconds": decode_export_seconds,
            "seconds": prefill_export_seconds + decode_export_seconds,
        },
        "aot_compile": {
            "inductor_configs": inductor_configs,
            "prefill_seconds": prefill_compile_seconds,
            "decode_seconds": decode_compile_seconds,
            "seconds": prefill_compile_seconds + decode_compile_seconds,
        },
        "kv_cache": {
            "enabled": True,
            "num_hidden_layers": num_hidden_layers,
            "num_cache_tensors": expected_cache_tensors,
        },
        "verify": verify_metrics,
        "runtime_fingerprint": runtime_fingerprint(device),
    }
    write_json(metadata_path, metadata)
    print(f"[build] wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
