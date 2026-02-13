from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from aot_export_utils import (
    CausalLMLogitsWrapper,
    apply_transformers_masking_workaround,
    load_model_and_tokenizer,
    require_cuda,
    resolve_dynamic_seq_dim,
    resolve_dtype,
    runtime_fingerprint,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build AOTInductor deployment artifact for Llama 3.2 3B Instruct "
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
        help="Output .pt2 package path.",
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
        help="Export with dynamic sequence length dimension.",
    )
    parser.add_argument(
        "--no-dynamic-seq-len",
        dest="dynamic_seq_len",
        action="store_false",
        help="Export with static sequence length only.",
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
            "Use 1 to disable. Default 8 is a practical workaround for current "
            "torch.export+transformers constraints."
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
        help="Load .pt2 after build and run one correctness check.",
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


def main() -> None:
    args = parse_args()
    device = require_cuda()
    dtype, dtype_name = resolve_dtype(args.dtype, device)

    print(f"[build] loading model/tokenizer: {args.model_id}")
    model, tokenizer = load_model_and_tokenizer(args.model_id, device, dtype)
    wrapped = CausalLMLogitsWrapper(model).eval()

    encoded = tokenizer(
        args.example_prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.example_seq_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    dynamic_shapes: Any = None
    effective_min_seq_len = args.min_seq_len
    effective_max_seq_len = args.max_seq_len
    masking_workaround_applied = False
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
        dynamic_shapes = ({1: seq_dim}, {1: seq_dim})
        print(
            "[build] dynamic export enabled: "
            f"requested=[{args.min_seq_len}, {args.max_seq_len}], "
            f"effective=[{effective_min_seq_len}, {effective_max_seq_len}], "
            f"multiple={args.dynamic_seq_multiple}"
        )
    else:
        print("[build] static export enabled.")

    print(
        f"[build] exporting model (strict={args.strict_export}) "
        f"with example_shape={tuple(input_ids.shape)}"
    )
    export_start = time.perf_counter()
    exported_program = torch.export.export(
        wrapped,
        (input_ids, attention_mask),
        dynamic_shapes=dynamic_shapes,
        strict=args.strict_export,
    )
    export_seconds = time.perf_counter() - export_start
    print(f"[build] export completed in {export_seconds:.2f}s")

    inductor_configs = {"max_autotune": args.max_autotune}
    package_path = Path(args.package_path)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[build] compiling AOTInductor package -> {package_path} "
        f"with inductor_configs={inductor_configs}"
    )
    compile_start = time.perf_counter()
    built_package = torch._inductor.aoti_compile_and_package(
        exported_program,
        package_path=str(package_path),
        inductor_configs=inductor_configs,
    )
    compile_seconds = time.perf_counter() - compile_start
    package_size = package_path.stat().st_size
    print(f"[build] compile+package completed in {compile_seconds:.2f}s")
    print(f"[build] package created: {built_package} ({package_size} bytes)")

    verify_metrics: dict[str, Any] = {"enabled": args.verify_load}
    if args.verify_load:
        print("[build] verifying package load + numerical parity")
        verify_start = time.perf_counter()
        compiled_model = torch._inductor.aoti_load_package(str(package_path))
        with torch.inference_mode():
            ref = wrapped(input_ids, attention_mask)
            got = compiled_model(input_ids, attention_mask)
        torch.cuda.synchronize()
        verify_seconds = time.perf_counter() - verify_start
        max_abs_diff = (ref - got).abs().max().item()
        verify_metrics.update(
            {
                "seconds": verify_seconds,
                "max_abs_diff": float(max_abs_diff),
                "output_shape": list(got.shape),
            }
        )
        print(
            f"[build] verify completed in {verify_seconds:.2f}s, "
            f"max_abs_diff={max_abs_diff:.6f}"
        )

    metadata_path = Path(args.metadata_path)
    metadata = {
        "model_id": args.model_id,
        "package_path": str(package_path),
        "package_size_bytes": package_size,
        "dtype": dtype_name,
        "example": {
            "prompt": args.example_prompt,
            "seq_len": args.example_seq_len,
            "shape": list(input_ids.shape),
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
            "seconds": export_seconds,
        },
        "aot_compile": {
            "inductor_configs": inductor_configs,
            "seconds": compile_seconds,
        },
        "verify": verify_metrics,
        "runtime_fingerprint": runtime_fingerprint(device),
    }
    write_json(metadata_path, metadata)
    print(f"[build] wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
