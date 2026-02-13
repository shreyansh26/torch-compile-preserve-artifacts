# Llama 3.2 3B AOTInductor Artifact (`torch.export` -> `.pt2`)

This folder implements a true ahead-of-time deployment artifact flow:

1. `torch.export.export(...)` to capture an `ExportedProgram`
2. `torch._inductor.aoti_compile_and_package(...)` to create a `.pt2` package
3. `torch._inductor.aoti_load_package(...)` for runtime inference

## Environment

- `torch==2.9.1`
- `transformers==4.57.5`

## 1) Build AOT Artifact

```bash
./build.sh
```

Default output:

- `artifacts/llama3b_aotinductor.pt2`
- `artifacts/llama3b_aotinductor_meta.json`

Equivalent direct command:

```bash
python build_llama_aotinductor.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --package-path artifacts/llama3b_aotinductor.pt2 \
  --metadata-path artifacts/llama3b_aotinductor_meta.json \
  --example-seq-len 128 \
  --no-dynamic-seq-len \
  --max-autotune
```

## 2) Run From `.pt2` Package

```bash
./run.sh
```

Equivalent direct command:

```bash
python run_llama_aotinductor.py \
  --package-path artifacts/llama3b_aotinductor.pt2 \
  --metadata-path artifacts/llama3b_aotinductor_meta.json \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --bucket-pad \
  --prompt "Write a short explanation of portable torch.compile caches." \
  --max-new-tokens 1
```

## Notes

- The exported wrapper compiles `forward(input_ids, attention_mask) -> logits`.
- Runtime generation here uses a simple greedy autoregressive loop (no KV cache), intended as a clean artifact-loading example.
- Default build is static-sequence export (`--no-dynamic-seq-len`) because dynamic-sequence export may fail for some HF models/versions.
- For static exports, use `--max-new-tokens 1` unless you rebuild with `--dynamic-seq-len`.
- Since this is `torch.export`-based, model/export graph compatibility matters more than `torch.compile` mega-cache flow.
- Keep PyTorch/CUDA/GPU stack compatible between build and runtime.

## Dynamic Sequence Dimensions

Yes, you can define dynamic sequence length for `torch.export`:

```bash
./build.sh \
  --dynamic-seq-len \
  --min-seq-len 1 \
  --max-seq-len 2048
```

Implementation detail in this repo:

- `build_llama_aotinductor.py` uses `torch.export.Dim("seq_len", min=..., max=...)`
- `dynamic_shapes=({1: seq_dim}, {1: seq_dim})` for `(input_ids, attention_mask)`

If dynamic export fails for your exact stack/model, keep static export as fallback.

## Benchmark Modes (Compile+Preload / No-Preload / Eager)

This folder now includes `benchmark_llama_modes.py` and `benchmark.sh` with three modes:

- `compile_preload`: load prebuilt `.pt2` and run inference
- `compile_no_preload`: export+compile `.pt2` at runtime, then run inference
- `eager`: plain HF `generate(..., disable_compile=True)`

Run:

```bash
./benchmark.sh
```

Quick cold-start comparison (2 repeats, 1 request per run):

```bash
./compare.sh
```

For your requested stress run:

```bash
./benchmark.sh --num-requests 100 --repeats 2
```

## Benchmark Metrics

- `ttft_cold` / `ttft_cold_start_seconds`: process-start TTFT including build/load work.
- `ttft_ready` / `ttft_after_ready_seconds`: first-request latency after setup is complete.
- `tail_mean`: average request latency after request 1.
- `mean`: average request latency across all requests.
- `tokens/s_all` / `tokens_per_second_all`: throughput including request 1.
- `tail_tokens/s` / `tail_tokens_per_second`: steady-state throughput using requests 2..N only.
- `end_to_end_tokens/s` / `end_to_end_tokens_per_second`: throughput including full script time.
- `total_script`: wall time of the benchmark subprocess.
