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
  --dynamic-seq-len \
  --min-seq-len 1 \
  --max-seq-len 2048 \
  --dynamic-seq-multiple 8 \
  --max-autotune
```

## 2) Run From `.pt2` Package

```bash
./run.sh
```

Run without loading prebuilt `.pt2` package (runtime `torch.compile` path):

```bash
./run.sh --no-load-artifacts
```

Explicitly force package preload path:

```bash
./run.sh --load-artifacts
```

`run.sh` modes:

- `--load-artifacts` (default): compile_preload (`aoti_load_package` from prebuilt `.pt2`)
- `--no-load-artifacts`: compile_no_preload (`torch.compile` at runtime, then run)

Compare both modes (completion text + timing) in isolated cold-start subprocesses:

```bash
./run.sh --compare-load-vs-skip --compare-runs 1
```

Optional: add `--show-completions` to print generated text per trial.

Default compare output (without `--show-completions`) includes:

- Per-trial: `setup`, `generation`, `script_total`, and subprocess `wall` time.
- Summary averages for load vs skip, including setup/generation/total speedups.
- Compile/cache counters: average `unique_graphs` and `async_compile_miss`.

Equivalent direct command:

```bash
python run_llama_aotinductor.py \
  --package-path artifacts/llama3b_aotinductor.pt2 \
  --metadata-path artifacts/llama3b_aotinductor_meta.json \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --bucket-pad \
  --prompt "Write a short explanation of portable torch.compile caches." \
  --max-new-tokens 128
```

## Notes

- The exported wrapper compiles `forward(input_ids, attention_mask) -> logits`.
- Runtime generation here uses a simple greedy autoregressive loop (no KV cache), intended as a clean artifact-loading example.
- `build.sh` now defaults to dynamic-sequence export with `dynamic_seq_multiple=8`.
- To force static export fallback, pass `--no-dynamic-seq-len` to `./build.sh` and keep runtime `--max-new-tokens 1`.
- Since this is `torch.export`-based, model/export graph compatibility matters more than `torch.compile` mega-cache flow.
- Keep PyTorch/CUDA/GPU stack compatible between build and runtime.

## Dynamic Sequence Dimensions

`build.sh` already enables dynamic sequence length by default. Explicit equivalent:

```bash
./build.sh \
  --dynamic-seq-len \
  --min-seq-len 1 \
  --max-seq-len 2048 \
  --dynamic-seq-multiple 8
```

Implementation detail in this repo:

- `build_llama_aotinductor.py` applies a masking workaround by default (`--masking-workaround`) for current `transformers==4.57.x` dynamic export behavior.
- Dynamic sequence dim uses `torch.export.Dim` with optional derived multiple (default `--dynamic-seq-multiple 8`).
- `dynamic_shapes=({1: seq_dim}, {1: seq_dim})` for `(input_ids, attention_mask)`
- `run_llama_aotinductor.py` auto-pads per-step inputs to the exported multiple and reads logits from the true (unpadded) token position, so `--max-new-tokens > 1` works for dynamic packages.

Example dynamic run with multi-token generation:

```bash
python run_llama_aotinductor.py \
  --package-path artifacts/llama3b_aotinductor.pt2 \
  --metadata-path artifacts/llama3b_aotinductor_meta.json \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --no-bucket-pad \
  --max-new-tokens 20
```

If dynamic export still fails for your exact stack/model, keep static export as fallback.

## Dynamic Flag Naming

- `--dynamic-seq-len` is a build/export flag (`build_llama_aotinductor.py`) and controls whether the `.pt2` artifact is exported with dynamic sequence guards.
- `--compile-dynamic-seq-len` is a runtime compile flag (`run_llama_aotinductor.py` / `benchmark_llama_modes.py`) and controls `torch.compile(dynamic=...)` in `compile_no_preload` mode.
- In `compile_no_preload` mode, `torch.compile` is run with `fullgraph=True` by default in both run and benchmark flows (no `--no-fullgraph` toggle in `torch_export` scripts).
- `build.sh` uses `--dynamic-seq-len` (export-time behavior), not `--compile-dynamic-seq-len`.
- `run.sh` does not pass `--compile-dynamic-seq-len` explicitly because `run_llama_aotinductor.py` defaults it to enabled; override with `./run.sh --no-load-artifacts --no-compile-dynamic-seq-len` if needed.
- `benchmark.sh` passes `--compile-dynamic-seq-len` explicitly because benchmark defaults keep that flag off unless requested.

## Benchmark Modes (Compile+Preload / No-Preload / Eager)

This folder now includes `benchmark_llama_modes.py` and `benchmark.sh` with three modes:

- `compile_preload`: load prebuilt `.pt2` and run inference
- `compile_no_preload`: runtime `torch.compile` (no `.pt2` load), then run inference
- `eager`: plain HF `generate(..., disable_compile=True)`

`benchmark.sh` is intentionally separate from `run.sh`; both now use runtime `torch.compile` for `compile_no_preload`.

Before running the default benchmark scripts, run `./build.sh` once to refresh the preload package.

Run:

```bash
./benchmark.sh
```

Current `benchmark.sh` defaults:

- `--max-new-tokens 128`
- `--num-requests 50`
- `--repeats 2`
- `--compile-dynamic-seq-len --min-seq-len 1 --max-seq-len 2048 --dynamic-seq-multiple 8`

Quick cold-start comparison (2 repeats, 1 request per run):

```bash
./compare.sh
```

Equivalent direct benchmark command:

```bash
python benchmark_llama_modes.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --package-path artifacts/llama3b_aotinductor.pt2 \
  --metadata-path artifacts/llama3b_aotinductor_meta.json \
  --cache-implementation static \
  --bucket-pad \
  --prompt "Write a short explanation of portable torch.compile caches." \
  --max-new-tokens 128 \
  --num-requests 50 \
  --repeats 2 \
  --compile-dynamic-seq-len \
  --min-seq-len 1 \
  --max-seq-len 2048 \
  --dynamic-seq-multiple 8
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

Interpretation note:

- `tokens/s_all` can be slightly higher for `compile_no_preload` because compile-time warmup can reduce its first-request (`ttft_ready`) cost.
- Use `tail_tokens/s` for steady-state model speed comparison.
- Use `end_to_end_tokens/s` and `ttft_cold` for service-level startup/cold-path comparisons.
