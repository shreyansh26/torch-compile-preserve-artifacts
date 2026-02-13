# Llama 3.2 3B Instruct Portable torch.compile Cache

This repo implements a two-stage workflow so you can compile once, save portable cache artifacts, and load them later to avoid most cold-start compilation/autotuning work.

It uses:

- `torch.compiler.save_cache_artifacts()` after offline warmup
- `torch.compiler.load_cache_artifacts()` at server/runtime startup

## Environment

- `torch==2.9.1`
- `transformers==4.57.5`

## 1) Offline compile + cache export

Run this once on the same GPU class and software stack you will use in serving:

```bash
python build_llama_portable_cache.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --cache-path artifacts/llama3b_torchcompile_cache.bin \
  --metadata-path artifacts/llama3b_torchcompile_cache_meta.json \
  --prefill-lengths 128,256,512,1024 \
  --decode-length 128 \
  --cache-implementation static \
  --compile-dynamic auto \
  --compile-mode max-autotune \
  --fullgraph
```

What this does:

- Loads model + tokenizer on CUDA
- Uses `generate(..., cache_implementation="static", compile_config=...)`
- Runs representative warmup buckets so generate auto-compilation executes offline
- Saves cache bytes to `artifacts/llama3b_torchcompile_cache.bin`

## 2) Runtime startup with preloaded cache

Load cache bytes before creating/compiling the runtime model:

```bash
python run_llama_with_portable_cache.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --cache-path artifacts/llama3b_torchcompile_cache.bin \
  --metadata-path artifacts/llama3b_torchcompile_cache_meta.json \
  --cache-implementation static \
  --compile-dynamic auto \
  --compile-mode max-autotune \
  --fullgraph \
  --bucket-pad \
  --prompt "Write a short explanation of portable torch.compile caches." \
  --max-new-tokens 128
```

## 3) Compare startup latency with and without artifact preload

Use this to run cold-start trials for both modes in isolated compiler cache directories:

```bash
python run_llama_with_portable_cache.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --cache-path artifacts/llama3b_torchcompile_cache.bin \
  --metadata-path artifacts/llama3b_torchcompile_cache_meta.json \
  --cache-implementation static \
  --compile-dynamic auto \
  --compile-mode max-autotune \
  --fullgraph \
  --bucket-pad \
  --prompt "Write a short explanation of portable torch.compile caches." \
  --max-new-tokens 128 \
  --compare-load-vs-skip \
  --compare-runs 2
```

Optional: add `--show-completions` to print generated text per trial.

Default compare output (without `--show-completions`) includes:

- Per-trial: `generation`, `script_total`, and subprocess `wall` time.
- Summary averages for load vs skip, including setup/generation/total speedups.
- Compile/cache counters: average `unique_graphs` and `async_compile_miss`.

If you only want one mode:

- With preload (default): `--load-artifacts`
- Without preload: `--no-load-artifacts`

For a fair single-run manual A/B, isolate compiler caches per run:

```bash
TORCHINDUCTOR_CACHE_DIR=$(mktemp -d) TRITON_CACHE_DIR=$(mktemp -d) \
  ./run.sh --load-artifacts
TORCHINDUCTOR_CACHE_DIR=$(mktemp -d) TRITON_CACHE_DIR=$(mktemp -d) \
  ./run.sh --no-load-artifacts
```

## 4) Baseline HF Eager (No Compile)

This script does not use `torch.compile` or `compile_config` (it passes `disable_compile=True`):

```bash
python run_llama_hf_eager.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --cache-implementation static \
  --bucket-pad \
  --prompt "Write a short explanation of portable torch.compile caches." \
  --max-new-tokens 128
```

## 5) Multi-Request Benchmark (All Modes)

Runs in isolated subprocesses for:

- `eager` (no compile)
- `compile_preload` (portable artifact preload)
- `compile_no_preload` (same compile path, no artifact preload)

```bash
./benchmark.sh
```

You can change requests per mode and repeats:

```bash
./benchmark.sh --num-requests 100 --repeats 2
```

## Benchmark Metrics

- `ttft_cold` / `ttft_cold_start_seconds`: process-start TTFT including load/preload/model setup.
- `ttft_ready` / `ttft_after_ready_seconds`: first-request latency after setup is complete.
- `tail_mean`: average request latency after request 1.
- `mean`: average request latency across all requests.
- `tokens/s_all` / `tokens_per_second_all`: throughput including request 1.
- `tail_tokens/s` / `tail_tokens_per_second`: steady-state throughput using requests 2..N only.
- `end_to_end_tokens/s` / `end_to_end_tokens_per_second`: throughput including full script time.
- `total_script`: wall time of the benchmark subprocess.

For portable-cache A/B fidelity, keep runtime decode length aligned with build warmup
(`max_new_tokens == warmup.decode_length`). The benchmark defaults to fixed decode
(`min_new_tokens=max_new_tokens`) to avoid EOS-shortened runs.

## Notes

- Cache hits require matching compile/runtime keys (model, mode, fullgraph, dtype, torch/triton stack, and CUDA GPU compatibility).
- If compile settings or platform differ, PyTorch may recompile parts of the model.
- For best results, warm up the same shape buckets you expect in production traffic.
- Use `--bucket-pad` at runtime so prompt lengths map to warmed prefill buckets.
- Llama gated models may require `HF_TOKEN` and accepted model license in Hugging Face.
