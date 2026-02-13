# Llama 3.2 3B Instruct Portable `torch.compile` Cache

This folder provides a portable cache workflow:

1. Build and save cache artifacts (`torch.compiler.save_cache_artifacts()`)
2. Run with preload (`torch.compiler.load_cache_artifacts()`) or skip preload
3. Compare cold-start behavior and benchmark all modes

## Environment

- `torch==2.9.1`
- `transformers==4.57.5`

## Build

Script:

```bash
./build.sh
```

Expanded command (`build.sh`):

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

Default outputs:

- `artifacts/llama3b_torchcompile_cache.bin`
- `artifacts/llama3b_torchcompile_cache_meta.json`

## Run

Script:

```bash
./run.sh
```

Expanded command (`run.sh`):

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

Common mode overrides:

- Skip artifact preload: `./run.sh --no-load-artifacts`
- Force preload: `./run.sh --load-artifacts`

## Compare

Script:

```bash
./compare.sh
```

Expanded command (`compare.sh`):

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

Useful options:

- Show text output per trial: `./compare.sh --show-completions`
- If `/tmp` is constrained: `TMPDIR=/path/with/space ./compare.sh`

Compare output includes:

- Per trial: `generation`, `script_total`, `wall`, `new_tokens`, `gen_tps`
- Summary: setup/generation/total averages and speedups, plus compile/cache counters

## Benchmark

Script:

```bash
./benchmark.sh
```

If `/tmp` is constrained:

```bash
TMPDIR=/path/with/space ./benchmark.sh
```

Expanded command (`benchmark.sh`):

```bash
python benchmark_llama_modes.py \
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
  --num-requests 50 \
  --repeats 2
```

Modes covered by benchmark:

- `compile_preload`
- `compile_no_preload`
- `eager`

## Notes

- For best cache-hit fidelity, keep runtime compile settings aligned with build settings.
- For fair comparisons, keep prompt, bucketing, and token count identical.
- If comparing preload vs skip manually, isolate compiler caches between runs.
