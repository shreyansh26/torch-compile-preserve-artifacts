# Llama 3.2 3B AOTInductor Artifact (`torch.export` -> `.pt2`)

This folder provides an AOT workflow:

1. Export with `torch.export.export(...)`
2. Package with `torch._inductor.aoti_compile_and_package(...)`
3. Run with preload (`aoti_load_package`) or no-preload (`torch.compile`)

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

Default outputs:

- `artifacts/llama3b_aotinductor.pt2`
- `artifacts/llama3b_aotinductor_meta.json`

## Run

Script:

```bash
./run.sh
```

Expanded command (`run.sh`):

```bash
python run_llama_aotinductor.py \
  --package-path artifacts/llama3b_aotinductor.pt2 \
  --metadata-path artifacts/llama3b_aotinductor_meta.json \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --bucket-pad \
  --prompt "Write a short explanation of portable torch.compile caches." \
  --max-new-tokens 128
```

Common mode overrides:

- No preload runtime compile: `./run.sh --no-load-artifacts`
- Force preload: `./run.sh --load-artifacts`

## Compare

Script:

```bash
./compare.sh
```

Expanded command (`compare.sh`):

```bash
python run_llama_aotinductor.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --package-path artifacts/llama3b_aotinductor.pt2 \
  --metadata-path artifacts/llama3b_aotinductor_meta.json \
  --bucket-pad \
  --prompt "Write a short explanation of portable torch.compile caches." \
  --max-new-tokens 128 \
  --compare-load-vs-skip \
  --compare-runs 2
```

Useful options:

- Show text output for each trial: `./compare.sh --show-completions`
- If `/tmp` is constrained: `TMPDIR=/path/with/space ./compare.sh`

Compare output includes:

- Per trial: `setup`, `generation`, `script_total`, `wall`, `new_tokens`, `gen_tps`
- Summary: load vs skip averages, speedups, and compile/cache counters

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

Modes covered by benchmark:

- `compile_preload`
- `compile_no_preload`
- `eager`

## Notes

- `build.sh` controls export-time dynamic behavior (`--dynamic-seq-len` flags).
- Runtime compile behavior for no-preload is controlled by `--compile-dynamic-seq-len` flags.
- In no-preload mode, runtime uses `torch.compile(..., fullgraph=True)`.
- For fair comparisons, keep prompt and token settings identical across modes.
