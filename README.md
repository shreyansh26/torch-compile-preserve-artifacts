# Torch Compile Artifacts: `mega_cache` vs `torch_export`

This project contains two different ways to reduce runtime compilation cost for Llama 3.2 3B Instruct.

## Environment

- `torch==2.9.1`
- `transformers==4.57.5`

## Approach 1: Portable Runtime Cache (`mega_cache`)

Uses:

- `torch.compiler.save_cache_artifacts()` after offline warmup
- `torch.compiler.load_cache_artifacts()` before serving

### Main Commands

Build cache artifacts:

```bash
cd mega_cache
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

Run with preload:

```bash
cd mega_cache
./run.sh --load-artifacts
```

Benchmark all modes:

```bash
cd mega_cache
./benchmark.sh --num-requests 50 --repeats 1
```

### Limitations

- Benefits depend on cache-key match (model/settings/shape/GPU/software stack).
- If runtime decode length differs from build warmup decode length, portable-cache hit rate can drop.
- If compile path is not actually triggered for a test case, preload vs no-preload may look similar.

## Approach 2: AOT Artifact (`torch_export`)

Uses:

- `torch.export.export(...)`
- `torch._inductor.aoti_compile_and_package(...)` -> `.pt2`
- `torch._inductor.aoti_load_package(...)`

### Main Commands

Build `.pt2`:

```bash
cd torch_export
./build.sh
```

Run from `.pt2`:

```bash
cd torch_export
./run.sh
```

Benchmark all modes:

```bash
cd torch_export
./benchmark.sh --num-requests 50 --repeats 1
```

### Limitations

- Current reliable path is static-sequence export in this stack.
- With static export, generation is effectively limited to `--max-new-tokens 1` in the provided runtime loop.
- Dynamic sequence export is supported in code, but may fail for this exact model/version stack.

## Metric Naming (Shared Across Both Benchmarks)

- `ttft_cold`: process-start TTFT including setup work.
- `ttft_ready`: first-request latency after setup is complete.
- `tail_mean`: mean latency over requests `2..N`.
- `tokens/s_all`: throughput including request 1.
- `tail_tokens/s`: steady-state throughput over requests `2..N`.
- `end_to_end_tokens/s`: throughput including full script time.

See folder-specific READMEs for all flags and details:

- `mega_cache/README.md`
- `torch_export/README.md`
