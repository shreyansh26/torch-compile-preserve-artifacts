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
./run.sh --no-bucket-pad --max-new-tokens 128
```

Benchmark all modes:

```bash
cd torch_export
./benchmark.sh
```

### Limitations

- Dynamic export is now viable on this stack using the masking workaround in `build_llama_aotinductor.py`.
- Dynamic benchmark defaults assume `dynamic_seq_multiple=8` and that preload package metadata matches it.
- If you need static fallback for export, run `./build.sh --no-dynamic-seq-len` and use `--max-new-tokens 1` at runtime.
- `tokens/s_all` can favor `compile_no_preload` due to first-request warmup effects; compare `tail_tokens/s` and `end_to_end_tokens/s` for fair steady-state and full-path comparisons.

## Important Comparison Notes

### 1) Why `torch_export` can do true dynamic shape (for this flow) and `mega_cache` cannot

- `torch_export` captures an explicit export contract (`torch.export.Dim` bounds/guards) and packages it as a `.pt2` artifact. Runtime executes that compiled contract directly.
- `mega_cache` stores compiler cache artifacts produced from warmup traffic. It reuses cached entries when runtime keys and shapes match; otherwise recompilation can still happen.
- In short: `torch_export` is contract-driven; `mega_cache` is warmup/cache-key-driven.

### 2) Does `mega_cache` support full `1..2048` prompt lengths?

- Not as a strict guarantee by default.
- Default warmup uses fixed buckets (`128,256,512,1024`) and fixed decode length (`128`).
- You can improve coverage by warming more buckets and keeping `--bucket-pad` enabled at runtime.
- `--compile-dynamic true` can help shape generalization, but it is still not equivalent to an explicit exported dynamic-shape contract.

Example of wider warmup coverage:

```bash
cd mega_cache
./build.sh \
  --compile-dynamic true \
  --prefill-lengths 8,16,32,64,128,256,384,512,768,1024,1536,2048 \
  --decode-length 128
```

### 3) Meaning of `--dynamic-seq-multiple 8` in `torch_export`

- Sequence length is exported as a constrained dynamic dim (`seq_len = 8 * k`) within min/max bounds.
- Runtime pads per decode step to that multiple before calling the compiled model, then reads logits at the true unpadded token position.
- This is a reliability workaround for the current torch/transformers stack, not a model-quality requirement.

### 4) How to read `compile_preload` vs `compile_no_preload` benchmark metrics

- `tokens/s_all` includes request 1 and can be slightly higher for `compile_no_preload` because in-process compile work often warms runtime state before first measured request.
- `tail_tokens/s` is the best steady-state throughput comparison.
- `ttft_cold` and `end_to_end_tokens/s` are best for service-level startup/cold-path comparison.
- Practical benefit of preload is lower cold-start latency and usually better full-path throughput, even if `tokens/s_all` is close.

### 5) Which approach to use

- Use `torch_export` when you want a deployable artifact with bounded dynamic-shape behavior and predictable runtime execution.
- Use `mega_cache` when you want to stay in HF `generate` + `torch.compile` flow and are willing to manage warmup buckets/cache-key matching for best hit rates.

### 6) Practical Findings From Fresh Compare Runs

- `torch_export` preload is the most logical cold-path winner in this repo: it loads an already compiled `.pt2` package and typically shows near-zero runtime compile misses in preload mode.
- `torch_export` no-preload is intentionally expensive because it compiles at runtime, so setup and first-run generation are much slower than preload.
- `mega_cache` preload improves generation versus skip, but setup load-vs-skip often remains close because most work still happens in runtime `generate` compile/autotune paths.
- Even with preload, `mega_cache` can still show compile misses; this is expected because it is cache-key reuse over runtime compilation, not an AOT packaged executable graph.
- Bottom line: if your priority is lowest cold-start latency and strongest predictability, prefer `torch_export`; if your priority is staying in native HF `generate` workflow with incremental cache reuse, use `mega_cache`.

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
