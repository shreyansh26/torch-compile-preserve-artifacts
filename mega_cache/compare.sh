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
    --compare-runs 2 \
    "$@"
