python build_llama_portable_cache.py \
    --model-id meta-llama/Llama-3.2-3B-Instruct \
    --cache-path artifacts/llama3b_torchcompile_cache.bin \
    --metadata-path artifacts/llama3b_torchcompile_cache_meta.json \
    --prefill-lengths 128,256,512,1024 \
    --decode-length 128 \
    --cache-implementation static \
    --compile-dynamic auto \
    --compile-mode max-autotune \
    --fullgraph \
    "$@"
