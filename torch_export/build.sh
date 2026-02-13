python build_llama_aotinductor.py \
    --model-id meta-llama/Llama-3.2-3B-Instruct \
    --package-path artifacts/llama3b_aotinductor.pt2 \
    --metadata-path artifacts/llama3b_aotinductor_meta.json \
    --example-seq-len 128 \
    --no-dynamic-seq-len \
    --max-autotune \
    "$@"
