python build_llama_aotinductor.py \
    --model-id meta-llama/Llama-3.2-3B-Instruct \
    --package-path artifacts/llama3b_aotinductor.pt2 \
    --metadata-path artifacts/llama3b_aotinductor_meta.json \
    --example-seq-len 128 \
    --dynamic-seq-len \
    --min-seq-len 1 \
    --max-seq-len 2048 \
    --dynamic-seq-multiple 8 \
    --max-autotune \
    "$@"
