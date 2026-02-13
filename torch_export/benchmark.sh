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
    --dynamic-seq-multiple 8 \
    "$@"
