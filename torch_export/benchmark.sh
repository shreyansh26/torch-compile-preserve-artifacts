python benchmark_llama_modes.py \
    --model-id meta-llama/Llama-3.2-3B-Instruct \
    --package-path artifacts/llama3b_aotinductor.pt2 \
    --metadata-path artifacts/llama3b_aotinductor_meta.json \
    --cache-implementation static \
    --bucket-pad \
    --prompt "Write a short explanation of portable torch.compile caches." \
    --max-new-tokens 1 \
    --num-requests 3 \
    --repeats 1 \
    "$@"
