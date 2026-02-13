python run_llama_aotinductor.py \
    --package-path artifacts/llama3b_aotinductor.pt2 \
    --metadata-path artifacts/llama3b_aotinductor_meta.json \
    --model-id meta-llama/Llama-3.2-3B-Instruct \
    --bucket-pad \
    --prompt "Write a short explanation of portable torch.compile caches." \
    --max-new-tokens 1 \
    "$@"
