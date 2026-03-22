#!/bin/bash
set -e

MODELS=("Qwen/Qwen3.5-0.8B" "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B")

for model in "${MODELS[@]}"; do
    for think in false true; do
        echo "============================================"
        echo "Training: $model | thinking=$think"
        echo "============================================"
        python main.py sft \
            --model "$model" \
            --enable-thinking "$think" \
            --wandb
    done
done

echo "All training runs complete."
