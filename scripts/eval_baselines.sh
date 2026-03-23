#!/bin/bash
set -e

MODELS=("Qwen/Qwen3.5-0.8B" "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B")

for model in "${MODELS[@]}"; do
    for think in false true; do
        echo "============================================"
        echo "Zero-shot: $model | thinking=$think"
        echo "============================================"
        python main.py evaluate \
            --model "$model" \
            --enable-thinking "$think" \
            --zero-shot
    done
done

echo "All baselines evaluated."
