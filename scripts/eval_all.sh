#!/bin/bash
set -e

MODELS=("Qwen/Qwen3.5-0.8B" "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B")

for model in "${MODELS[@]}"; do
    for think in false true; do
        echo "============================================"
        echo "Evaluating: $model | thinking=$think"
        echo "============================================"
        python main.py evaluate \
            --model "$model" \
            --enable-thinking "$think"
    done
done

echo "All evaluations complete."
