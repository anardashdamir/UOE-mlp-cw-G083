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

        # Determine adapter directory
        model_short=$(echo "$model" | cut -d'/' -f2)
        if [ "$think" = "true" ]; then
            adapter_dir="output/${model_short}/thinking/sft"
        else
            adapter_dir="output/${model_short}/no_thinking/sft"
        fi

        # Evaluate every checkpoint
        for ckpt in "$adapter_dir"/checkpoint-*; do
            if [ -d "$ckpt" ]; then
                echo "--------------------------------------------"
                echo "Evaluating checkpoint: $ckpt"
                echo "--------------------------------------------"
                python main.py evaluate \
                    --model "$model" \
                    --enable-thinking "$think" \
                    --sft-adapter "$ckpt"
            fi
        done

        # Evaluate final adapter
        echo "--------------------------------------------"
        echo "Evaluating final: $adapter_dir"
        echo "--------------------------------------------"
        python main.py evaluate \
            --model "$model" \
            --enable-thinking "$think" \
            --sft-adapter "$adapter_dir"
    done
done

echo "All runs complete."
