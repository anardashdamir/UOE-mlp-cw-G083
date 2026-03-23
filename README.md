# AutoFilter

**Natural-language queries → structured filter expressions**, powered by LoRA fine-tuned LLMs.

```
User:  "cheap Toyota cars under 20k with low mileage"
Model: brand == 'toyota' AND price < 20000 AND mileage < 50000
```

AutoFilter fine-tunes small language models (Qwen3.5 0.8B/2B/4B) with LoRA to convert free-form, often misspelled or slang-heavy queries into precise, schema-aware filter expressions. It supports two training stages: **SFT** (supervised fine-tuning) and **GRPO** (reward-based reinforcement learning on top of SFT).

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [CLI Commands](#cli-commands)
  - [SFT Training](#sft-training)
  - [GRPO Training](#grpo-training)
  - [Evaluate](#evaluate)
  - [Predict](#predict)
  - [Utilities](#utilities)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Evaluation Framework](#evaluation-framework)

---

## How It Works

```
┌─────────────────┐     ┌────────────────┐     ┌──────────────────┐
│  Natural-Language│     │   Base LLM     │     │   Structured     │
│  Query + Schema  │ ──► │   + LoRA       │ ──► │   Filter Expr    │
│                  │     │   (fine-tuned)  │     │                  │
└─────────────────┘     └────────────────┘     └──────────────────┘
```

| Stage | Description | Module |
|-------|-------------|--------|
| **Data loading** | Load query–filter pairs, split by schema into train/eval | `src/data_loader.py` |
| **SFT Training** | LoRA fine-tune the base model with SFTTrainer | `src/train.py` |
| **GRPO Training** | Reward-based RL on top of the best SFT model | `src/train_grpo.py` |
| **Inference** | Generate filter expressions from new queries | `src/inference.py` |
| **Evaluation** | 12 metrics with per-schema and per-difficulty breakdowns | `src/evaluate/` |

---

## Project Structure

```
AutoFilter/
├── main.py                          # Entry point
├── config.yaml                      # All hyperparameters
├── pyproject.toml                   # Dependencies (uv)
├── src/
│   ├── cli.py                       # Typer CLI (sft, grpo, predict, evaluate, ...)
│   ├── config.py                    # Pydantic config with YAML loading
│   ├── data_loader.py               # Schema formatting + HF Dataset construction
│   ├── train.py                     # SFT with LoRA
│   ├── train_grpo.py                # GRPO with multi-reward functions
│   ├── inference.py                 # Model loading + filter generation
│   ├── training_utils.py            # Shared helpers (thinking mode, logging)
│   └── evaluate/                    # Modular evaluation framework
│       ├── base.py                  # BaseMetric ABC, SampleContext, EvaluationResult
│       ├── orchestrator.py          # Runs inference + computes all metrics
│       ├── parsing.py               # Clause parsing, field extraction
│       ├── exact_match.py           # Full expression exact match
│       ├── f1.py, precision.py, recall.py
│       ├── field_accuracy.py        # Schema field validation
│       ├── hallucination.py         # Extra clauses not in ground truth
│       ├── operator_accuracy.py     # F1 over operators
│       ├── value_accuracy.py        # F1 over literal values
│       └── ...
├── schemas/                         # 17 JSON dataset schemas
├── data/
│   ├── train.json                   # ~2,100 training samples (12 schemas)
│   └── test.json                    # ~460 eval samples (5 unseen schemas)
├── scripts/
│   ├── eval_baselines.sh            # Zero-shot baseline evaluation
│   ├── run_all.sh                   # Train + evaluate all model/thinking combos
│   ├── train_all.sh                 # Train all 6 variants
│   ├── eval_all.sh                  # Evaluate all checkpoints
│   ├── compare_checkpoints.py       # Compare checkpoint metrics
│   └── plot_results.py              # Generate result plots
├── docs/
│   ├── train.png                    # Training pipeline diagram
│   └── eval.png                     # Evaluation framework diagram
└── output/                          # Created during training
    └── <Model>/<thinking>/sft/      # Saved LoRA adapters
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (L40S 48GB recommended, or any GPU with 24GB+ VRAM)
- [uv](https://docs.astral.sh/uv/) — fast Python package manager

### Install

```bash
# 1. Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Clone the repository
git clone -b main https://github.com/anardashdamir/UOE-mlp-cw-G083.git
cd UOE-mlp-cw-G083

# 3. Create virtual environment and install all dependencies
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e .

# 4. Verify
python main.py --help
```

### GPU Server (RunPod / Cloud)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone -b main https://github.com/anardashdamir/UOE-mlp-cw-G083.git
cd UOE-mlp-cw-G083

uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e .
```

### SLURM Cluster (UoE)

The cluster has **no internet on GPU nodes** — download everything on the login node first.

```bash
# On the login node (has internet)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone -b main https://github.com/anardashdamir/UOE-mlp-cw-G083.git
cd UOE-mlp-cw-G083
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e .

# Pre-download the model
huggingface-cli download Qwen/Qwen3.5-4B

# Request a GPU node
srun --partition=Teaching --gres=gpu:1 --time=12:00:00 --mem=36G --pty bash

# On the GPU node
source $HOME/.local/bin/env
cd UOE-mlp-cw-G083
source venv/bin/activate
python main.py sft
```

---

## CLI Commands

All commands go through:

```bash
python main.py <command> [options]
```

### SFT Training

Fine-tune a base model with LoRA (supervised fine-tuning):

```bash
# Default settings from config.yaml
python main.py sft

# Specify model and thinking mode
python main.py sft --model Qwen/Qwen3.5-4B --enable-thinking true

# Custom hyperparameters
python main.py sft --model Qwen/Qwen3.5-2B --epochs 3 --lr 5e-5 --batch-size 32 --lora-r 16

# Quick test run
python main.py sft --max-steps 100

# With W&B logging
python main.py sft --wandb --run-name "4b-thinking-v1"

# Enable gradient checkpointing (saves VRAM)
python main.py sft --grad-ckpt

# QLoRA (4-bit base model)
python main.py sft --qlora
```

**Options:**

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | `-m` | HuggingFace model name |
| `--enable-thinking` | | `true` / `false` — enable Qwen thinking mode |
| `--epochs` | `-e` | Number of training epochs |
| `--batch-size` | `-b` | Per-device batch size |
| `--lr` | | Learning rate |
| `--lora-r` | | LoRA rank |
| `--max-steps` | | Max steps (-1 = use epochs) |
| `--output-dir` | `-o` | Output directory |
| `--wandb` | | Enable W&B logging |
| `--run-name` | `-r` | Experiment name |
| `--grad-ckpt` | | Gradient checkpointing |
| `--qlora` | | Load model in 4-bit |

The adapter is saved to `output/<Model>/<thinking_mode>/sft/`.

### GRPO Training

Run reward-based RL on top of a trained SFT adapter:

```bash
# Default: loads SFT adapter from config-derived path
python main.py grpo --model Qwen/Qwen3.5-4B --enable-thinking true

# Explicit SFT adapter path
python main.py grpo \
    --model Qwen/Qwen3.5-4B \
    --enable-thinking true \
    --sft-adapter output/Qwen3.5-4B/thinking/sft

# With W&B logging
python main.py grpo --model Qwen/Qwen3.5-4B --enable-thinking true --wandb
```

GRPO uses 5 reward functions:
- **Exact match** (weight 5.0) — binary match against ground truth
- **Clause F1** (weight 2.0) — F1 over normalized clauses
- **Syntax** (weight 1.0) — valid operators, balanced parens/quotes
- **Field** (weight 1.0) — correct field names
- **Hallucination penalty** (weight 1.0) — penalizes extra clauses

The GRPO adapter is saved to `output/<Model>/<thinking_mode>/grpo/`.

### Evaluate

Evaluate on 5 held-out schemas (460 samples):

```bash
# Evaluate SFT model
python main.py evaluate --model Qwen/Qwen3.5-4B --enable-thinking true

# Explicit adapter paths
python main.py evaluate \
    --model Qwen/Qwen3.5-4B \
    --enable-thinking true \
    --sft-adapter output/Qwen3.5-4B/thinking/sft

# Evaluate GRPO model (loads SFT + GRPO adapters)
python main.py evaluate \
    --model Qwen/Qwen3.5-4B \
    --enable-thinking true \
    --sft-adapter output/Qwen3.5-4B/thinking/sft \
    --grpo-adapter output/Qwen3.5-4B/thinking/grpo

# Zero-shot baseline (no adapter)
python main.py evaluate --model Qwen/Qwen3.5-4B --zero-shot

# Limit samples
python main.py evaluate --max-samples 100

# Compare quantization modes
python main.py evaluate -q fp16 -q int8 -q int4

# Verbose (print each prediction vs expected)
python main.py evaluate --verbose
```

### Predict

Generate a filter from a single query:

```bash
python main.py predict "cheap red Toyota cars" schemas/used_cars.json

# With thinking mode
python main.py predict "cheap red Toyota cars" schemas/used_cars.json --thinking

# With adapter
python main.py predict "cheap cars" schemas/used_cars.json \
    --sft-adapter output/Qwen3.5-4B/thinking/sft

# INT4 quantization
python main.py predict "budget hotels" schemas/hotel_bookings.json -q int4
```

### Utilities

```bash
# List available schemas
python main.py schemas
python main.py schemas --verbose

# Show train/eval dataset statistics
python main.py data-stats

# Check which schemas exceed token threshold
python main.py check-schemas --threshold 1024 --verbose
```

---

## Configuration

All settings are in `config.yaml`. CLI flags override config values.

```yaml
model:
  name: Qwen/Qwen3.5-4B
  enable_thinking: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear

training:
  num_epochs: 1
  batch_size: 32
  gradient_accumulation_steps: 1
  learning_rate: 5e-5
  lr_scheduler_type: cosine
  warmup_steps: 30
  max_seq_length: 2048
  gradient_checkpointing: true

grpo:
  num_epochs: 1
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 5e-6
  max_completion_length: 256
  num_generations: 8
  beta: 0.05
  max_steps: 50

generation:
  max_new_tokens: 512
  temperature: 0.0

paths:
  train_path: data/train.json
  test_path: data/test.json
  schema_dir: schemas
  output_dir: output
```

---

## Data Format

### Schemas

Each JSON schema in `schemas/` describes a dataset:

```json
{
  "name": "used_cars",
  "row_count": 106256,
  "columns": {
    "year":         { "type": "int", "min": 1991, "max": 2020, "median": 2017 },
    "price":        { "type": "int", "min": 450, "max": 159999, "median": 14579 },
    "transmission": { "type": "categorical", "values": ["Automatic", "Manual", "Semi-Auto"] },
    "brand":        { "type": "categorical", "values": ["audi", "bmw", "ford", "toyota"] }
  }
}
```

### Training Data

Each sample in `data/train.json`:

```json
{
  "file_path": "adidas_vs_nike__adversarial__23__v0",
  "query": "gonna cop some adiddas kicks, any deals?",
  "filters": "Brand == 'adidas'",
  "selected_fields": ["Brand"]
}
```

- **12 training schemas**, ~2,100 samples
- **5 eval schemas** (unseen during training), ~460 manually reviewed samples
- Queries include adversarial variants: typos, slang, abbreviations, informal language

---

## Evaluation Framework

12 metrics across four categories:

| Category | Metrics |
|----------|---------|
| **Core Quality** | Precision, Recall, F1, Exact Match |
| **Schema Alignment** | Field Accuracy, Misaligned Fields |
| **Structural** | Structural Validity, Complexity Accuracy, Hallucination Rate |
| **Fine-Grained** | Operator Accuracy, Value Accuracy, Latency |

Results are broken down by **schema** and **difficulty level** (easy, medium, hard, adversarial variants).

### Architecture Diagrams

![Training and Inference Pipeline](docs/train.png)
![Evaluation Framework](docs/eval.png)

---
