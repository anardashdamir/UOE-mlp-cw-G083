#!/bin/bash
set -e

# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Clone repo
git clone -b main https://github.com/anardashdamir/UOE-mlp-cw-G083.git
cd UOE-mlp-cw-G083

# 3. Install base dependencies (skip GPU extras)
uv sync --no-extra gpu
source .venv/bin/activate

# 4. Install GPU dependencies via pip
pip install "unsloth[cu124-ampere-torch260] @ git+https://github.com/unslothai/unsloth.git" --no-build-isolation

echo "Setup complete. Run: cd UOE-mlp-cw-G083 && source .venv/bin/activate"
