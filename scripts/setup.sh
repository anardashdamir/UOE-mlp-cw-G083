#!/bin/bash
set -e

# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Clone repo
cd ~
git clone -b main https://github.com/anardashdamir/UOE-mlp-cw-G083.git
cd UOE-mlp-cw-G083

# 3. Install base dependencies
uv sync
source .venv/bin/activate

# 4. Install unsloth into venv (uv pip, not uv sync)
uv pip install "unsloth[cu124-ampere-torch260] @ git+https://github.com/unslothai/unsloth.git" --no-build-isolation

echo "Setup complete. Run: cd ~/UOE-mlp-cw-G083 && source .venv/bin/activate"
