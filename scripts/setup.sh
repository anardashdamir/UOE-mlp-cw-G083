#!/bin/bash
set -e

# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Install base dependencies
uv sync
source .venv/bin/activate

# 3. Install GPU dependencies
pip install "unsloth[cu124-ampere-torch260] @ git+https://github.com/unslothai/unsloth.git" --no-build-isolation

echo "Setup complete."
