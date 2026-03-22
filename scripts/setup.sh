#!/bin/bash
set -e

cd ~
git clone -b main https://github.com/anardashdamir/UOE-mlp-cw-G083.git
cd UOE-mlp-cw-G083

pip install -e . --no-build-isolation
pip install "unsloth[cu124-ampere-torch260] @ git+https://github.com/unslothai/unsloth.git" --no-build-isolation
pip install pydantic-settings wandb tensorboard
pip uninstall torchaudio -y 2>/dev/null || true

echo "Setup complete."
