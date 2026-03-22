#!/bin/bash
set -e

# Clone repo
cd ~
git clone -b main https://github.com/anardashdamir/UOE-mlp-cw-G083.git
cd UOE-mlp-cw-G083

# Install everything into system python
pip install -e .
pip install "unsloth[cu124-ampere-torch260] @ git+https://github.com/unslothai/unsloth.git" --no-build-isolation
pip uninstall torchaudio -y 2>/dev/null || true

echo "Setup complete. Run: cd ~/UOE-mlp-cw-G083"
