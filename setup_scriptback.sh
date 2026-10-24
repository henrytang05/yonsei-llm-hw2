#!/usr/bin/env bash
set -euo pipefail

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clear uv cache to free disk space before large downloads
rm -rf "$HOME/.cache/uv"

# Create virtualenv
uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install setuptools

# Clone OpenRLHF (skip if already done)
mkdir -p third_party
if [ ! -d "third_party/OpenRLHF" ]; then
    git clone https://github.com/OpenRLHF/OpenRLHF.git third_party/OpenRLHF
fi

# Install PyTorch cu124 wheels FIRST and pin them explicitly.
# torch==2.6.0 is the latest version available on the cu124 index.
# Installing torch alone first prevents other packages from pulling in cu130 wheels.
uv pip install \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

# Install OpenRLHF in editable mode
uv pip install -e "third_party/OpenRLHF" --no-build-isolation

# Remove flash-attn: pulled in by OpenRLHF but compiled against wrong torch ABI.
# vLLM uses the Triton attention backend instead.
uv pip uninstall flash-attn -y || true

# Install vllm — latest compatible with torch 2.6.0 / cu124
uv pip install "vllm==0.8.5.post1"

# Install Jupyter and extras — torch pin prevents accelerate from upgrading to cu130
# No -U flags: do not upgrade already-resolved torch/nvidia packages
uv pip install \
    jupyter ipykernel \
    transformers datasets accelerate \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

# Register Jupyter kernel
python -m ipykernel install --user --name hw2 --display-name "Python (hw2)"

# DeepSpeed last — it builds CUDA extensions against the installed torch
uv pip install deepspeed

echo "Setup complete. Run 'source .venv/bin/activate' to activate the environment."

# Set HF_TOKEN in your shell or .env — do not hardcode tokens here
# export HF_TOKEN='...'
