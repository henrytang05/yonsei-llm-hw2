#!/usr/bin/env bash
set -euo pipefail

TORCH_IDX="https://download.pytorch.org/whl/cu124"
# Never write downloaded wheels to the uv cache — saves ~20 GB on the server.
export UV_NO_CACHE=1

# Pin torch so every subsequent uv pip install call (transformers, accelerate, etc.)
# never re-resolves to a different torch version or a CPU wheel from PyPI.
CONSTRAINTS_FILE="$(pwd)/.torch-constraints.txt"
cat > "$CONSTRAINTS_FILE" <<EOF
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
flash-attn==0.0.0
EOF
export UV_CONSTRAINT="$CONSTRAINTS_FILE"

# ── 1. Install uv (idempotent) ────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# ── 2. Create virtualenv ──────────────────────────────────────────────────────
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install setuptools wheel

# ── 3. Clone OpenRLHF (skip if already present) ──────────────────────────────
mkdir -p third_party
if [ ! -d "third_party/OpenRLHF/.git" ]; then
    git clone https://github.com/OpenRLHF/OpenRLHF.git third_party/OpenRLHF
fi

# ── 4. Install PyTorch cu124 FIRST ───────────────────────────────────────────
# Pin torch before OpenRLHF[vllm] so its resolution sees the right ABI.
# These are the latest cu124 wheels; adjust if the cu124 index adds newer ones.
uv pip install \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0" \
    --index-url "$TORCH_IDX"

# ── 5. Install OpenRLHF (without vllm extra) ─────────────────────────────────
# OpenRLHF[vllm] pins vllm==0.19.1 → torch==2.10.0, which has no cu124 wheel.
# Install core OpenRLHF first, then add vllm separately at a cu124-compatible version.
uv pip install \
    -e "third_party/OpenRLHF" \
    --no-build-isolation \
    --extra-index-url "$TORCH_IDX"

# ── 5b. Install vLLM compatible with torch 2.6.0 + cu124 ─────────────────────
# vllm 0.8.5.post1 is the latest release that works with torch 2.6.0.
# (vllm>=0.9 pins torch==2.7+; vllm==0.19.1 requires torch==2.10.0/cu130)
uv pip install "vllm==0.8.5.post1" --extra-index-url "$TORCH_IDX"

# ── 6. Remove flash-attn (compiled for wrong ABI; vLLM uses Triton instead) ──
uv pip uninstall flash-attn 2>/dev/null || true

# ── 7. Jupyter + common ML extras ────────────────────────────────────────────
uv pip install \
    jupyter \
    ipykernel \
    ipywidgets \
    transformers \
    datasets \
    accelerate

# ── 8. DeepSpeed (builds CUDA extensions against the installed torch) ─────────
uv pip install deepspeed

# ── 9. Register Jupyter kernel ───────────────────────────────────────────────
python -m ipykernel install --user --name hw2 --display-name "Python (hw2)"

echo ""
echo "Setup complete."
echo "  Activate: source .venv/bin/activate"
echo "  Verify:   python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo ""
echo "Note: set HF_TOKEN in your shell before running the notebook:"
echo "  export HF_TOKEN='hf_...'"
