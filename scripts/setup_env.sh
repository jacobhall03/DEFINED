#!/bin/bash
# =============================================================================
# One-time environment setup for DEFINED.
#
# Run this interactively on a GPU node before submitting any SLURM job:
#
#   srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G \
#        --time=1:00:00 --pty bash
#   bash scripts/setup_env.sh
# =============================================================================

set -e   # exit immediately on any error

# Move to project root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "===== DEFINED environment setup ============================="
echo "  Working directory: $(pwd)"

# ── Load modules ──────────────────────────────────────────────────────────────
module purge
module load miniforge

# ── Create conda environment ──────────────────────────────────────────────────
if conda env list | grep -q "^defined "; then
    echo "  Conda env 'defined' already exists — skipping creation."
else
    echo "  Creating conda env 'defined' (Python 3.10)..."
    conda create -n defined python=3.10 -y
fi

# ── Activate ──────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate defined

# ── Install dependencies ──────────────────────────────────────────────────────
# Detect GPU compute capability to decide between stable and nightly PyTorch.
# RTX 5080 / Blackwell (sm_120+) requires nightly; everything older uses stable.
GPU_CAP=$(python - <<'PYEOF'
try:
    import subprocess, re
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap",
                                   "--format=csv,noheader"]).decode().strip()
    major = int(out.split(".")[0])
    print(major)
except Exception:
    print(0)
PYEOF
)

if [ "$GPU_CAP" -ge 12 ]; then
    echo "  Detected sm_${GPU_CAP}x GPU (Blackwell) — installing PyTorch nightly (CUDA 12.4)..."
    pip install --quiet --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu124
else
    echo "  Installing PyTorch stable (CUDA 12.1)..."
    pip install --quiet torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
fi

echo "  Installing remaining dependencies..."
pip install --quiet transformers wandb matplotlib numpy

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "===== Verification =========================================="
python - <<'PYEOF'
import torch, transformers, wandb, matplotlib, numpy
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
print(f"transformers : {transformers.__version__}")
print(f"numpy        : {numpy.__version__}")
print(f"matplotlib   : {matplotlib.__version__}")
PYEOF
echo "============================================================="
echo "  Setup complete."
echo "  Flat-fading experiments : sbatch scripts/slurm_flat_fading.sh"
echo "  OFDM experiments        : sbatch scripts/slurm_ofdm.sh"
