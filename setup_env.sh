#!/bin/bash
# =============================================================================
# One-time environment setup for DEFINED paper replication.
#
# Run this interactively on a GPU node before submitting slurm_job.sh:
#
#   srun --partition=gpu --gres=gpu:1 --nodelist=nekomata01 \
#        --cpus-per-task=8 --mem=32G --time=1:00:00 --pty bash
#   bash setup_env.sh
# =============================================================================

set -e   # exit immediately on any error

echo "===== DEFINED environment setup ============================="

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
echo "  Installing PyTorch (CUDA 12.1)..."
pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

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
echo "  Setup complete. You can now run:  sbatch slurm_job.sh"
