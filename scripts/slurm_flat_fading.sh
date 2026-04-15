#!/bin/bash
# =============================================================================
# SLURM job script — flat-fading paper replication (Fan, Yang, Shen — ICC 2025)
#
# Trains all 12 models (DEFINED + ICL-only for each of 6 configs) and
# generates Figure 4.  Total estimated wall time on one A100: ~18–22 hrs.
#
# ── Submission options ────────────────────────────────────────────────────────
#
# Option A — single job (all 6 configs sequential, simplest):
#   sbatch scripts/slurm_flat_fading.sh
#
# Option B — job array (6 configs in parallel, ~3–4 hrs wall time each):
#   TRAIN_JID=$(sbatch --parsable --array=0-5 scripts/slurm_flat_fading.sh --array)
#   sbatch --dependency=afterok:$TRAIN_JID scripts/slurm_flat_fading.sh --eval-only
#
# =============================================================================

# ── Scheduler directives ──────────────────────────────────────────────────────
#SBATCH --job-name=DEFINED_flat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cheetah01,cheetah04,serval03,serval[06-09]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/flat_%j.out
#SBATCH --error=logs/flat_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weh7xp@virginia.edu   # ← update this

# ── Move to project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Parse optional script-level flags ────────────────────────────────────────
# --array        : act as a job-array element (uses $SLURM_ARRAY_TASK_ID)
# --eval-only    : skip training, run evaluate.py only
# --config_idx N : train and/or evaluate only config index N (0-5)
RUN_ARRAY=false
EVAL_ONLY=false
CONFIG_IDX=-1
while [[ $# -gt 0 ]]; do
    case $1 in
        --array)       RUN_ARRAY=true;  shift ;;
        --eval-only)   EVAL_ONLY=true;  shift ;;
        --config_idx)  CONFIG_IDX="$2"; shift 2 ;;
        *)             shift ;;
    esac
done

# =============================================================================
# Environment setup
# =============================================================================
mkdir -p logs figures models

echo "===== Job info =============================================="
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Array task  : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "  Node        : $SLURMD_NODENAME"
echo "  Working dir : $(pwd)"
echo "  Start time  : $(date)"
echo "============================================================="

module purge
module load miniforge

# Create conda env on first run; activate on all subsequent runs
if ! conda env list | grep -q "^defined "; then
    echo "===== Creating conda environment (first run only) ==========="
    conda create -n defined python=3.10 -y
    conda activate defined

    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
               | head -1 | cut -d. -f1)
    if   [ "$CUDA_VER" -ge 126 ] 2>/dev/null; then TORCH_CUDA="cu126"
    elif [ "$CUDA_VER" -ge 124 ] 2>/dev/null; then TORCH_CUDA="cu124"
    elif [ "$CUDA_VER" -ge 121 ] 2>/dev/null; then TORCH_CUDA="cu121"
    else                                            TORCH_CUDA="cu118"
    fi
    pip install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
    pip install transformers wandb matplotlib numpy
    echo "============================================================="
else
    conda activate defined
fi

echo ""
echo "===== GPU / software versions ==============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PYEOF'
import torch, transformers
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
print(f"transformers: {transformers.__version__}")
PYEOF
echo "============================================================="

# =============================================================================
# Training
# =============================================================================
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "===== Training started : $(date) =========================="

    if [ "$RUN_ARRAY" = true ]; then
        echo "  Array element $SLURM_ARRAY_TASK_ID"
        python run_experiments.py --config_idx "$SLURM_ARRAY_TASK_ID"
    elif [ "$CONFIG_IDX" -ge 0 ] 2>/dev/null; then
        echo "  Training config index $CONFIG_IDX only"
        python run_experiments.py --config_idx "$CONFIG_IDX"
    else
        python run_experiments.py
    fi

    echo ""
    echo "===== Training complete : $(date) ========================="
fi

# =============================================================================
# Evaluation  (skipped when running as an array training element)
# =============================================================================
if [ "$EVAL_ONLY" = true ] || [ "$RUN_ARRAY" = false ]; then
    echo ""
    echo "===== Evaluation started : $(date) ========================"
    if [ "$CONFIG_IDX" -ge 0 ] 2>/dev/null; then
        python evaluate.py --config_idx "$CONFIG_IDX"
    else
        python evaluate.py
    fi
    echo ""
    echo "===== Evaluation complete : $(date) ======================="
    echo ""
    echo "  Figure saved to:"
    ls -lh figures/figure4.* 2>/dev/null || echo "  (no figure4 files yet)"
fi

echo ""
echo "===== Job finished : $(date) ================================="
