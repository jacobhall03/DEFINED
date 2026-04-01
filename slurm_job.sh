#!/bin/bash
# =============================================================================
# SLURM job script — DEFINED paper replication (Fan, Yang, Shen — ICC 2025)
#
# Trains all 12 models (DEFINED + ICL-only for each of 6 configs) and
# generates Figure 4.  Total estimated wall time on one A100: ~18–22 hrs.
#
# ── Submission options ────────────────────────────────────────────────────────
#
# Option A — single job (all configs sequential, simplest):
#   sbatch slurm_job.sh
#
# Option B — job array (6 configs in parallel, ~3–4 hrs total wall time):
#   TRAIN_JID=$(sbatch --parsable slurm_job.sh --array)
#   sbatch --dependency=afterok:$TRAIN_JID slurm_job.sh --eval-only
#
# =============================================================================

# ── Scheduler directives ──────────────────────────────────────────────────────
#SBATCH --job-name=DEFINED_replicate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=nekomata01
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/defined_%j.out
#SBATCH --error=logs/defined_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weh7xp@virginia.edu   # ← update this

# ── Parse optional script-level flags ────────────────────────────────────────
# --array      : act as job-array element (uses $SLURM_ARRAY_TASK_ID)
# --eval-only  : skip training, run evaluate.py only
RUN_ARRAY=false
EVAL_ONLY=false
for arg in "$@"; do
    case $arg in
        --array)     RUN_ARRAY=true ;;
        --eval-only) EVAL_ONLY=true ;;
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
echo "  Start time  : $(date)"
echo "============================================================="

# ── Load cluster modules ─────────────────────────────────────────────────────
module purge
module load miniforge

# ── Activate conda environment ────────────────────────────────────────────────
# First-time setup — run this once interactively before submitting:
#
#   srun --partition=gpu --gres=gpu:1 --nodelist=nekomata01 \
#        --cpus-per-task=8 --mem=32G --time=1:00:00 --pty bash
#   module load miniforge
#   conda create -n defined python=3.10 -y
#   conda activate defined
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#   pip install transformers wandb matplotlib numpy
#   exit
#
conda activate defined

# ── Confirm GPU allocation ───────────────────────────────────────────────────
echo ""
echo "===== GPU / software versions ==============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PYEOF'
import torch, transformers
print(f"PyTorch    : {torch.__version__}")
print(f"CUDA avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
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
        # ── Job-array mode: each element trains one config ────────────────────
        # Submit as:
        #   sbatch --array=0-5 slurm_job.sh --array
        echo "  Array element $SLURM_ARRAY_TASK_ID / config index $SLURM_ARRAY_TASK_ID"
        python run_experiments.py --config_idx "$SLURM_ARRAY_TASK_ID"
    else
        # ── Single-job mode: all 6 configs sequentially ───────────────────────
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
    python evaluate.py
    echo ""
    echo "===== Evaluation complete : $(date) ======================="
    echo ""
    echo "  Figure saved to:"
    ls -lh figures/figure4.*
fi

echo ""
echo "===== Job finished : $(date) ================================="
