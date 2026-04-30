#!/bin/bash
# =============================================================================
# SLURM job script — OFDM frequency-selective channel experiments
#
# Trains DEFINED and ICL-only models for OFDM configs and produces
# SER-vs-SNR and SER-vs-context figures per config.
#
# ── Submission options ────────────────────────────────────────────────────────
#
# Option A — single job (all configs sequential):
#   sbatch scripts/slurm_ofdm.sh
#
# Option B — job array (one config per array element):
#   TRAIN_JID=$(sbatch --parsable --array=0-4 scripts/slurm_ofdm.sh --array)
#   sbatch --dependency=afterok:$TRAIN_JID scripts/slurm_ofdm.sh --eval-only
#
# BPSK/QPSK array only:
#   sbatch --array=0-1 scripts/slurm_ofdm.sh --array
#
# Option C — evaluate previously trained checkpoints only:
#   sbatch scripts/slurm_ofdm.sh --eval-only
#
# Option D — BPSK and QPSK SISO only:
#   sbatch scripts/slurm_ofdm.sh --bpsk-qpsk
#
# =============================================================================

# ── Scheduler directives ──────────────────────────────────────────────────────
#SBATCH --job-name=DEFINED_ofdm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# Current defined env reports CUDA arch support through sm_90, but not Ada sm_89
# or Blackwell sm_120. Exclude those nodes unless the env is rebuilt for them.
# UVA CS gpu partition reference:
#   cheetah02  = RTX 4000 Ada Generation (sm_89)
#   nekomata01 = RTX 5080 (sm_120)
#SBATCH --exclude=cheetah02,nekomata01
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/ofdm_%j.out
#SBATCH --error=logs/ofdm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weh7xp@virginia.edu   # ← update this

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Move to project root ──────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/run_experiments_ofdm.py" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR/.."
fi

if [ ! -f "run_experiments_ofdm.py" ]; then
    echo "ERROR: Could not locate project root containing run_experiments_ofdm.py" >&2
    echo "       Submit from the repo root, e.g. cd ~/MLComms/DEFINED first." >&2
    exit 1
fi

# ── Parse optional script-level flags ────────────────────────────────────────
# --array        : act as a job-array element (uses $SLURM_ARRAY_TASK_ID)
# --eval-only    : skip training, run evaluation only
# --config_idx N : train and/or evaluate only config index N (0-4)
# --bpsk-qpsk    : train and/or evaluate SISO BPSK (0) and SISO QPSK (1)
RUN_ARRAY=false
EVAL_ONLY=false
CONFIG_IDX=-1
BPSK_QPSK=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --array)       RUN_ARRAY=true;  shift ;;
        --eval-only)   EVAL_ONLY=true;  shift ;;
        --config_idx)  CONFIG_IDX="$2"; shift 2 ;;
        --bpsk-qpsk)   BPSK_QPSK=true;  shift ;;
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
source "$(conda info --base)/etc/profile.d/conda.sh"

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
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    arch_list = torch.cuda.get_arch_list()
    print(f"Compute cap : {sm}")
    print(f"Torch archs : {arch_list}")
    if sm not in arch_list:
        raise SystemExit(
            f"ERROR: This PyTorch build does not support {sm}. "
            "Use a non-Blackwell GPU node or reinstall PyTorch nightly for RTX 50-series."
        )
print(f"transformers: {transformers.__version__}")
PYEOF
echo "============================================================="

# =============================================================================
# Training + evaluation
# =============================================================================
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "===== OFDM Training/evaluation started : $(date) =========="

    if [ "$RUN_ARRAY" = true ]; then
        echo "  Array element $SLURM_ARRAY_TASK_ID"
        python run_experiments_ofdm.py --config_idx "$SLURM_ARRAY_TASK_ID"
    elif [ "$BPSK_QPSK" = true ]; then
        echo "  Running SISO BPSK and SISO QPSK configs only"
        python run_experiments_ofdm.py --config_idx 0
        python run_experiments_ofdm.py --config_idx 1
    elif [ "$CONFIG_IDX" -ge 0 ] 2>/dev/null; then
        echo "  Training config index $CONFIG_IDX only"
        python run_experiments_ofdm.py --config_idx "$CONFIG_IDX"
    else
        python run_experiments_ofdm.py
    fi

    echo ""
    echo "===== OFDM Training/evaluation complete : $(date) ========"
fi

# =============================================================================
# Evaluation
# =============================================================================
if [ "$EVAL_ONLY" = true ]; then
    echo ""
    echo "===== OFDM Evaluation started : $(date) ==================="
    if [ "$BPSK_QPSK" = true ]; then
        python run_experiments_ofdm.py --eval_only --config_idx 0
        python run_experiments_ofdm.py --eval_only --config_idx 1
    elif [ "$CONFIG_IDX" -ge 0 ] 2>/dev/null; then
        python run_experiments_ofdm.py --eval_only --config_idx "$CONFIG_IDX"
    else
        python run_experiments_ofdm.py --eval_only
    fi
    echo ""
    echo "===== OFDM Evaluation complete : $(date) =================="
    echo ""
    echo "  Figures saved to figures/:"
    ls -lh figures/ofdm_*.* 2>/dev/null || echo "  (no ofdm figures yet)"
fi

echo ""
echo "===== Job finished : $(date) ================================="
