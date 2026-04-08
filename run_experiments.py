"""
Train all models needed to replicate Figure 4 of the DEFINED paper.

For each of the 6 experimental configurations (Figure 4a-f), trains:
  1. A DEFINED model  — two-phase: ICL pre-training + DFE fine-tuning
  2. An ICL-only model — teacher forcing only (no DFE phase)

Usage — all configs sequentially:
    python run_experiments.py

Usage — single config index (0-5), for use as a SLURM job array element:
    python run_experiments.py --config_idx 2
"""

import os
import argparse
import torch
from types import SimpleNamespace

# Disable WandB before any train/wandb imports (can override with --use_wandb)
os.environ.setdefault("WANDB_MODE", "disabled")

from data import build_joint_constellation                              # noqa: E402
from train import build_model, trainNetwork, plot_training_history     # noqa: E402


# ── Experiment configurations matching Figure 4 panels ────────────────────────
# Fields: MIMO size, modulation, fixed eval SNR, pilot count k,
#         and the SNR training range (centred on eval SNR ± 5 dB).
EXP_CONFIGS = [
    dict(num_ant=1, modulation="BPSK",  eval_snr=5,  pilot_len=1, snr_min=0,  snr_max=10),   # (a)
    dict(num_ant=1, modulation="QPSK",  eval_snr=10, pilot_len=1, snr_min=5,  snr_max=15),   # (b)
    dict(num_ant=1, modulation="16QAM", eval_snr=20, pilot_len=1, snr_min=15, snr_max=25),   # (c)
    dict(num_ant=1, modulation="64QAM", eval_snr=25, pilot_len=1, snr_min=20, snr_max=30),   # (d)
    dict(num_ant=2, modulation="BPSK",  eval_snr=10, pilot_len=2, snr_min=5,  snr_max=15),   # (e)
    dict(num_ant=2, modulation="QPSK",  eval_snr=15, pilot_len=2, snr_min=10, snr_max=20),   # (f)
]

# ── Shared hyperparameters (Section IV-A of the paper) ────────────────────────
TRANSFORMER_CFG = dict(num_layer=8, num_head=8, embedding_dim=64)

# Settings that are the same for every modulation.
SHARED_CFG = dict(
    prompt_seq_length=31,
    batch_size=512,
    learning_rate=1e-4,
    loss_weight=0.7,   # α in Eq. (4): L = α·L_DF + (1-α)·L_ICL
)

# ── Per-constellation training configuration ──────────────────────────────────
# All per-modulation settings are explicit here so they are easy to compare
# and adjust without hunting through multiple dicts.
#
# epochs          : total training epochs
# DFE_epoch       : hard upper bound — switch to DFE at this epoch at the latest
#                   (also the fixed switch point when adaptive_dfe=False)
# adaptive_dfe    : detect ICL plateau automatically and switch early;
#                   DFE_epoch still acts as a fallback
# dfe_min_epochs  : minimum ICL epochs before plateau detection begins
# dfe_patience    : consecutive validation checks (each 10 epochs) below
#                   dfe_min_delta improvement before triggering DFE switch
# dfe_min_delta        : minimum relative SER improvement to reset the ICL plateau counter
#
# early_stop_patience  : consecutive DFE-phase validation checks below
#                        early_stop_min_delta improvement before stopping training
# early_stop_min_delta : minimum relative SER improvement to reset the DFE plateau counter
#
# curriculum      : progressively increase context length during ICL pre-training
# curr_start_len  : initial context length
# curr_step_size  : positions added to context length per step
# curr_step_epochs: epochs between each context length increase
#
# Context length growth: starts at curr_start_len, reaches prompt_seq_length (31)
# after ceil((31 - curr_start_len) / curr_step_size) steps.
CONSTELLATION_CFG = {
    "BPSK": dict(
        # Training schedule
        epochs=8000,
        DFE_epoch=2500,
        # Adaptive DFE switching
        adaptive_dfe=True,
        dfe_min_epochs=1000,
        dfe_patience=10,
        dfe_min_delta=5e-4,
        # DFE-phase early stopping
        early_stop_patience=15,
        early_stop_min_delta=1e-4,
        # Curriculum: start=4, +4 every 100 epochs → full length ~8 steps ≈ 700 epochs
        curriculum=True,
        curr_start_len=4,
        curr_step_size=4,
        curr_step_epochs=100,
    ),
    "QPSK": dict(
        # Training schedule
        epochs=8000,
        DFE_epoch=2500,
        # Adaptive DFE switching
        adaptive_dfe=True,
        dfe_min_epochs=1000,
        dfe_patience=10,
        dfe_min_delta=5e-4,
        # DFE-phase early stopping
        early_stop_patience=15,
        early_stop_min_delta=1e-4,
        # Curriculum: start=4, +3 every 100 epochs → full length ~9 steps ≈ 900 epochs
        curriculum=True,
        curr_start_len=4,
        curr_step_size=3,
        curr_step_epochs=100,
    ),
    "16QAM": dict(
        # Training schedule
        epochs=20000,
        DFE_epoch=10000,
        # Adaptive DFE switching
        adaptive_dfe=True,
        dfe_min_epochs=2000,
        dfe_patience=10,
        dfe_min_delta=5e-4,
        # DFE-phase early stopping
        early_stop_patience=15,
        early_stop_min_delta=1e-4,
        # Curriculum: start=4, +2 every 250 epochs → full length ~14 steps ≈ 1750 epochs
        curriculum=True,
        curr_start_len=4,
        curr_step_size=2,
        curr_step_epochs=250,
    ),
    "64QAM": dict(
        # Training schedule
        epochs=30000,
        DFE_epoch=15000,
        # Adaptive DFE switching
        adaptive_dfe=True,
        dfe_min_epochs=9000,
        dfe_patience=10,
        dfe_min_delta=5e-4,
        # DFE-phase early stopping
        early_stop_patience=15,
        early_stop_min_delta=1e-4,
        # Curriculum: start=2, +1 every 300 epochs → full length ~29 steps ≈ 8700 epochs
        curriculum=True,
        curr_start_len=2,
        curr_step_size=1,
        curr_step_epochs=300,
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_args(cfg: dict, dfe_train: bool) -> SimpleNamespace:
    """Build a complete args namespace for one training run."""
    joint = build_joint_constellation(cfg["modulation"], cfg["num_ant"])
    return SimpleNamespace(
        num_ant=cfg["num_ant"],
        modulation=cfg["modulation"],
        SNR_dB_min=cfg["snr_min"],
        SNR_dB_max=cfg["snr_max"],
        train_pilot_len=cfg["pilot_len"],
        DFE_TRAIN=dfe_train,
        modu_num=joint.shape[0],   # will be overwritten inside trainNetwork
        **TRANSFORMER_CFG,
        **SHARED_CFG,
        **CONSTELLATION_CFG[cfg["modulation"]],
    )


def checkpoint_path(cfg: dict, dfe_train: bool) -> str:
    """Return the canonical path for a final model checkpoint."""
    prefix = "DEFINED" if dfe_train else "ICL"
    return (
        f"models/{prefix}_{cfg['num_ant']}ant_{cfg['modulation']}"
        f"_SNR{cfg['snr_min']}-{cfg['snr_max']}_final.pth"
    )


def task_name(cfg: dict, dfe_train: bool) -> str:
    prefix = "DEFINED" if dfe_train else "ICL"
    return (
        f"{prefix}_ant{cfg['num_ant']}_{cfg['modulation']}"
        f"_SNR[{cfg['snr_min']},{cfg['snr_max']}]"
        f"_Seq{TRAIN_CFG['prompt_seq_length']}"
        f"_Layer{TRANSFORMER_CFG['num_layer']}"
        f"Emb{TRANSFORMER_CFG['embedding_dim']}"
        f"Head{TRANSFORMER_CFG['num_head']}"
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(cfg: dict, dfe_train: bool, device: torch.device):
    """
    Train a single model and save its final state_dict.

    If a final checkpoint already exists for this config, the run is skipped
    so that re-running the script after a partial failure is safe.
    """
    out_path = checkpoint_path(cfg, dfe_train)
    if os.path.exists(out_path):
        print(f"  [skip] checkpoint already exists: {out_path}")
        return

    args = make_args(cfg, dfe_train)
    name = task_name(cfg, dfe_train)

    print(f"\n{'='*64}")
    print(f"  Training : {name}")
    print(f"{'='*64}")

    model = build_model(
        embedding_dim=args.embedding_dim,
        n_positions=args.prompt_seq_length,
        num_heads=args.num_head,
        num_layers=args.num_layer,
        num_classes=args.modu_num,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    history = trainNetwork(model, args, task_name=name, device=device)

    # Save training history plot
    plot_training_history(history, task_name=name)

    # Save the model state at the end of training.
    # trainNetwork also saves periodic checkpoints every 200 epochs to ./models/;
    # this separate "final" file gives evaluate.py a stable, known filename.
    os.makedirs("models", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "config": cfg,
            "dfe_train": dfe_train,
        },
        out_path,
    )
    print(f"  Saved final checkpoint: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train DEFINED paper replication models."
    )
    parser.add_argument(
        "--config_idx", type=int, default=-1,
        help=(
            "Which experiment config to run (0-5). "
            "Default -1 runs all configs sequentially."
        ),
    )
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Enable WandB logging (disabled by default).",
    )
    cli = parser.parse_args()

    if cli.use_wandb:
        os.environ.pop("WANDB_MODE", None)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = (
        [EXP_CONFIGS[cli.config_idx]] if 0 <= cli.config_idx < len(EXP_CONFIGS)
        else EXP_CONFIGS
    )

    for cfg in configs:
        train_one(cfg, dfe_train=True,  device=device)   # DEFINED model
        train_one(cfg, dfe_train=False, device=device)   # ICL-only baseline

    print("\nAll training runs complete.")


if __name__ == "__main__":
    main()
