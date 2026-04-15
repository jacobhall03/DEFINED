"""
Train and evaluate DEFINED for OFDM frequency-selective channels.

Each experiment config defines:
  - MIMO size and modulation
  - Number of OFDM sub-carriers (T = num_subcarriers)
  - Multipath channel parameters (num_taps, delay_spread)
  - SNR training range and fixed evaluation SNR

Usage — all configs sequentially:
    python run_experiments_ofdm.py

Usage — single config index (for SLURM job arrays):
    python run_experiments_ofdm.py --config_idx 2

Usage — evaluate only (checkpoints must already exist):
    python run_experiments_ofdm.py --eval_only
"""

import os
import argparse
import copy

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from types import SimpleNamespace

os.environ.setdefault("WANDB_MODE", "disabled")

from data.modulation import build_joint_constellation
from data.dataset import MIMOSequenceDataset
from channels import build_channel
from train import build_model, trainNetwork, plot_training_history, icl_val, DEFINED_val


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

# Transformer architecture (same as flat-fading paper)
TRANSFORMER_CFG = dict(num_layer=8, num_head=8, embedding_dim=64)

# Shared OFDM channel parameters
OFDM_CHANNEL_CFG = dict(
    channel_type="ofdm",
    num_subcarriers=64,
    num_taps=8,
    delay_spread=2.0,
)

# Shared training parameters
SHARED_CFG = dict(
    batch_size=512,
    learning_rate=1e-4,
    loss_weight=0.7,
    # prompt_seq_length is overridden by channel.seq_length = num_subcarriers
    prompt_seq_length=64,
)

# Per-experiment configs
OFDM_EXP_CONFIGS = [
    dict(num_ant=1, modulation="BPSK",  eval_snr=5,  pilot_len=1,
         snr_min=0,  snr_max=10),   # (a) SISO BPSK
    dict(num_ant=1, modulation="QPSK",  eval_snr=10, pilot_len=1,
         snr_min=5,  snr_max=15),   # (b) SISO QPSK
    dict(num_ant=1, modulation="16QAM", eval_snr=20, pilot_len=1,
         snr_min=15, snr_max=25),   # (c) SISO 16QAM
    dict(num_ant=2, modulation="BPSK",  eval_snr=10, pilot_len=2,
         snr_min=5,  snr_max=15),   # (d) MIMO 2-ant BPSK
    dict(num_ant=2, modulation="QPSK",  eval_snr=15, pilot_len=2,
         snr_min=10, snr_max=20),   # (e) MIMO 2-ant QPSK
]

# Per-modulation training schedule
CONSTELLATION_CFG = {
    "BPSK": dict(
        epochs=8000,
        DFE_epoch=2500,
        adaptive_dfe=True,
        dfe_min_epochs=1000,
        dfe_patience=10,
        dfe_min_delta=5e-4,
        early_stop_patience=15,
        early_stop_min_delta=1e-4,
        curriculum=True,
        curr_start_len=8,
        curr_step_size=8,
        curr_step_epochs=100,
    ),
    "QPSK": dict(
        epochs=10000,
        DFE_epoch=3000,
        adaptive_dfe=True,
        dfe_min_epochs=1000,
        dfe_patience=10,
        dfe_min_delta=5e-4,
        early_stop_patience=15,
        early_stop_min_delta=1e-4,
        curriculum=True,
        curr_start_len=8,
        curr_step_size=6,
        curr_step_epochs=150,
    ),
    "16QAM": dict(
        epochs=25000,
        DFE_epoch=12000,
        adaptive_dfe=True,
        dfe_min_epochs=3000,
        dfe_patience=10,
        dfe_min_delta=5e-4,
        early_stop_patience=15,
        early_stop_min_delta=1e-4,
        curriculum=True,
        curr_start_len=8,
        curr_step_size=4,
        curr_step_epochs=300,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        modu_num=joint.shape[0],
        **TRANSFORMER_CFG,
        **SHARED_CFG,
        **OFDM_CHANNEL_CFG,
        **CONSTELLATION_CFG[cfg["modulation"]],
    )


def make_eval_args(cfg: dict) -> SimpleNamespace:
    joint = build_joint_constellation(cfg["modulation"], cfg["num_ant"])
    return SimpleNamespace(
        num_ant=cfg["num_ant"],
        modulation=cfg["modulation"],
        SNR_dB_min=cfg["eval_snr"],
        SNR_dB_max=cfg["eval_snr"],
        train_pilot_len=cfg["pilot_len"],
        modu_num=joint.shape[0],
        **TRANSFORMER_CFG,
        **SHARED_CFG,
        **OFDM_CHANNEL_CFG,
    )


def checkpoint_path(cfg: dict, dfe_train: bool) -> str:
    prefix = "DEFINED_OFDM" if dfe_train else "ICL_OFDM"
    return (
        f"models/{prefix}_{cfg['num_ant']}ant_{cfg['modulation']}"
        f"_K{OFDM_CHANNEL_CFG['num_subcarriers']}"
        f"_SNR{cfg['snr_min']}-{cfg['snr_max']}_final.pth"
    )


def task_name(cfg: dict, dfe_train: bool) -> str:
    prefix = "DEFINED_OFDM" if dfe_train else "ICL_OFDM"
    K = OFDM_CHANNEL_CFG["num_subcarriers"]
    return (
        f"{prefix}_ant{cfg['num_ant']}_{cfg['modulation']}"
        f"_K{K}_SNR[{cfg['snr_min']},{cfg['snr_max']}]"
        f"_Layer{TRANSFORMER_CFG['num_layer']}"
        f"Emb{TRANSFORMER_CFG['embedding_dim']}"
        f"Head{TRANSFORMER_CFG['num_head']}"
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(cfg: dict, dfe_train: bool, device: torch.device):
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
    plot_training_history(history, task_name=name)

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


# ---------------------------------------------------------------------------
# Evaluation — SER vs SNR sweep
# ---------------------------------------------------------------------------

N_EVAL = 20_000
MINI_BATCH = 2_048
SNR_SWEEP = list(range(0, 31, 5))  # 0, 5, 10, 15, 20, 25, 30 dB


def load_model(cfg: dict, dfe_train: bool, device: torch.device):
    path = checkpoint_path(cfg, dfe_train)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run  python run_experiments_ofdm.py  first."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sa = ckpt["args"]
    num_classes = build_joint_constellation(sa["modulation"], sa["num_ant"]).shape[0]
    model = build_model(
        embedding_dim=sa["embedding_dim"],
        n_positions=sa["prompt_seq_length"],
        num_heads=sa["num_head"],
        num_layers=sa["num_layer"],
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def eval_ser_at_snr(model, args, snr_db: float, device: torch.device) -> float:
    """Evaluate mean ICL SER at a fixed SNR over N_EVAL samples."""
    eval_args = copy.deepcopy(args)
    eval_args.SNR_dB_min = snr_db
    eval_args.SNR_dB_max = snr_db

    joint = build_joint_constellation(eval_args.modulation, eval_args.num_ant)
    channel = build_channel(eval_args)
    ds = MIMOSequenceDataset(
        args=eval_args,
        num_samples=N_EVAL,
        channel=channel,
        joint_constellation=joint,
        seed=42,
    )
    loader = DataLoader(ds, batch_size=MINI_BATCH, shuffle=False, num_workers=0)

    total_errors = 0
    total_symbols = 0
    for batch in loader:
        y = batch["y"].to(device)
        x = batch["x"].to(device)
        _, mean_err = icl_val(model, y, x, eval_args)
        total_errors += mean_err * y.shape[0]
        total_symbols += y.shape[0]

    return total_errors / max(total_symbols, 1)


def evaluate_one(cfg: dict, device: torch.device):
    """SER vs SNR sweep for one config, saves a figure."""
    print(f"\n--- Evaluating: {cfg['num_ant']}ant {cfg['modulation']} ---")

    defined_model = load_model(cfg, dfe_train=True,  device=device)
    icl_model     = load_model(cfg, dfe_train=False, device=device)
    args = make_eval_args(cfg)

    defined_sers, icl_sers = [], []
    for snr in SNR_SWEEP:
        d_ser = eval_ser_at_snr(defined_model, args, snr, device)
        i_ser = eval_ser_at_snr(icl_model,     args, snr, device)
        defined_sers.append(d_ser)
        icl_sers.append(i_ser)
        print(f"  SNR={snr:3d} dB   DEFINED={d_ser:.4f}  ICL={i_ser:.4f}")

    # Plot
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(SNR_SWEEP, defined_sers, "o-", label="DEFINED-OFDM")
    ax.semilogy(SNR_SWEEP, icl_sers,     "s--", label="ICL-OFDM")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Symbol Error Rate")
    ax.set_title(
        f"OFDM SER vs SNR — {cfg['num_ant']}ant {cfg['modulation']} "
        f"K={OFDM_CHANNEL_CFG['num_subcarriers']} sub-carriers"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    safe = f"{cfg['num_ant']}ant_{cfg['modulation']}_K{OFDM_CHANNEL_CFG['num_subcarriers']}"
    for ext in ("pdf", "png"):
        path = f"figures/ofdm_ser_{safe}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate DEFINED for OFDM channels."
    )
    parser.add_argument(
        "--config_idx", type=int, default=-1,
        help="Which config to run (0-indexed). -1 = all.",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training; run SER vs SNR evaluation only.",
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
        [OFDM_EXP_CONFIGS[cli.config_idx]]
        if 0 <= cli.config_idx < len(OFDM_EXP_CONFIGS)
        else OFDM_EXP_CONFIGS
    )

    if not cli.eval_only:
        for cfg in configs:
            train_one(cfg, dfe_train=True,  device=device)
            train_one(cfg, dfe_train=False, device=device)
        print("\nAll OFDM training runs complete.")

    for cfg in configs:
        evaluate_one(cfg, device=device)
    print("\nAll OFDM evaluations complete.")


if __name__ == "__main__":
    main()
