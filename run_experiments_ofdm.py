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
    pilot_spacing=8,
)

# Shared training parameters
SHARED_CFG = dict(
    batch_size=128,
    learning_rate=1e-4,
    loss_weight=0.7,
    train_batches_per_epoch=10,
    validation_samples=512,
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
        curr_start_len=16,
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
        curr_start_len=16,
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
        curr_start_len=16,
        curr_step_size=4,
        curr_step_epochs=300,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pilot_indices_for_cfg() -> list[int]:
    K = OFDM_CHANNEL_CFG["num_subcarriers"]
    explicit = OFDM_CHANNEL_CFG.get("pilot_indices")
    if explicit is not None:
        pilots = sorted(set(int(p) for p in explicit))
        if not pilots or pilots[0] < 0 or pilots[-1] >= K:
            raise ValueError(f"pilot_indices must be in [0, {K - 1}]")
        return pilots
    spacing = OFDM_CHANNEL_CFG.get("pilot_spacing")
    if spacing is not None:
        return list(range(0, K, int(spacing)))
    return [0]


def make_args(cfg: dict, dfe_train: bool) -> SimpleNamespace:
    """Build a complete args namespace for one training run."""
    joint = build_joint_constellation(cfg["modulation"], cfg["num_ant"])
    pilot_indices = pilot_indices_for_cfg()
    args = SimpleNamespace(
        num_ant=cfg["num_ant"],
        modulation=cfg["modulation"],
        SNR_dB_min=cfg["snr_min"],
        SNR_dB_max=cfg["snr_max"],
        train_pilot_len=len(pilot_indices),
        pilot_indices=pilot_indices,
        DFE_TRAIN=dfe_train,
        modu_num=joint.shape[0],
        **TRANSFORMER_CFG,
        **SHARED_CFG,
        **OFDM_CHANNEL_CFG,
        **CONSTELLATION_CFG[cfg["modulation"]],
    )
    args.prompt_seq_length = args.num_subcarriers
    return args


def make_eval_args(cfg: dict) -> SimpleNamespace:
    joint = build_joint_constellation(cfg["modulation"], cfg["num_ant"])
    pilot_indices = pilot_indices_for_cfg()
    args = SimpleNamespace(
        num_ant=cfg["num_ant"],
        modulation=cfg["modulation"],
        SNR_dB_min=cfg["eval_snr"],
        SNR_dB_max=cfg["eval_snr"],
        train_pilot_len=len(pilot_indices),
        pilot_indices=pilot_indices,
        modu_num=joint.shape[0],
        **TRANSFORMER_CFG,
        **SHARED_CFG,
        **OFDM_CHANNEL_CFG,
    )
    args.prompt_seq_length = args.num_subcarriers
    return args


def checkpoint_path(cfg: dict, dfe_train: bool) -> str:
    prefix = "DEFINED_OFDM" if dfe_train else "ICL_OFDM"
    num_pilots = len(pilot_indices_for_cfg())
    return (
        f"models/{prefix}_{cfg['num_ant']}ant_{cfg['modulation']}"
        f"_K{OFDM_CHANNEL_CFG['num_subcarriers']}"
        f"_P{num_pilots}"
        f"_SNR{cfg['snr_min']}-{cfg['snr_max']}_final.pth"
    )


def task_name(cfg: dict, dfe_train: bool) -> str:
    prefix = "DEFINED_OFDM" if dfe_train else "ICL_OFDM"
    K = OFDM_CHANNEL_CFG["num_subcarriers"]
    P = len(pilot_indices_for_cfg())
    return (
        f"{prefix}_ant{cfg['num_ant']}_{cfg['modulation']}"
        f"_K{K}_P{P}_SNR[{cfg['snr_min']},{cfg['snr_max']}]"
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
MINI_BATCH = 512
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


def eval_ser_at_snr(mode: str, model, args, snr_db: float, device: torch.device) -> float:
    """Evaluate mean data-subcarrier SER at a fixed SNR over N_EVAL samples."""
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
        subcarrier_indices = batch.get("subcarrier_indices")
        data_mask = batch.get("data_mask")
        if subcarrier_indices is not None:
            subcarrier_indices = subcarrier_indices.to(device)
        if data_mask is not None:
            data_mask = data_mask.to(device)
        if mode == "defined":
            _, mean_err = DEFINED_val(
                model, y, x, eval_args, eval_args.train_pilot_len,
                subcarrier_indices=subcarrier_indices,
                data_mask=data_mask,
            )
        else:
            _, mean_err = icl_val(
                model, y, x, eval_args,
                subcarrier_indices=subcarrier_indices,
                data_mask=data_mask,
            )
        total_errors += mean_err * y.shape[0]
        total_symbols += y.shape[0]

    return total_errors / max(total_symbols, 1)


def eval_context_batched(mode: str, model, args, device: torch.device):
    """Return per-prompt-position SER at the config's fixed evaluation SNR."""
    joint = build_joint_constellation(args.modulation, args.num_ant)
    channel = build_channel(args)
    ds = MIMOSequenceDataset(
        args=args,
        num_samples=N_EVAL,
        channel=channel,
        joint_constellation=joint,
        seed=42,
    )
    loader = DataLoader(ds, batch_size=MINI_BATCH, shuffle=False, num_workers=0)

    counts = np.zeros(args.prompt_seq_length, dtype=np.float64)
    total = 0
    subcarrier_indices = None
    data_mask = None

    for batch in loader:
        y = batch["y"].to(device)
        x = batch["x"].to(device)
        batch_subcarriers = batch.get("subcarrier_indices")
        batch_data_mask = batch.get("data_mask")
        if batch_subcarriers is not None:
            batch_subcarriers = batch_subcarriers.to(device)
            if subcarrier_indices is None:
                subcarrier_indices = batch_subcarriers[0].cpu().numpy()
        if batch_data_mask is not None:
            batch_data_mask = batch_data_mask.to(device)
            if data_mask is None:
                data_mask = batch_data_mask[0].cpu().numpy().astype(bool)

        if mode == "defined":
            err_list, _ = DEFINED_val(
                model, y, x, args, args.train_pilot_len,
                subcarrier_indices=batch_subcarriers,
                data_mask=batch_data_mask,
            )
        else:
            err_list, _ = icl_val(
                model, y, x, args,
                subcarrier_indices=batch_subcarriers,
                data_mask=batch_data_mask,
            )

        bsize = y.shape[0]
        counts += np.array(err_list) * bsize
        total += bsize

    if subcarrier_indices is None:
        subcarrier_indices = np.arange(args.prompt_seq_length)
    if data_mask is None:
        data_mask = np.ones(args.prompt_seq_length, dtype=bool)
        data_mask[:args.train_pilot_len] = False

    return counts / max(total, 1), subcarrier_indices, data_mask


def evaluate_one(cfg: dict, device: torch.device):
    """SER vs SNR sweep for one config, saves a figure."""
    print(f"\n--- Evaluating: {cfg['num_ant']}ant {cfg['modulation']} ---")

    defined_model = load_model(cfg, dfe_train=True,  device=device)
    icl_model     = load_model(cfg, dfe_train=False, device=device)
    args = make_eval_args(cfg)

    defined_sers, icl_sers = [], []
    for snr in SNR_SWEEP:
        d_ser = eval_ser_at_snr("defined", defined_model, args, snr, device)
        i_ser = eval_ser_at_snr("icl",     icl_model,     args, snr, device)
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

    safe = (
        f"{cfg['num_ant']}ant_{cfg['modulation']}"
        f"_K{OFDM_CHANNEL_CFG['num_subcarriers']}_P{len(pilot_indices_for_cfg())}"
    )
    for ext in ("pdf", "png"):
        path = f"figures/ofdm_ser_{safe}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)

    # SER vs causal context length at the config's fixed evaluation SNR.
    print(f"  Context SER at fixed SNR={cfg['eval_snr']} dB ...")
    defined_icl, subcarriers, data_mask = eval_context_batched(
        "icl", defined_model, args, device
    )
    defined_df, _, _ = eval_context_batched(
        "defined", defined_model, args, device
    )
    icl_icl, _, _ = eval_context_batched(
        "icl", icl_model, args, device
    )
    icl_df, _, _ = eval_context_batched(
        "defined", icl_model, args, device
    )

    context_lengths = np.arange(args.prompt_seq_length)
    x = context_lengths[data_mask]
    data_bins = subcarriers[data_mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(x, defined_icl[data_mask], "o-", linewidth=1.5,
                markersize=4, markevery=5, label="DEFINED-ICL")
    ax.semilogy(x, defined_df[data_mask], "D-", linewidth=1.5,
                markersize=4, markevery=5, label="DEFINED-DF")
    ax.semilogy(x, icl_icl[data_mask], "s--", linewidth=1.5,
                markersize=4, markevery=5, label="ICL-ICL")
    ax.semilogy(x, icl_df[data_mask], "^--", linewidth=1.5,
                markersize=4, markevery=5, label="ICL-DF")
    ax.axvline(args.train_pilot_len, color="black", linestyle=":", linewidth=1.2,
               label=f"{args.train_pilot_len} pilots")
    ax.set_xlabel("Causal context length")
    ax.set_ylabel("Symbol Error Rate")
    ax.set_title(
        f"OFDM SER vs Context — {cfg['num_ant']}ant {cfg['modulation']} "
        f"SNR={cfg['eval_snr']} dB, K={args.num_subcarriers}, "
        f"P={args.train_pilot_len}"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    if len(data_bins) <= 32:
        ax2 = ax.secondary_xaxis("top")
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(int(k)) for k in data_bins], fontsize=7)
        ax2.set_xlabel("Data subcarrier bin")

    for ext in ("pdf", "png"):
        path = f"figures/ofdm_context_{safe}.{ext}"
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
