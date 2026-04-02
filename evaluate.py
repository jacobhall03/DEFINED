"""
Evaluate trained models and reproduce Figure 4 from the DEFINED paper.

For each of the 6 experimental configurations, loads the DEFINED and ICL-only
checkpoints, evaluates all six curves shown in the paper, computes the classical
MMSE baselines, and saves a 2×3 subplot figure matching Figure 4.

Output:
    figures/figure4.pdf
    figures/figure4.png

Usage:
    python evaluate.py

Dependencies: run_experiments.py must have been run first.
"""

import os
import copy

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless / SLURM use
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from types import SimpleNamespace

from run_experiments import (
    EXP_CONFIGS,
    TRANSFORMER_CFG,
    TRAIN_CFG,
    checkpoint_path,
)
from data import build_joint_constellation, MIMOSequenceDataset
from train import build_model, icl_val, DEFINED_val
from baseline import calculate_ser, DFE_MMSE_SER


# ── Evaluation settings ───────────────────────────────────────────────────────
N_EVAL_NEURAL   = 80_000    # test samples for neural models  (matches paper)
N_EVAL_BASELINE = 5_000     # test samples for classical baselines (pure Python loop)
MINI_BATCH      = 4_096     # mini-batch size to keep GPU memory manageable
T               = 31        # sequence / block length (paper: T = 31)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_eval_args(cfg: dict) -> SimpleNamespace:
    """Args namespace for evaluation: fixed eval SNR, correct modu_num."""
    joint = build_joint_constellation(cfg["modulation"], cfg["num_ant"])
    return SimpleNamespace(
        num_ant=cfg["num_ant"],
        modulation=cfg["modulation"],
        SNR_dB_min=cfg["eval_snr"],
        SNR_dB_max=cfg["eval_snr"],
        train_pilot_len=cfg["pilot_len"],
        modu_num=joint.shape[0],
        **TRANSFORMER_CFG,
        **TRAIN_CFG,   # includes prompt_seq_length=31
    )


def load_model(cfg: dict, dfe_train: bool, device: torch.device):
    """Load a saved checkpoint and reconstruct the model."""
    path = checkpoint_path(cfg, dfe_train)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run  python run_experiments.py  first."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sa   = ckpt["args"]   # saved args dict

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


def get_eval_data(cfg: dict, n: int, device: torch.device):
    """Generate n test samples at the config's fixed eval SNR (seed=42)."""
    args  = make_eval_args(cfg)
    joint = build_joint_constellation(args.modulation, args.num_ant)
    ds    = MIMOSequenceDataset(
        args=args, num_samples=n,
        joint_constellation=joint,
        channel_type="rayleigh", seed=42,
    )
    loader = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0)
    batch  = next(iter(loader))
    return batch["y"].to(device), batch["x"].to(device)


def eval_batched(mode: str, model, y_full, x_full, args, pilot_len: int) -> np.ndarray:
    """
    Run icl_val or DEFINED_val in mini-batches and return per-position SER
    array of shape (T,), averaged over all N_EVAL_NEURAL samples.

    mode: "icl"     → icl_val   (ground-truth context, *-ICL curves)
          "defined" → DEFINED_val (decision feedback,  *-DF  curves)
    """
    n      = y_full.shape[0]
    counts = np.zeros(T, dtype=np.float64)
    total  = 0

    for start in range(0, n, MINI_BATCH):
        yb    = y_full[start : start + MINI_BATCH]
        xb    = x_full[start : start + MINI_BATCH]
        bsize = yb.shape[0]

        if mode == "icl":
            err_list, _ = icl_val(model, yb, xb, args)
        else:
            err_list, _ = DEFINED_val(model, yb, xb, args, pilot_len)

        counts += np.array(err_list) * bsize
        total  += bsize

    return counts / total   # (T,)  per-position SER


def compute_baselines(cfg: dict, pilot_len: int):
    """
    Returns:
        mmse_pk   – scalar SER for MMSE with k pilots      (horizontal line)
        mmse_p30  – scalar SER for MMSE with 30 pilots     (horizontal line)
        mmse_df   – (T,) per-position SER for MMSE-DF-Pk   (descending curve)
    """
    snr = cfg["eval_snr"]

    mmse_pk  = calculate_ser(
        copy.deepcopy(make_eval_args(cfg)), num_samples=N_EVAL_BASELINE,
        pilot_len=pilot_len, snr_db=snr,
    )
    mmse_p30 = calculate_ser(
        copy.deepcopy(make_eval_args(cfg)), num_samples=N_EVAL_BASELINE,
        pilot_len=30, snr_db=snr,
    )
    mmse_df  = DFE_MMSE_SER(
        copy.deepcopy(make_eval_args(cfg)), num_samples=N_EVAL_BASELINE,
        pilot_len=pilot_len, snr_db=snr,
    )
    return mmse_pk, mmse_p30, mmse_df


# ── Plotting ──────────────────────────────────────────────────────────────────

# Subplot titles from the paper (gain values are from the paper, not recomputed)
_PANEL_TITLES = [
    "(a) SISO BPSK, SNR=5 dB, 1 Pilot\ngain$_{\\mathrm{DF}}$=11.2%",
    "(b) SISO QPSK, SNR=10 dB, 1 Pilot\ngain$_{\\mathrm{DF}}$=11.6%",
    "(c) SISO 16QAM, SNR=20 dB, 1 Pilot\ngain$_{\\mathrm{DF}}$=43.7%",
    "(d) SISO 64QAM, SNR=25 dB, 1 Pilot\ngain$_{\\mathrm{DF}}$=45.6%",
    "(e) MIMO BPSK, SNR=10 dB, 2 Pilots\ngain$_{\\mathrm{DF}}$=16.8%",
    "(f) MIMO QPSK, SNR=15 dB, 2 Pilots\ngain$_{\\mathrm{DF}}$=38.6%",
]


def plot_figure4(all_results: list, save_dir: str = "./figures"):
    os.makedirs(save_dir, exist_ok=True)

    # x-axis: context sequence length 1 … 30
    cl = np.arange(1, T)          # [1, 2, ..., 30]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ax, r, title in zip(axes.flatten(), all_results, _PANEL_TITLES):
        k    = r["pilot_len"]
        cl_k = cl[k - 1:]         # [k, k+1, ..., 30]  — context lengths for DF curves

        # ── Classical baselines ──────────────────────────────────────────────
        ax.axhline(r["mmse_pk"],  color="#888888", linestyle="--", linewidth=1.2,
                   label=f"MMSE-P{k}")
        ax.axhline(r["mmse_p30"], color="#000000", linestyle="--", linewidth=1.2,
                   label="MMSE-P30")
        ax.plot(cl_k, r["mmse_df"][k:], color="pink", linewidth=1.5,
                marker="+", markevery=5, label=f"MMSE-DF-P{k}")

        # ── ICL-only model ───────────────────────────────────────────────────
        ax.plot(cl,   r["icl_icl"][1:], color="green",  linewidth=1.5,
                marker="s", markevery=5, markersize=4, label="ICL-ICL")
        ax.plot(cl_k, r["icl_df"][k:],  color="blue",   linewidth=1.5,
                marker="^", markevery=5, markersize=4, label=f"ICL-DF-P{k}")

        # ── DEFINED model ────────────────────────────────────────────────────
        ax.plot(cl,   r["defined_icl"][1:], color="red",    linewidth=1.5,
                marker="o", markevery=5, markersize=4, label="DEFINED-ICL")
        ax.plot(cl_k, r["defined_df"][k:],  color="purple", linewidth=1.5,
                marker="D", markevery=5, markersize=4, label=f"DEFINED-DF-P{k}")

        ax.set_xlabel("Context Sequence Length", fontsize=9)
        ax.set_ylabel("Symbol Error Rate",       fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, 30])

    fig.suptitle(
        "Figure 4 — DEFINED vs. Baselines  (Fan, Yang, Shen — ICC 2025)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(save_dir, f"figure4.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")

    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    all_results = []

    for idx, cfg in enumerate(EXP_CONFIGS):
        k     = cfg["pilot_len"]
        label = (
            f"({['a','b','c','d','e','f'][idx]}) "
            f"{cfg['num_ant']}ant {cfg['modulation']}  "
            f"SNR={cfg['eval_snr']} dB  k={k}"
        )
        print(f"\n{'─'*64}")
        print(f"  {label}")
        print(f"{'─'*64}")

        # ── Load trained models ──────────────────────────────────────────────
        print("  Loading checkpoints...")
        defined_model = load_model(cfg, dfe_train=True,  device=device)
        icl_model     = load_model(cfg, dfe_train=False, device=device)

        eval_args = make_eval_args(cfg)

        # ── Test data ────────────────────────────────────────────────────────
        print(f"  Generating {N_EVAL_NEURAL:,} test samples at SNR={cfg['eval_snr']} dB...")
        y_eval, x_eval = get_eval_data(cfg, N_EVAL_NEURAL, device)

        # ── Neural curves ────────────────────────────────────────────────────
        print("  DEFINED-ICL ...")
        defined_icl = eval_batched("icl",     defined_model, y_eval, x_eval, eval_args, k)

        print(f"  DEFINED-DF-P{k} ...")
        defined_df  = eval_batched("defined", defined_model, y_eval, x_eval, eval_args, k)

        print("  ICL-ICL ...")
        icl_icl     = eval_batched("icl",     icl_model,     y_eval, x_eval, eval_args, k)

        print(f"  ICL-DF-P{k} ...")
        icl_df      = eval_batched("defined", icl_model,     y_eval, x_eval, eval_args, k)

        # ── Classical baselines ──────────────────────────────────────────────
        print(f"  Baselines (N={N_EVAL_BASELINE:,})...")
        mmse_pk, mmse_p30, mmse_df = compute_baselines(cfg, k)

        # Gain metric from paper: (SER_k - SER_{T-1}) / SER_k × 100 %
        if defined_df[k] > 0:
            gain = (defined_df[k] - defined_df[-1]) / defined_df[k] * 100
            print(f"  DEFINED-DF gain (replicated): {gain:.1f}%")

        all_results.append(
            dict(
                pilot_len   = k,
                defined_icl = defined_icl,
                defined_df  = defined_df,
                icl_icl     = icl_icl,
                icl_df      = icl_df,
                mmse_pk     = mmse_pk,
                mmse_p30    = mmse_p30,
                mmse_df     = mmse_df,
            )
        )

    print("\n  Generating Figure 4...")
    plot_figure4(all_results)
    print("Done.")


if __name__ == "__main__":
    main()
