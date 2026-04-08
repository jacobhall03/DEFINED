import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

from model import TransformerModel
from data import (
    count_modulation_symbols,
    build_joint_constellation,
    MIMOSequenceDataset,
)
from config import parameter_reading



def build_model(embedding_dim, n_positions, num_heads, num_layers, num_classes):
    """Build the TransformerModel used for ICL equalization."""
    model = TransformerModel(
        n_positions=2 * n_positions,   # because x/y are interleaved
        n_embd=embedding_dim,
        n_layer=num_layers,
        n_head=num_heads,
        n_classes=num_classes,
    )
    return model



def icl_train(model, ys_batch, xs_batch, optimizer, loss_func, args):
    """One-step ICL training.

    ys_batch: (B, T, dim_y) = real-valued received features [Re(y), Im(y)]
    xs_batch: (B, T, C)     = one-hot ground truth transmit symbols (joint constellation)
    """
    model.train()

    # Forward pass through the model
    logits = model(ys_batch, xs_batch)               # (B, T, C)

    # Targets: class indices from one-hot x
    xs_indices = torch.argmax(xs_batch, dim=-1)      # (B, T)

    # Cross-entropy over classes at each time step
    # loss_func is CrossEntropyLoss(reduction="none") -> (B, T)
    loss_per_token = loss_func(logits.transpose(1, 2), xs_indices)
    loss_mean = loss_per_token.mean()

    optimizer.zero_grad()
    loss_mean.backward()
    optimizer.step()

    return loss_mean.detach().item(), logits.detach()


def DEFINED_train(args, model, ys_batch, xs_batch,
                  optimizer, loss_func, train_pilot_len):
    """Decision Feedback training (DFE), equivalent to original sequence_train_step.

    ys_batch: (B, T, dim_y) = received features
    xs_batch: (B, T, C)     = one-hot ground truth transmit symbols
    """
    torch.autograd.set_detect_anomaly(True)
    model.train()

    bsize, length, dim = xs_batch.shape
    xin = torch.zeros_like(xs_batch)   # feedback sequence
    pilot_len = train_pilot_len

    # Build the decision-feedback sequence xin under no_grad
    with torch.no_grad():
        for i in range(length):
            if i < pilot_len:
                # For pilot positions, we feed the true x
                xin = torch.cat([xin[:, :i, :], xs_batch[:, i:, :]], dim=1)
            else:
                # Run the model with current feedback sequence xin
                # Note: original code uses full ys_batch here (not truncated yin)
                x_hat = model(ys_batch, xin)                    # (B, T, C)
                probabilities = torch.softmax(x_hat, dim=-1)
                _, max_indices = torch.max(probabilities, dim=-1)  # (B, T)

                one_hot_encoded = torch.nn.functional.one_hot(
                    max_indices, num_classes=args.modu_num
                ).float()                                       # (B, T, C)

                # Replace from position i onward with predicted symbols
                xin = torch.cat([xin[:, :i, :], one_hot_encoded[:, i:, :]], dim=1)

    # 1) Loss when using decision-feedback sequence xin
    x_prob1 = model(ys_batch, xin)                  # (B, T, C)
    xs_real_indices = torch.argmax(xs_batch, dim=-1)   # (B, T)
    loss1 = loss_func(x_prob1.transpose(1, 2), xs_real_indices)  # (B, T)

    # 2) Loss when using full teacher forcing xs_batch (pure ICL)
    x_prob2 = model(ys_batch, xs_batch)             # (B, T, C)
    loss2 = loss_func(x_prob2.transpose(1, 2), xs_real_indices)  # (B, T)

    # Weighted combination (same as original)
    weight = args.loss_weight
    total_loss = (weight * loss1 + (1.0 - weight) * loss2).mean()

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.detach().item(), x_prob1.detach()



def icl_val(model, ys_batch, xs_batch, args):
    """Evaluate symbol error rate using pure ICL (no decision feedback)."""
    model.eval()
    with torch.no_grad():
        logits = model(ys_batch, xs_batch)         # (B, T, C)
        pred_indices = torch.argmax(logits, dim=-1)    # (B, T)
        true_indices = torch.argmax(xs_batch, dim=-1)  # (B, T)

        errors = (pred_indices != true_indices)        # (B, T)
        length_errors = errors.float().mean(dim=0)     # (T,)
        mean_errors = length_errors.mean().item()
        length_errors_list = length_errors.cpu().numpy().tolist()

    return length_errors_list, mean_errors



def DEFINED_val(model, ys_batch, xs_batch, args, train_pilot_len):
    """Evaluate sequence error rate with decision feedback (DFE)."""
    model.eval()
    with torch.no_grad():
        bsize, length, dim = xs_batch.shape
        xin = torch.zeros_like(xs_batch)
        yin = torch.zeros_like(ys_batch)
        pilot_len = train_pilot_len

        for i in range(length):
            # Accumulate ys into yin, so the model only "sees" up to time i
            yin[:, i, :] = ys_batch[:, i, :]
            if i < pilot_len:
                xin[:, i, :] = xs_batch[:, i, :]
            else:
                x_hat = model(yin, xin)                    # (B, T, C)
                probabilities = torch.softmax(x_hat, dim=-1)
                _, max_indices = torch.max(probabilities, dim=-1)
                one_hot_encoded = torch.nn.functional.one_hot(
                    max_indices, num_classes=args.modu_num
                ).float()
                xin[:, i, :] = one_hot_encoded[:, i, :]

        # Compare predicted xin with ground truth xs_batch
        errors = (xs_batch != xin).any(dim=2)       # (B, T)
        length_errors = errors.float().mean(dim=0)  # (T,)
        mean_errors = length_errors.mean().item()
        remaining_mean_errors = length_errors[train_pilot_len:].mean().item()
        length_errors_list = length_errors.cpu().numpy().tolist()

    return length_errors_list, remaining_mean_errors



def trainNetwork(model_GPT2, args, task_name, device):
    """Full training loop using MIMOSequenceDataset and ICL/DEFINED training."""
    # ---------------- WandB init ----------------
    wandb.init(
        project="DEFINED",
        name=task_name,
    )

    loss_function_model_GPT2 = nn.CrossEntropyLoss(reduction="none")
    optimizer_model_GPT2 = optim.AdamW(model_GPT2.parameters(), lr=args.learning_rate)

    # ---------------- Build joint constellation & datasets ----------------
    # For SISO (num_ant=1), joint_constellation size == args.modu_num
    joint_constellation = build_joint_constellation(args.modulation, args.num_ant)
    num_classes = joint_constellation.shape[0]
    args.modu_num = num_classes  # ensure consistency with joint constellation size

    n_train = int(1e3)
    # Train dataset: large "infinite" style dataset
    train_dataset = MIMOSequenceDataset(
        args=args,
        num_samples=n_train,
        joint_constellation=joint_constellation,
        channel_type="rayleigh",
        K_factor=1.0,
        seed=0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # Validation dataset: fixed samples for stable evaluation
    n_val = int(2e3)
    val_dataset = MIMOSequenceDataset(
        args=args,
        num_samples=n_val,
        joint_constellation=joint_constellation,
        channel_type="rayleigh",
        K_factor=1.0,
        seed=123,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=n_val,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    # Take one big validation batch
    val_batch = next(iter(val_loader))
    y_val = val_batch["y"].to(device)  # (B_val, T, 2N)
    x_val = val_batch["x"].to(device)  # (B_val, T, C)

    # ---------------- Training config ----------------
    n_it_per_epoch = 10
    log_every = 10

    resume_path = os.path.join("./models", f"{task_name}_resume.pth")
    os.makedirs("./models", exist_ok=True)

    best_val = 1e9
    best_it = 0

    # ── Adaptive DFE switching setup ──────────────────────────────────────────
    # Read adaptive args with defaults so old args objects stay compatible.
    adaptive_dfe   = getattr(args, 'adaptive_dfe',   False)
    dfe_patience   = getattr(args, 'dfe_patience',   10)
    dfe_min_delta  = getattr(args, 'dfe_min_delta',  5e-4)
    dfe_min_epochs = getattr(args, 'dfe_min_epochs', 1000)

    # ── DFE-phase early stopping setup ───────────────────────────────────────
    # Mirrors the ICL plateau detector but fires during DFE fine-tuning.
    # Uses validation SER (not loss) so it is immune to the lull period and
    # consistent with the adaptive switching logic above.
    early_stop_patience  = getattr(args, 'early_stop_patience',  15)
    early_stop_min_delta = getattr(args, 'early_stop_min_delta', 1e-4)
    dfe_plateau_count    = 0
    best_dfe_ser         = float('inf')

    # ── Curriculum learning setup ─────────────────────────────────────────────
    # During ICL pre-training, context length grows from curr_start_len to
    # prompt_seq_length, increasing by curr_step_size every curr_step_epochs.
    curriculum       = getattr(args, 'curriculum',        False)
    curr_start_len   = getattr(args, 'curr_start_len',    args.prompt_seq_length)
    curr_step_size   = getattr(args, 'curr_step_size',    1)
    curr_step_epochs = getattr(args, 'curr_step_epochs',  100)

    curr_seq_len = curr_start_len if curriculum else args.prompt_seq_length
    if curriculum:
        print(f"*** Curriculum: start_len={curr_start_len}, step={curr_step_size} "
              f"every {curr_step_epochs} epochs → full len={args.prompt_seq_length}")

    # effective_dfe_epoch tracks when the ICL→DFE switch actually happens.
    # In fixed mode it equals args.DFE_epoch throughout.
    # In adaptive mode it may be updated earlier when a plateau is detected;
    # args.DFE_epoch remains a hard fallback upper bound.
    effective_dfe_epoch = args.DFE_epoch
    best_icl_ser        = float('inf')   # tracks ICL-phase val SER for plateau detection
    icl_plateau_count   = 0
    dfe_phase_started   = False          # used to print the phase-switch message once

    if args.DFE_TRAIN:
        mode = "adaptive" if adaptive_dfe else "fixed"
        print(f"*** Start DFE Train ({mode} switch): {task_name}")
    else:
        print(f"*** Start ICL Train: {task_name}")

    # ── Training history (for post-run plots) ─────────────────────────────────
    history = {
        "train_loss":         [],   # (epoch, loss)
        "val_ser":            [],   # (epoch, ser)
        "curriculum_changes": [],   # (epoch, new_len)  — curriculum length step-ups
        "dfe_switch_epoch":   None, # epoch where DFE phase began
        "early_stop_epoch":   None, # epoch where DFE early stopping fired (None = ran to completion)
    }

    # ── Resume from crash checkpoint if one exists ────────────────────────────
    start_epoch = 0
    if os.path.exists(resume_path):
        print(f"*** Resuming from crash checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model_GPT2.load_state_dict(ckpt["model_state_dict"])
        optimizer_model_GPT2.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch          = ckpt["epoch"] + 1
        best_val             = ckpt["best_val"]
        best_it              = ckpt["best_it"]
        effective_dfe_epoch  = ckpt["effective_dfe_epoch"]
        best_icl_ser         = ckpt["best_icl_ser"]
        icl_plateau_count    = ckpt["icl_plateau_count"]
        curr_seq_len         = ckpt["curr_seq_len"]
        history              = ckpt["history"]
        dfe_phase_started    = ckpt["dfe_phase_started"]
        dfe_plateau_count    = ckpt.get("dfe_plateau_count", 0)
        best_dfe_ser         = ckpt.get("best_dfe_ser", float('inf'))
        print(f"*** Resumed at epoch {start_epoch} / {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0

        # Determine phase once per epoch (before any updates this epoch).
        in_icl_phase = args.DFE_TRAIN and (epoch < effective_dfe_epoch)

        # ── Update curriculum sequence length ─────────────────────────────────
        if curriculum and in_icl_phase:
            steps_done   = epoch // curr_step_epochs
            new_seq_len  = min(curr_start_len + steps_done * curr_step_size,
                               args.prompt_seq_length)
            if new_seq_len != curr_seq_len:
                curr_seq_len = new_seq_len
                history["curriculum_changes"].append((epoch, curr_seq_len))
                print(f"*** Curriculum: context length → {curr_seq_len} at epoch {epoch}")
        elif not in_icl_phase:
            curr_seq_len = args.prompt_seq_length   # DFE phase always uses full length

        # One-time message when DFE phase begins.
        if args.DFE_TRAIN and not in_icl_phase and not dfe_phase_started:
            dfe_phase_started = True
            history["dfe_switch_epoch"] = epoch
            print(f"*** DFE fine-tuning phase started at epoch {epoch} "
                  f"({'adaptive plateau' if adaptive_dfe else 'fixed schedule'})")

        # One epoch: take n_it_per_epoch mini-batches
        for it, batch in enumerate(train_loader):
            if it >= n_it_per_epoch:
                break

            ys_batch = batch["y"].to(device)[:, :curr_seq_len, :]  # (B, L, 2N)
            xs_batch = batch["x"].to(device)[:, :curr_seq_len, :]  # (B, L, C)

            if in_icl_phase or not args.DFE_TRAIN:
                loss, output = icl_train(
                    model_GPT2,
                    ys_batch=ys_batch,
                    xs_batch=xs_batch,
                    optimizer=optimizer_model_GPT2,
                    loss_func=loss_function_model_GPT2,
                    args=args,
                )
            else:
                loss, output = DEFINED_train(
                    args,
                    model_GPT2,
                    ys_batch=ys_batch,
                    xs_batch=xs_batch,
                    optimizer=optimizer_model_GPT2,
                    loss_func=loss_function_model_GPT2,
                    train_pilot_len=args.train_pilot_len,
                )

            running_loss += loss / n_it_per_epoch

        history["train_loss"].append((epoch, running_loss))
        # Log training loss
        wandb.log({"Train: Cross-Entropy Loss": running_loss, "epoch": epoch})

        # ---------------- Validation ----------------
        if epoch % log_every == 0:
            # During ICL pre-training use icl_val — it gives a clean convergence
            # signal uncontaminated by feedback errors, which is also what the
            # plateau detector needs.  Switch to DEFINED_val once DFE starts.
            # Truncate validation to curr_seq_len so the plateau detector measures
            # SER at the same length the model is currently being trained on.
            if in_icl_phase or not args.DFE_TRAIN:
                _, mean_errors = icl_val(
                    model_GPT2,
                    ys_batch=y_val[:, :curr_seq_len, :],
                    xs_batch=x_val[:, :curr_seq_len, :],
                    args=args,
                )
            else:
                _, mean_errors = DEFINED_val(
                    model_GPT2,
                    ys_batch=y_val,
                    xs_batch=x_val,
                    args=args,
                    train_pilot_len=args.train_pilot_len,
                )

            history["val_ser"].append((epoch, mean_errors))
            wandb.log({"Test: Symbol Error Rate": mean_errors, "epoch": epoch})

            if mean_errors < best_val:
                best_val = mean_errors
                best_it = epoch

            # ── Adaptive plateau detection (ICL phase only) ───────────────
            if adaptive_dfe and args.DFE_TRAIN and in_icl_phase and epoch >= dfe_min_epochs:
                rel_improvement = (best_icl_ser - mean_errors) / max(best_icl_ser, 1e-10)
                if rel_improvement > dfe_min_delta:
                    best_icl_ser      = mean_errors
                    icl_plateau_count = 0
                else:
                    icl_plateau_count += 1

                if icl_plateau_count >= dfe_patience:
                    effective_dfe_epoch = epoch + 1   # switch next epoch
                    icl_plateau_count   = 0
                    print(f"*** Adaptive DFE: plateau detected at epoch {epoch} "
                          f"(val SER={mean_errors:.4e}), switching to DFE at epoch "
                          f"{effective_dfe_epoch}")
            elif in_icl_phase and mean_errors < best_icl_ser:
                best_icl_ser = mean_errors

            # ── DFE-phase early stopping ──────────────────────────────────
            if args.DFE_TRAIN and not in_icl_phase:
                rel_improvement = (best_dfe_ser - mean_errors) / max(best_dfe_ser, 1e-10)
                if rel_improvement > early_stop_min_delta:
                    best_dfe_ser      = mean_errors
                    dfe_plateau_count = 0
                else:
                    dfe_plateau_count += 1

                if dfe_plateau_count >= early_stop_patience:
                    history["early_stop_epoch"] = epoch
                    print(f"*** Early stop: DFE plateau at epoch {epoch} "
                          f"(val SER={mean_errors:.4e}, no improvement for "
                          f"{early_stop_patience} checks), stopping training.")
                    break

        # ---------------- Overwrite single resume checkpoint every 200 epochs ----------------
        if epoch % 200 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_GPT2.state_dict(),
                    "optimizer_state_dict": optimizer_model_GPT2.state_dict(),
                    "args": vars(args),
                    "best_val": best_val,
                    "best_it": best_it,
                    "effective_dfe_epoch": effective_dfe_epoch,
                    "best_icl_ser": best_icl_ser,
                    "icl_plateau_count": icl_plateau_count,
                    "curr_seq_len": curr_seq_len,
                    "dfe_phase_started": dfe_phase_started,
                    "dfe_plateau_count": dfe_plateau_count,
                    "best_dfe_ser": best_dfe_ser,
                    "history": history,
                },
                resume_path,
            )

    print(f"*** Best validation SER: {best_val:.4e} at epoch {best_it}")
    if os.path.exists(resume_path):
        os.remove(resume_path)
    wandb.finish()
    return history


# -------------------------------------------------------------------------- #
# Training history plot
# -------------------------------------------------------------------------- #

def plot_training_history(history: dict, task_name: str, save_dir: str = "./figures"):
    """
    Save a two-panel figure showing cross-entropy loss and validation SER vs epoch.

    Vertical lines mark:
      - Each curriculum context-length increase (dashed blue, labelled with new length)
      - The ICL → DFE phase switch (solid red, labelled)
    """
    os.makedirs(save_dir, exist_ok=True)

    loss_epochs, loss_vals = zip(*history["train_loss"]) if history["train_loss"] else ([], [])
    ser_epochs,  ser_vals  = zip(*history["val_ser"])    if history["val_ser"]     else ([], [])

    loss_epochs = np.array(loss_epochs)
    loss_vals   = np.array(loss_vals)
    ser_epochs  = np.array(ser_epochs)
    ser_vals    = np.array(ser_vals)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # ── Panel 1: cross-entropy loss ──────────────────────────────────────────
    ax1.plot(loss_epochs, loss_vals, color="steelblue", linewidth=0.8, label="Train CE Loss")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # ── Panel 2: validation SER ──────────────────────────────────────────────
    ax2.semilogy(ser_epochs, ser_vals, color="darkorange", linewidth=0.8, label="Val SER")
    ax2.set_ylabel("Symbol Error Rate (log scale)")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(loc="upper right", fontsize=8)

    # ── Vertical lines ───────────────────────────────────────────────────────
    # Collect all x positions so we can stagger label heights to avoid overlap
    all_vlines = []

    for ep, new_len in history["curriculum_changes"]:
        all_vlines.append(("curriculum", ep, new_len))

    dfe_ep = history.get("dfe_switch_epoch")
    if dfe_ep is not None:
        all_vlines.append(("dfe", dfe_ep, None))

    es_ep = history.get("early_stop_epoch")
    if es_ep is not None:
        all_vlines.append(("early_stop", es_ep, None))

    # Draw lines on both axes; stagger label y-positions on ax1
    label_y_positions = np.linspace(0.92, 0.60, max(len(all_vlines), 1))

    for idx, (kind, ep, val) in enumerate(all_vlines):
        if kind == "curriculum":
            color, ls = "royalblue", "--"
            label     = f"len→{val}"
        elif kind == "dfe":
            color, ls = "crimson", "-"
            label     = "ICL→DFE"
        else:  # early_stop
            color, ls = "darkorange", "-"
            label     = "early stop"

        for ax in (ax1, ax2):
            ax.axvline(ep, color=color, linestyle=ls, linewidth=0.9, alpha=0.7)

        # Label only on ax1
        ax1.text(
            ep, label_y_positions[idx], label,
            transform=ax1.get_xaxis_transform(),
            color=color, fontsize=7, rotation=90,
            va="top", ha="right",
        )

    fig.suptitle(task_name, fontsize=9, fontweight="bold")
    plt.tight_layout()

    safe_name = task_name.replace("/", "_").replace(" ", "_")
    for ext in ("pdf", "png"):
        path = os.path.join(save_dir, f"training_{safe_name}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved training plot: {path}")

    plt.close(fig)


# -------------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    args = parameter_reading()

    # Model / training hyperparameters
    args.num_head = 8
    args.num_layer = 8
    args.embedding_dim = 64
    args.embedding_dim_single = 64

    args.prompt_seq_length = 31
    args.batch_size = 512
    args.epochs = 8000

    args.num_ant = 2

    args.modulation = "QPSK"
    args.train_pilot_len = 2
    args.SNR_dB_min = 10
    args.SNR_dB_max = 20
    args.modu_num = count_modulation_symbols(args)  # initial value, later overwritten by joint constellation size
    args.loss_weight = 0.7
    args.DFE_epoch = 2500
    args.DFE_TRAIN = True  # enable DFE two-phase training

    task_name = (
        "DEFINED_ant"
        + str(args.num_ant)
        + "_"
        + args.modulation
        + "_SNR["
        + str(args.SNR_dB_min)
        + ","
        + str(args.SNR_dB_max)
        + "]"
        + "_Seq"
        + str(args.prompt_seq_length)
        + "_Layer"
        + str(args.num_layer)
        + "Emb"
        + str(args.embedding_dim)
        + "Head"
        + str(args.num_head)
    )

    model = build_model(
        embedding_dim=args.embedding_dim,
        n_positions=args.prompt_seq_length,
        num_heads=args.num_head,
        num_layers=args.num_layer,
        num_classes=args.modu_num,
    ).to(device)

    print(args)

    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"***Total number of parameters in the model: {total_trainable_params}")

    trainNetwork(model.to(device), args, task_name=task_name, device=device)

    print("***Training is done:", task_name)