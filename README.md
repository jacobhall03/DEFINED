## DEFINED — Decision Feedback In-Context Symbol Detection

Implementation of the DEFINED framework for in-context learning (ICL) based
MIMO symbol detection, extended to support both flat-fading and OFDM
frequency-selective channels.

Original paper:

> *Decision Feedback In-Context Symbol Detection over Block-Fading Channels*
> L. Fan, J. Yang, C. Shen — IEEE ICC 2025

See `papers/` for the relevant references.

---

### Repository Layout

```
DEFINED/
├── channels/                  # Channel model abstraction
│   ├── __init__.py            # build_channel() factory
│   ├── base.py                # ChannelModel abstract base class
│   ├── flat_fading.py         # Rayleigh / Rician flat-fading
│   └── ofdm.py                # OFDM frequency-selective multipath
│
├── data/                      # Data pipeline (channel-agnostic)
│   ├── __init__.py            # Re-exports all public symbols
│   ├── modulation.py          # Symbol generation, constellation construction
│   ├── encoding.py            # One-hot encoding, complex→real, LMMSE/ML detection
│   └── dataset.py             # MIMOSequenceDataset (PyTorch Dataset)
│
├── scripts/                   # Shell scripts for SLURM cluster runs
│   ├── setup_env.sh           # One-time conda environment setup
│   ├── slurm_flat_fading.sh   # Flat-fading paper replication job
│   └── slurm_ofdm.sh          # OFDM experiments job
│
├── papers/                    # Reference PDFs
│
├── model.py                   # GPT-2 based TransformerModel
├── train.py                   # Training loop (ICL + DEFINED two-phase)
├── config.py                  # CLI argument definitions
├── baseline.py                # Classical LMMSE / DFE-MMSE baselines
├── evaluate.py                # Figure 4 reproduction (flat-fading)
├── run_experiments.py         # Flat-fading paper replication runner
└── run_experiments_ofdm.py    # OFDM experiment runner
```

---

### Channel Models

#### Flat-Fading (`channel_type=flat_fading`)

A single `(num_ant × num_ant)` complex Gaussian channel matrix H is drawn
per sample and applied uniformly across all T time steps — the block-fading
assumption of the original paper.

| Parameter | Default | Description |
|---|---|---|
| `--fading_type` | `rayleigh` | `rayleigh` or `rician` |
| `--K_factor` | `1.0` | Rician K-factor |

#### OFDM Frequency-Selective (`channel_type=ofdm`)

A multipath channel with L exponentially decaying taps is generated, then
converted to K per-sub-carrier channel matrices via the FFT.  Each transformer
sequence position corresponds to one sub-carrier (T = K).

| Parameter | Default | Description |
|---|---|---|
| `--num_subcarriers` | `64` | Number of OFDM sub-carriers (= T) |
| `--num_taps` | `8` | Number of CIR multipath taps |
| `--delay_spread` | `2.0` | Exponential decay constant for tap powers |

---

### Running Experiments

#### 1. Environment Setup (once per cluster account)

```bash
# Get an interactive GPU node first:
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=1:00:00 --pty bash
bash scripts/setup_env.sh
```

#### 2. Flat-Fading Paper Replication (Figure 4)

```bash
# Single job — all 6 configs sequentially (~18–22 hrs on one A100):
sbatch scripts/slurm_flat_fading.sh

# Job array — 6 configs in parallel (~3–4 hrs wall time):
TRAIN_JID=$(sbatch --parsable --array=0-5 scripts/slurm_flat_fading.sh --array)
sbatch --dependency=afterok:$TRAIN_JID scripts/slurm_flat_fading.sh --eval-only
```

Output: `figures/figure4.pdf`, `figures/figure4.png`

#### 3. OFDM Experiments

```bash
# Single job — all 5 configs sequentially:
sbatch scripts/slurm_ofdm.sh

# Job array — 5 configs in parallel:
TRAIN_JID=$(sbatch --parsable --array=0-4 scripts/slurm_ofdm.sh --array)
sbatch --dependency=afterok:$TRAIN_JID scripts/slurm_ofdm.sh --eval-only

# Evaluate previously trained checkpoints:
sbatch scripts/slurm_ofdm.sh --eval-only
```

Output: `figures/ofdm_ser_<config>.pdf/png` per config

#### 4. Local / Single-Config Runs

```bash
# Flat-fading, config 0 only:
python run_experiments.py --config_idx 0

# OFDM, config 1 only:
python run_experiments_ofdm.py --config_idx 1

# Evaluate after training:
python evaluate.py
python run_experiments_ofdm.py --eval_only
```

---

### Architecture

The model is a GPT-2 transformer that performs **in-context learning** for
symbol detection.  At each time step t, the model is given the interleaved
sequence `[y_0, x_0, y_1, x_1, …, y_{t-1}, x_{t-1}, y_t]` and predicts
the transmitted symbol `x_t` by attending over all prior context pairs.

- `y_t` — received signal: `[Re(y), Im(y)]`, shape `(2 * num_ant,)`
- `x_t` — transmitted symbol: one-hot over the joint constellation, shape `(M^num_ant,)`

The two-phase training (ICL → DEFINED) progressively shifts from teacher-forced
context (ICL) to decision-feedback context (DEFINED), improving robustness
at inference time when true labels are unavailable.

---

### Key Design: Channel Abstraction

All channel-specific logic lives in `channels/`.  The dataset, training loop,
and model depend only on the `ChannelModel` interface:

```python
x_seq, y_seq, snr_db = channel.generate(args, rng)
# x_seq: (T, num_ant) complex
# y_seq: (T, num_ant) complex
```

Adding a new channel type requires only a new file in `channels/` and a line
in `channels/__init__.py:build_channel()`.

---

### Dependencies

- Python 3.10
- PyTorch (CUDA build recommended)
- `transformers` (GPT-2)
- `numpy`, `matplotlib`, `wandb`

Install via `scripts/setup_env.sh` on a SLURM cluster, or manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers wandb matplotlib numpy
```
