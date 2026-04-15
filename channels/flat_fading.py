"""Flat-fading MIMO channel models: Rayleigh and Rician.

This module also provides ``generate_signals``, a convenience batch generator
used by the classical (non-neural) baselines in ``baseline.py``.
"""

from typing import List, Tuple

import numpy as np

from .base import ChannelModel
from data.modulation import generate_modulated_signal


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _draw_rayleigh(num_ant: int, rng: np.random.Generator) -> np.ndarray:
    """Return one (num_ant, num_ant) Rayleigh fading matrix."""
    return (
        rng.standard_normal((num_ant, num_ant))
        + 1j * rng.standard_normal((num_ant, num_ant))
    ) / np.sqrt(2)


def _draw_rician(num_ant: int, K_factor: float, rng: np.random.Generator) -> np.ndarray:
    """Return one (num_ant, num_ant) Rician fading matrix.

    The LOS component is a constant all-ones matrix (normalised) and the
    scatter component is i.i.d. complex Gaussian, weighted by the K-factor.
    """
    H_scatter = (
        rng.standard_normal((num_ant, num_ant))
        + 1j * rng.standard_normal((num_ant, num_ant))
    ) / np.sqrt(2)

    H_los = (
        np.ones((num_ant, num_ant)) + 1j * np.ones((num_ant, num_ant))
    ) / np.sqrt(2)

    return (
        np.sqrt(K_factor / (K_factor + 1.0)) * H_los
        + np.sqrt(1.0 / (K_factor + 1.0)) * H_scatter
    )


# ---------------------------------------------------------------------------
# Channel model class
# ---------------------------------------------------------------------------

class FlatFadingChannel(ChannelModel):
    """Flat-fading MIMO channel (Rayleigh or Rician).

    One channel matrix H is drawn per sample and applied uniformly to all
    T time steps.  This matches the block-fading assumption used in the
    DEFINED paper.

    Parameters
    ----------
    args : Namespace
        Must contain ``num_ant``, ``prompt_seq_length``, ``modulation``,
        ``SNR_dB_min``, ``SNR_dB_max``.
    fading_type : {'rayleigh', 'rician'}
    K_factor : float
        Rician K-factor (ignored for Rayleigh).
    """

    def __init__(self, args, fading_type: str = "rayleigh", K_factor: float = 1.0):
        fading_type = fading_type.lower()
        if fading_type not in {"rayleigh", "rician"}:
            raise ValueError("fading_type must be 'rayleigh' or 'rician'.")
        self.args = args
        self.fading_type = fading_type
        self.K_factor = K_factor

    # ------------------------------------------------------------------
    # ChannelModel interface
    # ------------------------------------------------------------------

    def generate(self, args, rng: np.random.Generator):
        """Generate one flat-fading sample.

        Returns
        -------
        x_seq  : (T, num_ant) complex — transmitted symbols
        y_seq  : (T, num_ant) complex — received signals
        snr_db : float
        """
        H = self._draw_H(args.num_ant, rng)

        snr_db = rng.uniform(args.SNR_dB_min, args.SNR_dB_max)
        noise_var = 10.0 ** (-snr_db / 10.0)

        x, _ = generate_modulated_signal(args, args.modulation, rng)  # (N, T)

        n = (
            rng.standard_normal((args.num_ant, args.prompt_seq_length))
            + 1j * rng.standard_normal((args.num_ant, args.prompt_seq_length))
        ) / np.sqrt(2) * np.sqrt(noise_var)

        y = H @ x + n  # (N, T)

        return x.T, y.T, snr_db  # (T, N), (T, N)

    @property
    def seq_length(self) -> int:
        return self.args.prompt_seq_length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_H(self, num_ant: int, rng: np.random.Generator) -> np.ndarray:
        if self.fading_type == "rayleigh":
            return _draw_rayleigh(num_ant, rng)
        return _draw_rician(num_ant, self.K_factor, rng)


# ---------------------------------------------------------------------------
# Batch helper (used by classical baselines only)
# ---------------------------------------------------------------------------

def generate_signals(
    batch_size: int,
    args,
    channel_type: str = "rayleigh",
    K_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Generate a batch of flat-fading signals.

    Used exclusively by the classical baselines in ``baseline.py``.  The
    neural training pipeline uses ``MIMOSequenceDataset`` instead.

    Returns
    -------
    X  : (batch_size, T, num_ant) complex — transmitted symbols
    Y  : (batch_size, T, num_ant) complex — received signals
    Hs : list of (num_ant, num_ant) complex — channel matrices
    """
    rng = np.random.default_rng()
    channel = FlatFadingChannel(args, fading_type=channel_type, K_factor=K_factor)

    X, Y, Hs = [], [], []
    for _ in range(batch_size):
        H = channel._draw_H(args.num_ant, rng)
        snr_db = rng.uniform(args.SNR_dB_min, args.SNR_dB_max)
        noise_var = 10.0 ** (-snr_db / 10.0)

        x, _ = generate_modulated_signal(args, args.modulation, rng)  # (N, T)
        n = (
            rng.standard_normal((args.num_ant, args.prompt_seq_length))
            + 1j * rng.standard_normal((args.num_ant, args.prompt_seq_length))
        ) / np.sqrt(2) * np.sqrt(noise_var)
        y = H @ x + n

        X.append(x.T)   # (T, N)
        Y.append(y.T)
        Hs.append(H)

    return np.stack(X), np.stack(Y), Hs
