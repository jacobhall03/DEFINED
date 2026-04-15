"""Signal encoding utilities and classical detection helpers.

This module contains:
  - Representation conversion (complex → real vectors, indices → one-hot)
  - Joint constellation encoding
  - Classical LMMSE channel estimation and ML symbol detection
    (used by the baseline evaluator)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Representation utilities
# ---------------------------------------------------------------------------

def complex_to_vec(X: np.ndarray) -> np.ndarray:
    """Stack real and imaginary parts along the last axis.

    X : (..., N) complex  →  (..., 2N) float
    """
    return np.concatenate([np.real(X), np.imag(X)], axis=-1)


def encode_joint_symbols(
    x_complex: np.ndarray,
    joint_constellation: np.ndarray,
) -> np.ndarray:
    """Map complex MIMO symbol vectors to joint constellation indices.

    Parameters
    ----------
    x_complex           : (B, T, num_ant) or (T, num_ant), complex
    joint_constellation : (M_joint, num_ant), complex

    Returns
    -------
    indices : (B, T) or (T,) integer ndarray, values in [0, M_joint)
    """
    squeezed = x_complex.ndim == 2
    if squeezed:
        x_complex = x_complex[None, :, :]  # (1, T, N)

    B, T, _ = x_complex.shape

    flat_x = x_complex.reshape(-1, x_complex.shape[-1])  # (B*T, N)
    diff = flat_x[:, None, :] - joint_constellation[None, :, :]
    dist2 = np.abs(diff) ** 2
    indices = np.argmin(dist2.sum(axis=-1), axis=1).reshape(B, T)

    return indices[0] if squeezed else indices


def one_hot_from_indices(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer indices to one-hot vectors.

    indices : (B, T) or (T,)  →  (..., num_classes) float32
    """
    return np.eye(num_classes, dtype=np.float32)[indices]


# ---------------------------------------------------------------------------
# Classical detection (used by baseline.py)
# ---------------------------------------------------------------------------

def lmmse_channel_estimation(
    x: np.ndarray,
    y: np.ndarray,
    snr: float,
) -> np.ndarray:
    """LMMSE-style channel estimation.

    Matches the original implementation:
        cov = X^H X + sigma2   (scalar offset, not sigma2 * I)

    Parameters
    ----------
    x   : (T, num_ant) complex — pilot symbols
    y   : (T, num_ant) complex — received pilots
    snr : float — SNR in dB

    Returns
    -------
    h_est : (num_ant, num_ant) complex
    """
    sigma2 = 10.0 ** (-snr / 10.0)
    cov = np.conj(x).T @ x + sigma2
    h_est = np.linalg.pinv(cov) @ np.conj(x).T @ y
    return h_est.T


def predict_symbol(
    h_est: np.ndarray,
    y: np.ndarray,
    constellation: np.ndarray,
) -> np.ndarray:
    """ML symbol detection via exhaustive nearest-neighbour search.

    Parameters
    ----------
    h_est        : (num_ant, num_ant) complex
    y            : (num_ant,) complex
    constellation: (M,) complex — single-antenna constellation

    Returns
    -------
    x_pred : (num_ant,) complex — the best joint symbol vector
    """
    num_ant = h_est.shape[0]
    all_combos = np.array(
        np.meshgrid(*[constellation] * num_ant)
    ).T.reshape(-1, num_ant)

    y_col = y.reshape(-1, 1)
    distances = np.sum(np.abs(y_col - h_est @ all_combos.T) ** 2, axis=0)
    return all_combos[np.argmin(distances)]
