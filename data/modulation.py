"""Modulation helpers: symbol generation and constellation construction."""

from typing import Optional, Tuple

import numpy as np


def generate_modulated_signal(
    args,
    modulation: str,
    rng: Optional[np.random.Generator] = None,
    seq_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a random baseband modulated symbol matrix.

    Parameters
    ----------
    args : Namespace
        Must contain ``num_ant`` and ``prompt_seq_length``.
    modulation : str
        One of 'BPSK', 'QPSK', '16QAM', '64QAM'.
    rng : numpy.random.Generator, optional
        RNG instance for reproducibility.  A fresh one is created if None.
    seq_length : int, optional
        Override the sequence length.  Defaults to ``args.prompt_seq_length``.
        Pass ``args.num_subcarriers`` when generating OFDM symbols.

    Returns
    -------
    symbols      : (num_ant, T) complex ndarray
    constellation: (M,) complex ndarray — the single-antenna constellation
    """
    if rng is None:
        rng = np.random.default_rng()

    T = seq_length if seq_length is not None else args.prompt_seq_length

    if modulation in {"16QAM", "64QAM"}:
        M = int(modulation[:-3])
        side = int(np.sqrt(M))
        const_1d = 2 * np.arange(side) - (side - 1)
        constellation = (const_1d + 1j * const_1d[:, np.newaxis]).flatten()
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
        symbols = rng.choice(constellation, size=(args.num_ant, T))

    elif modulation == "BPSK":
        constellation = np.array([1.0, -1.0])
        symbols = rng.choice(constellation, size=(args.num_ant, T)).astype(complex)

    elif modulation == "QPSK":
        constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        symbols = rng.choice(constellation, size=(args.num_ant, T))

    else:
        raise ValueError(f"Unsupported modulation: '{modulation}'")

    return symbols, constellation


def build_joint_constellation(modulation: str, num_ant: int) -> np.ndarray:
    """Build the joint constellation over all antennas (Cartesian product).

    Parameters
    ----------
    modulation : str
    num_ant    : int

    Returns
    -------
    joint : (M**num_ant, num_ant) complex ndarray
        Each row is one valid joint symbol vector.
    """
    # Single-antenna constellation
    if modulation in {"16QAM", "64QAM"}:
        M = int(modulation[:-3])
        side = int(np.sqrt(M))
        const_1d = 2 * np.arange(side) - (side - 1)
        constellation = (const_1d + 1j * const_1d[:, np.newaxis]).flatten()
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
    elif modulation == "BPSK":
        constellation = np.array([1.0, -1.0])
    elif modulation == "QPSK":
        constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    else:
        raise ValueError(f"Unsupported modulation: '{modulation}'")

    joint = np.array(
        np.meshgrid(*[constellation] * num_ant)
    ).T.reshape(-1, num_ant)

    return joint  # (M^num_ant, num_ant)


def count_modulation_symbols(args) -> int:
    """Return the joint constellation size for the given args.

    For MIMO == 2, returns M**2 (joint size) rather than M.
    """
    modulation = args.modulation
    MIMO = args.num_ant

    if MIMO == 2:
        if modulation == "BPSK":
            return 4
        elif modulation == "QPSK":
            return 16
        elif modulation == "16QAM":
            return 256
        else:
            raise ValueError(f"Unsupported modulation for MIMO=2: '{modulation}'")

    if modulation in {"4QAM", "16QAM", "64QAM"}:
        return int(modulation[:-3])
    elif modulation == "BPSK":
        return 2
    elif modulation == "QPSK":
        return 4
    else:
        raise ValueError(f"Unsupported modulation: '{modulation}'")
