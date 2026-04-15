"""Abstract base class for all channel models."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class ChannelModel(ABC):
    """Contract that every channel model must satisfy.

    Each concrete subclass encapsulates one channel type (e.g. flat Rayleigh,
    Rician, OFDM multipath).  The rest of the codebase — the dataset, training
    loop, and model — only depend on this interface and are therefore fully
    channel-agnostic.

    Calling convention
    ------------------
    x_seq, y_seq, snr_db = channel.generate(args, rng)

    Returns
    -------
    x_seq  : np.ndarray, shape (T, num_ant), complex
        Transmitted symbol sequence.
    y_seq  : np.ndarray, shape (T, num_ant), complex
        Received signal sequence after channel and noise.
    snr_db : float
        SNR (dB) used for this sample.
    """

    @abstractmethod
    def generate(
        self,
        args,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Generate one (x_seq, y_seq, snr_db) sample.

        Parameters
        ----------
        args : argparse.Namespace or SimpleNamespace
            Experiment configuration (num_ant, modulation, SNR range, …).
        rng : numpy.random.Generator
            Random number generator to use — passed in so the caller controls
            reproducibility via dataset-level seeds.
        """
        ...

    @property
    @abstractmethod
    def seq_length(self) -> int:
        """The sequence length T produced by this channel (read from args)."""
        ...
