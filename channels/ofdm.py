"""OFDM frequency-selective MIMO channel model.

An OFDM system converts a frequency-selective multipath channel into K
parallel flat-fading sub-channels via the FFT.  Each sub-carrier k has its
own channel matrix H_k, derived from the DFT of the L-tap channel impulse
response (CIR).

Sequence dimension
------------------
Unlike the flat-fading model where T = ``prompt_seq_length`` (time steps),
here T = ``num_subcarriers`` (frequency sub-carriers within one OFDM symbol).
Each position in the transformer sequence corresponds to one sub-carrier.

Channel model
-------------
The CIR has L taps with exponentially decaying average power:
    p_l = exp(-l / delay_spread),  l = 0, …, L-1
normalised so that sum_l p_l = 1.  Each tap is an independent (N_r × N_t)
complex Gaussian matrix.

The per-sub-carrier channel is:
    H_k = sum_{l=0}^{L-1} h_l * exp(-j 2π k l / K)
which is computed efficiently as an K-point DFT of the tap sequence.
"""

import numpy as np

from .base import ChannelModel
from data.modulation import generate_modulated_signal


class OFDMChannel(ChannelModel):
    """Frequency-selective MIMO OFDM channel.

    Parameters
    ----------
    args : Namespace
        Must contain:
          ``num_ant``         — number of TX/RX antennas (square MIMO)
          ``num_subcarriers`` — number of OFDM sub-carriers (K)
          ``num_taps``        — number of multipath CIR taps (L)
          ``delay_spread``    — exponential decay constant for tap powers
          ``modulation``      — modulation scheme string
          ``SNR_dB_min/max``  — SNR range in dB
    """

    def __init__(self, args):
        self.args = args

    # ------------------------------------------------------------------
    # ChannelModel interface
    # ------------------------------------------------------------------

    def generate(self, args, rng: np.random.Generator):
        """Generate one OFDM sample.

        Returns
        -------
        x_seq  : (K, num_ant) complex — transmitted symbols per sub-carrier
        y_seq  : (K, num_ant) complex — received signals per sub-carrier
        snr_db : float
        """
        K = args.num_subcarriers
        N = args.num_ant

        H_freq = self._generate_multipath(args, rng)  # (K, N, N)

        snr_db = rng.uniform(args.SNR_dB_min, args.SNR_dB_max)
        noise_var = 10.0 ** (-snr_db / 10.0)

        # Generate K symbols per antenna (one per sub-carrier).
        # We temporarily set prompt_seq_length = K so generate_modulated_signal
        # produces the right shape.
        x_mat, _ = generate_modulated_signal(
            args, args.modulation, rng, seq_length=K
        )  # (N, K) complex

        x_seq = x_mat.T  # (K, N)

        # Complex AWGN: (K, N)
        n = (
            rng.standard_normal((K, N)) + 1j * rng.standard_normal((K, N))
        ) / np.sqrt(2) * np.sqrt(noise_var)

        # Apply per-sub-carrier channel: y_k = H_k x_k + n_k
        y_seq = np.einsum("kij,kj->ki", H_freq, x_seq) + n  # (K, N)

        return x_seq, y_seq, snr_db

    @property
    def seq_length(self) -> int:
        return self.args.num_subcarriers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_multipath(self, args, rng: np.random.Generator) -> np.ndarray:
        """Return H_freq of shape (K, N, N) via DFT of the CIR taps.

        Tap powers decay exponentially with decay constant ``delay_spread``.
        The total power is normalised to 1 across all taps.
        """
        K = args.num_subcarriers
        N = args.num_ant
        L = args.num_taps

        # Exponentially decaying tap power profile
        tap_indices = np.arange(L, dtype=float)
        tap_powers = np.exp(-tap_indices / max(args.delay_spread, 1e-6))
        tap_powers /= tap_powers.sum()  # unit total power

        # i.i.d. complex Gaussian taps: (L, N, N)
        taps = np.zeros((L, N, N), dtype=complex)
        for l in range(L):
            taps[l] = (
                rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
            ) / np.sqrt(2) * np.sqrt(tap_powers[l])

        # K-point DFT along the tap axis → (K, N, N)
        H_freq = np.fft.fft(taps, n=K, axis=0)
        return H_freq
