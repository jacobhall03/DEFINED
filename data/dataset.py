"""PyTorch Dataset for MIMO symbol detection.

``MIMOSequenceDataset`` is channel-agnostic: it delegates all signal
generation to a ``ChannelModel`` instance.  The encoding pipeline
(joint constellation lookup, one-hot encoding, real-feature conversion)
is the same for every channel type.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

from data.modulation import build_joint_constellation
from data.encoding import complex_to_vec, encode_joint_symbols, one_hot_from_indices
from channels.base import ChannelModel


def _ofdm_pilot_indices(args, seq_length: int) -> np.ndarray:
    """Return explicit OFDM pilot bin indices for this configuration."""
    explicit = getattr(args, "pilot_indices", None)
    if explicit is not None:
        pilots = np.asarray(explicit, dtype=np.int64)
    else:
        spacing = getattr(args, "pilot_spacing", None)
        if spacing is not None:
            pilots = np.arange(0, seq_length, int(spacing), dtype=np.int64)
        else:
            pilots = np.arange(int(getattr(args, "train_pilot_len", 0)), dtype=np.int64)

    pilots = np.unique(pilots)
    if pilots.size == 0:
        raise ValueError("OFDM pilot_indices must contain at least one pilot bin")
    if pilots.min() < 0 or pilots.max() >= seq_length:
        raise ValueError(f"OFDM pilot_indices must be in [0, {seq_length - 1}]")
    return pilots


class MIMOSequenceDataset(Dataset):
    """On-the-fly MIMO sequence dataset.

    Each call to ``__getitem__`` draws a fresh channel realisation, generates
    one sequence of T symbol/observation pairs, and returns them as tensors
    ready for the transformer.

    Parameters
    ----------
    args : Namespace
        Experiment configuration.  Must provide:
          num_ant, modulation, SNR_dB_min, SNR_dB_max, prompt_seq_length
          (and any channel-specific fields required by ``channel``).
    num_samples : int
        Dataset length — number of unique samples (``__len__``).
    channel : ChannelModel
        Instantiated channel object (FlatFadingChannel or OFDMChannel).
    joint_constellation : np.ndarray
        Precomputed joint constellation, shape (num_classes, num_ant).
        Build with ``data.modulation.build_joint_constellation``.
    seed : int or None
        Random seed for this dataset partition.

    Returned dict keys
    ------------------
    x               : (T, num_classes), float32  — one-hot transmit symbols
    y               : (T, 2*num_ant),   float32  — real/imag received features
    H_not_available : placeholder (zeros) — H is not exposed to the model
    snr_db          : scalar float32
    joint_constellation : (num_classes, num_ant), complex64
    """

    def __init__(
        self,
        args,
        num_samples: int,
        channel: ChannelModel,
        joint_constellation: np.ndarray,
        seed=None,
    ):
        super().__init__()
        self.args = args
        self.num_samples = num_samples
        self.channel = channel
        self.rng = np.random.default_rng(seed)

        self.joint_constellation_np = np.asarray(joint_constellation)
        self.num_classes = self.joint_constellation_np.shape[0]
        self.joint_constellation_tensor = torch.from_numpy(
            self.joint_constellation_np.astype(np.complex64)
        )

        print(
            f"*** MIMOSequenceDataset: channel={type(channel).__name__}, "
            f"T={channel.seq_length}, num_classes={self.num_classes} ***"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict:
        x_seq, y_seq, snr_db = self.channel.generate(self.args, self.rng)
        # x_seq : (T, num_ant) complex
        # y_seq : (T, num_ant) complex

        x_indices = encode_joint_symbols(x_seq, self.joint_constellation_np)  # (T,)
        x_onehot  = one_hot_from_indices(x_indices, self.num_classes)          # (T, C)
        y_feat    = complex_to_vec(y_seq)                                       # (T, 2N)

        seq_length = x_onehot.shape[0]
        subcarrier_indices = np.arange(seq_length, dtype=np.int64)
        pilot_mask = np.zeros(seq_length, dtype=bool)

        if getattr(self.args, "channel_type", None) == "ofdm":
            pilot_indices = _ofdm_pilot_indices(self.args, seq_length)
            pilot_set = set(pilot_indices.tolist())
            data_indices = np.asarray(
                [k for k in range(seq_length) if k not in pilot_set],
                dtype=np.int64,
            )
            order = np.concatenate([pilot_indices, data_indices])
            x_onehot = x_onehot[order]
            y_feat = y_feat[order]
            subcarrier_indices = order
            pilot_mask[:len(pilot_indices)] = True

        data_mask = ~pilot_mask

        return {
            "x":    torch.from_numpy(x_onehot.astype(np.float32)),              # (T, C)
            "y":    torch.from_numpy(y_feat.astype(np.float32)),                 # (T, 2N)
            "subcarrier_indices": torch.from_numpy(subcarrier_indices),
            "pilot_mask": torch.from_numpy(pilot_mask),
            "data_mask": torch.from_numpy(data_mask),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),                 # scalar
            "joint_constellation": self.joint_constellation_tensor,              # (C, N)
        }
