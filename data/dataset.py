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

        return {
            "x":    torch.from_numpy(x_onehot.astype(np.float32)),              # (T, C)
            "y":    torch.from_numpy(y_feat.astype(np.float32)),                 # (T, 2N)
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),                 # scalar
            "joint_constellation": self.joint_constellation_tensor,              # (C, N)
        }
