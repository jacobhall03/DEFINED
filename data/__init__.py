"""Data package.

Preferred import style (explicit sub-module):
    from data.modulation import build_joint_constellation
    from data.encoding  import complex_to_vec
    from data.dataset   import MIMOSequenceDataset

Short-form re-exports are provided below for the numpy-only utilities.
MIMOSequenceDataset is NOT re-exported here because it depends on torch;
import it directly from data.dataset to avoid loading torch at package
import time (which would break channel modules that only need numpy).
"""

from data.modulation import (
    generate_modulated_signal,
    build_joint_constellation,
    count_modulation_symbols,
)
from data.encoding import (
    complex_to_vec,
    encode_joint_symbols,
    one_hot_from_indices,
    lmmse_channel_estimation,
    predict_symbol,
)

__all__ = [
    "generate_modulated_signal",
    "build_joint_constellation",
    "count_modulation_symbols",
    "complex_to_vec",
    "encode_joint_symbols",
    "one_hot_from_indices",
    "lmmse_channel_estimation",
    "predict_symbol",
]
