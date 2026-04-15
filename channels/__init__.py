"""Channel model package.

Provides the ChannelModel abstraction and a factory function
``build_channel(args)`` that returns the appropriate channel object
based on ``args.channel_type``.
"""

from .base import ChannelModel
from .flat_fading import FlatFadingChannel
from .ofdm import OFDMChannel


def build_channel(args) -> ChannelModel:
    """Instantiate and return the channel model specified by ``args.channel_type``.

    Parameters
    ----------
    args : argparse.Namespace or SimpleNamespace
        Must contain ``channel_type`` ('flat_fading' or 'ofdm').
        Flat-fading also reads ``fading_type`` ('rayleigh'/'rician') and
        ``K_factor`` (Rician K-factor, default 1.0).
        OFDM reads ``num_subcarriers``, ``num_taps``, ``delay_spread``.
    """
    channel_type = getattr(args, "channel_type", "flat_fading")

    if channel_type == "flat_fading":
        return FlatFadingChannel(
            args,
            fading_type=getattr(args, "fading_type", "rayleigh"),
            K_factor=getattr(args, "K_factor", 1.0),
        )
    elif channel_type == "ofdm":
        return OFDMChannel(args)
    else:
        raise ValueError(
            f"Unknown channel_type: '{channel_type}'. "
            "Choose 'flat_fading' or 'ofdm'."
        )


__all__ = [
    "ChannelModel",
    "FlatFadingChannel",
    "OFDMChannel",
    "build_channel",
]
