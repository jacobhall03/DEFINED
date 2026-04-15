from typing import Tuple
import numpy as np

from channels.flat_fading import generate_signals
from data.modulation import generate_modulated_signal
from data.encoding import lmmse_channel_estimation, predict_symbol


def predict_lmmse_known_h(
    H: np.ndarray,
    y: np.ndarray,
    snr_db: float,
    constellation: np.ndarray,
):
    num_ant = H.shape[0]
    sigma2 = 10.0 ** (-snr_db / 10.0)

    # All joint combinations across antennas, used for nearest-neighbor decision
    all_combinations = np.array(
        np.meshgrid(*[constellation] * num_ant)
    ).T.reshape(-1, num_ant)  # (M_joint, num_ant)

    I = np.eye(num_ant, dtype=complex)
    H_H = np.conj(H.T)

    # LMMSE linear estimate x_hat_lin = (H^H H + 2*sigma^2 I)^{-1} H^H y
    inverse_term = np.linalg.inv(2.0 * sigma2 * I + H_H @ H)
    x_hat_lin = inverse_term @ (H_H @ y)

    # Nearest-neighbor in the joint constellation
    distances = np.sum(np.abs(x_hat_lin - all_combinations) ** 2, axis=1)
    best_idx = np.argmin(distances)
    return all_combinations[best_idx]


def DFE_MMSE_SER(
    args,
    num_samples: int,
    pilot_len: int,
    snr_db: float,
    channel_type: str = "rayleigh",
    K_factor: float = 1.0,
):
    T = args.prompt_seq_length
    task = f"DFE-MMSE-MIMO_{args.num_ant} {args.modulation} Pilot_{pilot_len} SNR_{snr_db} dB"
    print(f"*** Start {task}")

    # Fix SNR range for generation
    args.SNR_dB_min = snr_db
    args.SNR_dB_max = snr_db

    # Generate complex-domain data using the flat-fading batch generator
    X, Y, Hs = generate_signals(
        batch_size=num_samples,
        args=args,
        channel_type=channel_type,
        K_factor=K_factor,
    )
    # X, Y: (num_samples, T, num_ant), complex
    assert X.shape == (num_samples, T, args.num_ant)
    assert Y.shape == (num_samples, T, args.num_ant)

    # Single-antenna constellation (then used jointly across antennas)
    _, constellation = generate_modulated_signal(args, args.modulation)

    errors = np.zeros(T, dtype=np.float64)
    total_sequences = num_samples

    for i in range(num_samples):
        # Pilot region
        x_pilot = X[i, :pilot_len, :]   # (pilot_len, num_ant)
        y_pilot = Y[i, :pilot_len, :]   # (pilot_len, num_ant)

        for t in range(pilot_len, T):
            # LMMSE channel estimation from current (pilot + decisions)
            h_est = lmmse_channel_estimation(x_pilot, y_pilot, snr_db)  # (num_ant, num_ant)

            # Current observation and ground truth
            y_t = Y[i, t, :]     # (num_ant,)
            x_true = X[i, t, :]  # (num_ant,)

            # DFE detection at time t
            x_pred = predict_symbol(h_est, y_t, constellation)

            # Append predicted symbol to the "pilot" set (decision feedback)
            x_pilot = np.vstack((x_pilot, x_pred))
            y_pilot = np.vstack((y_pilot, y_t))

            # Check for symbol error
            if not np.allclose(x_pred, x_true, atol=1e-5):
                errors[t] += 1

    ser = errors / max(total_sequences, 1)
    print("DFE-MMSE SER per time step:", np.round(ser, 4).tolist())

    return ser


def calculate_ser(
    args,
    num_samples: int,
    pilot_len: int,
    snr_db: float,
    channel_type: str = "rayleigh",
    K_factor: float = 1.0,
):
    # Ensure the sequence is long enough
    if args.prompt_seq_length < pilot_len + 1:
        args.prompt_seq_length = pilot_len + 1

    T = args.prompt_seq_length

    # Fix SNR range
    args.SNR_dB_min = snr_db
    args.SNR_dB_max = snr_db

    # Generate signals
    X, Y, Hs = generate_signals(
        batch_size=num_samples,
        args=args,
        channel_type=channel_type,
        K_factor=K_factor,
    )

    # Single-antenna constellation
    _, constellation = generate_modulated_signal(args, args.modulation)

    errors = 0
    total_sequences = num_samples

    for i in range(num_samples):
        # Pilot region: first `pilot_len` symbols
        x_pilot = X[i, :pilot_len, :]   # (pilot_len, num_ant)
        y_pilot = Y[i, :pilot_len, :]   # (pilot_len, num_ant)

        h_est = lmmse_channel_estimation(x_pilot, y_pilot, snr_db)

        # Detect the (pilot_len)-th symbol
        y_t = Y[i, pilot_len, :]     # (num_ant,)
        x_true = X[i, pilot_len, :]  # (num_ant,)

        x_pred = predict_symbol(h_est, y_t, constellation)

        if not np.allclose(x_pred, x_true, atol=1e-5):
            errors += 1

    ser = errors / max(total_sequences, 1)
    print(
        f"MIMO_{args.num_ant} {args.modulation} Pilot_{pilot_len} "
        f"SNR_{snr_db} dB h_MMSE SER_{ser:.4f}"
    )
    return ser
