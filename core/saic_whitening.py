import numpy as np


def estimate_interference_power(
    received_signal,
    channel_estimate,
    known_training_sequence
):
    """
    Estimate interference+noise power using the training sequence.

    Args:
        received_signal: received signal on the antenna
        channel_estimate: channel estimate h
        known_training_sequence: known training sequence (TS)

    Returns:
        Q_single: scalar estimate of interference+noise power

    Algorithm (ETSI TS 145.008, Annex C):
    1. Reconstruct expected signal: expected = h * TS
    2. Residual: e(t) = r(t) - expected
    3. Q_single = E[|e(t)|^2] = (1/M) * Σ|e(t)|^2
    """
    len_ts = len(known_training_sequence)

    # Expected useful signal on the training sequence
    expected_signal = np.convolve(
        channel_estimate, 
        known_training_sequence, 
        mode='same'
    )
    expected_signal = expected_signal[:len_ts]

    # Received signal on the training sequence
    r_ts = received_signal[:len_ts]

    # Residual = interference + noise
    residual = r_ts - expected_signal

    # Interference+noise power (scalar)
    Q_single = np.mean(np.abs(residual)**2)

    return Q_single


def compute_whitening_filter(
    channel_estimate,
    Q_single,
    regularization=1e-6
):
    """
    Compute whitening filter for a single antenna.

    Args:
        channel_estimate: h - channel estimate
        Q_single: scalar - interference+noise power
        regularization: regularization value for stability

    Returns:
        whitening_filter: g = h^*(-t) / sqrt(Q_single + lambda)

    Physical interpretation:
    - Matched filter: h^*(-t)
    - Normalization by sqrt(Q) normalizes the power
    - If Q_single ≈ σ^2_noise (pure AWGN): g ≈ matched filter
    - If Q_single >> σ^2_noise (strong interference): g suppresses interference
    """
    L = len(channel_estimate)

    # Matched filter: h^*(-t)
    matched_filter = channel_estimate[::-1].conj()

    # Normalization by interference power
    # Regularization for stability (similar to IRC)
    Q_regularized = Q_single + regularization
    normalization = np.sqrt(Q_regularized)

    # Whitening filter
    whitening_filter = matched_filter / normalization

    return whitening_filter


def apply_whitening_filter(
    received_signal,
    whitening_filter
):
    """
    Apply the whitening filter to the signal.

    Args:
        received_signal: received signal
        whitening_filter: g - whitening filter

    Returns:
        whitened_signal: y = r * g
    """
    whitened_signal = np.convolve(
        received_signal,
        whitening_filter,
        mode='same'
    )

    return whitened_signal


def compute_effective_channel_saic(
    channel_estimate,
    whitening_filter
):
    """
    Compute the effective channel for MLSE.

    Args:
        channel_estimate: h
        whitening_filter: g

    Returns:
        S: effective channel S = h * g
    """
    L = len(channel_estimate)

    # Convolution h * g
    S_full = np.convolve(
        channel_estimate,
        whitening_filter,
        mode='full'
    )

    # Take the central part of length L
    start_idx = L // 2
    S = S_full[start_idx : start_idx + L]
    
    return S


def saic_mlse_preprocess(
    received_signal,
    channel_estimate,
    known_training_sequence,
    regularization=1e-6
):
    """
    Full SAIC preprocessing pipeline for a single antenna.

    Args:
        received_signal: received signal at the antenna
        channel_estimate: channel estimate (LS)
        known_training_sequence: TS (26 symbols in GSM)
        regularization: regularization (default: 1e-6)

    Returns:
        whitened_signal: for MLSE detector
        effective_channel: for Viterbi metric

    Pipeline:
    1. Estimate interference power Q_single
    2. Compute whitening filter g = h^*(-t) / sqrt(Q)
    3. Apply whitening: y = r * g
    4. Compute effective channel: S = h * g
    """

    # 1. Estimate interference+noise power
    Q_single = estimate_interference_power(
        received_signal,
        channel_estimate,
        known_training_sequence
    )

    # 2. Whitening filter
    whitening_filter = compute_whitening_filter(
        channel_estimate,
        Q_single,
        regularization
    )

    # 3. Apply whitening
    whitened_signal = apply_whitening_filter(
        received_signal,
        whitening_filter
    )

    # 4. Effective channel
    effective_channel = compute_effective_channel_saic(
        channel_estimate,
        whitening_filter
    )

    return whitened_signal, effective_channel


# ====================================================================
# Advanced SAIC: Bias Removal Method (Cichocki et al., 2004)
# ====================================================================

def saic_bias_removal(
    received_signal,
    channel_estimate,
    known_training_sequence,
    thermal_noise_variance=1e-6
):
    """
    SAIC with bias removal (removes thermal noise bias from Q estimate).

    Algorithm (Cichocki et al., 2004):
    1. Estimate Q_total = interference + thermal_noise
    2. Remove thermal noise bias: Q_interference = Q_total - σ^2_thermal
    3. Use Q_interference for whitening

    Args:
        thermal_noise_variance: estimate of σ^2_thermal (from BS NF)

    Returns:
        whitened_signal, effective_channel
    """

    # 1. Estimate total interference+noise power
    Q_total = estimate_interference_power(
        received_signal,
        channel_estimate,
        known_training_sequence
    )

    # 2. Bias removal
    Q_interference = Q_total - thermal_noise_variance

    # Protect against negative values
    Q_interference = max(Q_interference, thermal_noise_variance / 10)

    # 3. Whitening with corrected Q
    whitening_filter = compute_whitening_filter(
        channel_estimate,
        Q_interference,
        regularization=1e-10  # Меньшая регуляризация после bias removal
    )

    whitened_signal = apply_whitening_filter(
        received_signal,
        whitening_filter
    )

    effective_channel = compute_effective_channel_saic(
        channel_estimate,
        whitening_filter
    )

    return whitened_signal, effective_channel


# ====================================================================
# Wrapper for integration into the simulator
# ====================================================================

def single_antenna_processing(
    received_signal,
    channel_estimate,
    training_sequence,
    enable_saic=False,
    method='basic',
    regularization=1e-6,
    thermal_noise_variance=1e-6
):
    """
    Wrapper for single-antenna processing with optional SAIC.

    Args:
        received_signal: received signal
        channel_estimate: channel estimate (LS)
        training_sequence: TS
        enable_saic: enable SAIC whitening?
        method: 'basic' or 'bias_removal'
        regularization: lambda for regularization
        thermal_noise_variance: σ^2_thermal for bias removal

    Returns:
        processed_signal: for MLSE
        effective_channel: for Viterbi
    """

    training_sequence = np.array(training_sequence, dtype=complex)

    if not enable_saic:
        # Simple matched filter (as before)
        matched_filter = channel_estimate[::-1].conj()
        processed_signal = np.convolve(
            received_signal,
            matched_filter,
            mode='same'
        )
        effective_channel = channel_estimate

        return processed_signal, effective_channel

    # SAIC mode
    if method == 'bias_removal':
        processed_signal, effective_channel = saic_bias_removal(
            received_signal,
            channel_estimate,
            training_sequence,
            thermal_noise_variance
        )
    else:  # method == 'basic'
        processed_signal, effective_channel = saic_mlse_preprocess(
            received_signal,
            channel_estimate,
            training_sequence,
            regularization
        )

    return processed_signal, effective_channel

def compute_saic_gain(
    Q_without_saic,
    Q_with_saic
):
    """
    Compute SNR gain from SAIC whitening.

    Args:
        Q_without_saic: interference power without SAIC
        Q_with_saic: interference power with SAIC

    Returns:
        gain_db: gain in dB
    """
    if Q_with_saic > 0:
        gain_db = 10 * np.log10(Q_without_saic / Q_with_saic)
    else:
        gain_db = 0.0

    return gain_db
