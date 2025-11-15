import numpy as np

def estimate_noise_covariance(
    received_signals_antennas, 
    known_training_sequence,
    channel_estimates_antennas
):
    """
    Estimation of noise+interference covariance matrix Q

    Q = E[e(t) * e(t)^H]
    where e(t) = r(t) - Σ b_m · h(t-mT)
    """
    N_R = len(received_signals_antennas)
    L = len(channel_estimates_antennas[0])
    len_ts = len(known_training_sequence)

    noise_interference_estimates = []

    for ant_idx in range(N_R):
        h_est = channel_estimates_antennas[ant_idx]
        r_ts = received_signals_antennas[ant_idx]

        # Restore expected signal
        expected_signal = np.convolve(h_est, known_training_sequence, mode='same')
        expected_signal = expected_signal[:len_ts]
        r_ts_crop = r_ts[:len_ts]

        # e(t) = r(t) - h*b
        e_t = r_ts_crop - expected_signal
        noise_interference_estimates.append(e_t)

    # Q = E[e * e^H]
    noise_matrix = np.array(noise_interference_estimates)  # (N_R x len_ts)
    Q = (noise_matrix @ noise_matrix.conj().T) / len_ts

    return Q


def compute_irc_filters(channel_estimates_antennas, Q_inv):
    """
    Calculation of IRC filters (equation 23)

    g_i(t) = Σ_j v_ij · h_j^*(-t)

    In discrete: g_i[n] = Σ_j v_ij · h_j[::-1].conj()
    Args:
        channel_estimates_antennas: list of arrays [h1, h2, ...]
        Q_inv: (N_R x N_R) inverse noise covariance

    Returns:
        irc_filters: list of arrays [g1, g2, ...]
    """
    N_R = len(channel_estimates_antennas)
    L = len(channel_estimates_antennas[0])

    irc_filters = []

    for i in range(N_R):
        g_i = np.zeros(L, dtype=complex)

        for j in range(N_R):
            v_ij = Q_inv[i, j]
            # h_j^*(-t) = reverse(conj(h_j))
            h_j_matched = channel_estimates_antennas[j][::-1].conj()
            g_i += v_ij * h_j_matched

        irc_filters.append(g_i)

    return irc_filters


def apply_irc_combining(received_signals_antennas, irc_filters):
    """
    Application of IRC filters (equation 24)

    ŷ_n = Σ_i ∫ r_i(t) · g_i(t-nT) dt

    In discrete: ŷ[n] = Σ_i [r_i * g_i][n]
    Args:
        received_signals_antennas: list of arrays
        irc_filters: list of arrays

    Returns:
        combined_signal: array
    """
    N_R = len(received_signals_antennas)
    signal_length = len(received_signals_antennas[0])

    combined_signal = np.zeros(signal_length, dtype=complex)

    for i in range(N_R):
        # Convolution r_i * g_i
        filtered = np.convolve(received_signals_antennas[i], irc_filters[i], mode='same')
        combined_signal += filtered

    return combined_signal


def compute_effective_channel(channel_estimates_antennas, irc_filters):
    """
    Calculation of effective channel (equation 25)

    S_n = Σ_i ∫ h_i(t) · g_i(t-nT) dt

    In discrete: S[n] = Σ_i [h_i * g_i][n]
    Args:
        channel_estimates_antennas: list of arrays
        irc_filters: list of arrays

    Returns:
        S: array, effective channel for MLSE
    """
    N_R = len(channel_estimates_antennas)
    L = len(channel_estimates_antennas[0])

    # Convolution h * g gives length (L + L - 1)
    conv_len = 2 * L - 1
    S_full = np.zeros(conv_len, dtype=complex)

    for i in range(N_R):
        # Convolution h_i * g_i
        conv_result = np.convolve(channel_estimates_antennas[i], irc_filters[i], mode='full')
        S_full += conv_result

    # Take the central part of length L
    # S_full has length 2*L-1, center at index L-1
    start_idx = L // 2
    S = S_full[start_idx:start_idx + L]

    return S


def irc_mlse_preprocess(
    received_signals_antennas,
    channel_estimates_antennas,
    known_training_sequence,
    regularization=0
):
    """
    Args:
        received_signals_antennas: list of arrays
        channel_estimates_antennas: list of arrays
        known_training_sequence: array
        regularization: float

    Returns:
        combined_signal: array
        effective_channel: array
    """
    N_R = len(received_signals_antennas)

    # 1. estimate Q
    Q = estimate_noise_covariance(
        received_signals_antennas,
        known_training_sequence,
        channel_estimates_antennas
    )

    # 2. Inversion of Q with regularization
    Q_reg = Q + regularization * np.eye(N_R)

    try:
        Q_inv = np.linalg.inv(Q_reg)
    except np.linalg.LinAlgError:
        Q_inv = np.linalg.pinv(Q_reg)

    # 3. Calculation of IRC filters
    irc_filters = compute_irc_filters(channel_estimates_antennas, Q_inv)

    # 4. Application of IRC combining
    combined_signal = apply_irc_combining(received_signals_antennas, irc_filters)

    # 5. Calculation of effective channel
    effective_channel = compute_effective_channel(channel_estimates_antennas, irc_filters)

    return combined_signal, effective_channel


def irc_diversity_combining(
    rx_ant1, 
    rx_ant2,
    h_est_ant1,
    h_est_ant2,
    training_sequence,
    enable_irc=True
):
    """
    Args:
        rx_ant1, rx_ant2: arrays
        h_est_ant1, h_est_ant2: arrays
        training_sequence: array
        enable_irc: bool

    Returns:
        combined_signal: array
        effective_channel: array
    """
    training_sequence = np.array(training_sequence, dtype=complex)

    if not enable_irc:
        # MRC fallback
        # Matched filters
        g1 = h_est_ant1[::-1].conj()
        g2 = h_est_ant2[::-1].conj()

        mf1 = np.convolve(rx_ant1, g1, mode='same')
        mf2 = np.convolve(rx_ant2, g2, mode='same')

        combined_signal = mf1 + mf2
    
        # Effective channel
        effective_channel = h_est_ant1 + h_est_ant2  # Simplification for MRC

        return combined_signal, effective_channel

    # IRC mode
    received_signals = [rx_ant1, rx_ant2]
    channel_estimates = [h_est_ant1, h_est_ant2]

    combined_signal, effective_channel = irc_mlse_preprocess(
        received_signals,
        channel_estimates,
        training_sequence
    )

    return combined_signal, effective_channel