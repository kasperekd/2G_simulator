import numpy as np

def estimate_noise_covariance_improved(
    received_signals_antennas,
    known_training_sequence,
    channel_estimates_antennas,
    shrinkage_factor=0.0
):
    N_R = len(received_signals_antennas)
    len_ts = len(known_training_sequence)
    
    noise_interference_estimates = []
    
    for ant_idx in range(N_R):
        h_est = channel_estimates_antennas[ant_idx]
        r_ts = received_signals_antennas[ant_idx][:len_ts]
        
        expected_signal = np.convolve(h_est, known_training_sequence, mode='same')
        
        ts_len_process = min(len(r_ts), len_ts)
        r_ts_crop = r_ts[:ts_len_process]
        expected_signal = expected_signal[:ts_len_process]
        
        e_t = r_ts_crop - expected_signal
        noise_interference_estimates.append(e_t)
        
    noise_matrix = np.array(noise_interference_estimates) 
    
    Q_scm = (noise_matrix @ noise_matrix.conj().T) / ts_len_process
    
    if shrinkage_factor > 0:
        avg_power = np.trace(Q_scm).real / N_R
        Target = avg_power * np.eye(N_R)
        Q = (1 - shrinkage_factor) * Q_scm + shrinkage_factor * Target
    else:
        Q = Q_scm
        
    return Q

def irc_corrected_process(
    received_signals_antennas,
    channel_estimates_antennas,
    known_training_sequence,
    shrinkage=0.1,
    loading_factor=0.2
):
    N_R = len(received_signals_antennas)
    L = len(channel_estimates_antennas[0])
    
    Q = estimate_noise_covariance_improved(
        received_signals_antennas,
        known_training_sequence,
        channel_estimates_antennas,
        shrinkage_factor=shrinkage
    )
    
    # 2. Adaptive Diagonal Loading
    avg_noise_power = np.trace(Q).real / N_R
    # Q_reg = (1-LF)*Q + LF*sigma^2*I
    Q_reg = (1 - loading_factor) * Q + loading_factor * avg_noise_power * np.eye(N_R)
    
    try:
        Q_inv = np.linalg.inv(Q_reg)
    except np.linalg.LinAlgError:
        Q_inv = np.linalg.pinv(Q_reg)
        
    # 3.
    irc_filters = []
    for i in range(N_R):
        g_i = np.zeros(L, dtype=complex)
        for j in range(N_R):
            v_ij = Q_inv[i, j]
            h_j_matched = channel_estimates_antennas[j][::-1].conj()
            g_i += v_ij * h_j_matched
        irc_filters.append(g_i)
        
    # 4.
    signal_length = len(received_signals_antennas[0])
    combined_signal = np.zeros(signal_length, dtype=complex)
    for i in range(N_R):
        filtered = np.convolve(received_signals_antennas[i], irc_filters[i], mode='same')
        combined_signal += filtered
        
    conv_len = 2 * L - 1
    effective_channel = np.zeros(conv_len, dtype=complex)
    for i in range(N_R):
        conv_result = np.convolve(channel_estimates_antennas[i], irc_filters[i], mode='full')
        effective_channel += conv_result
        
    start_idx = L // 2
    effective_channel_final = effective_channel[start_idx:start_idx + L]
    
    return combined_signal, effective_channel_final