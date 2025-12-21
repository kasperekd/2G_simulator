import numpy as np

import numpy as np

def oas_shrinkage_covariance(Q_sample, n_samples):
    """
    Oracle Approximating Shrinkage (OAS) for covariance matrix estimation.
    
    Based on: Chen et al., "Shrinkage Algorithms for MMSE Covariance Estimation",
    IEEE Trans. Signal Processing, Vol. 58, No. 10, October 2010.
    
    Args:
        Q_sample: Sample covariance matrix (NR x NR), complex Hermitian
        n_samples: Number of samples used to compute Q_sample (e.g., 26 for GSM TS)
    
    Returns:
        Q_oas: Shrunk covariance estimate (NR x NR)
        rho_oas: Shrinkage coefficient used (for debugging/logging)
    """
    NR = Q_sample.shape[0]
    
    # Compute traces
    tr_Q = np.trace(Q_sample).real
    tr_Q2 = np.trace(Q_sample @ Q_sample).real
    
    # Sphericity statistic U_hat
    # U measures how far Q is from a scaled identity matrix
    # U ≈ 0: Q ≈ σ²I (uncorrelated) → large shrinkage
    # U large: Q has structure (correlated) → small shrinkage
    U_hat = (NR / (NR - 1)) * ((NR * tr_Q2) / (tr_Q ** 2) - 1)
    
    # OAS shrinkage coefficient
    # Two components:
    # rho_1: depends on sample size n
    # rho_2: depends on sphericity U_hat
    rho_1 = 1.0 / (n_samples + 1 - 2.0 / NR)
    rho_2 = 1.0 / U_hat if U_hat > 1e-10 else 1.0  # Avoid division by zero
    
    rho_oas = min(rho_1, rho_2)
    rho_oas = np.clip(rho_oas, 0.0, 1.0)  # Safety: ensure [0, 1]
    
    # Shrinkage target: scaled identity
    # F = (Tr(Q) / NR) * I
    F = (tr_Q / NR) * np.eye(NR, dtype=complex)
    
    # Shrunk covariance estimate
    # Q_oas = (1 - rho) * Q_sample + rho * F
    Q_oas = (1 - rho_oas) * Q_sample + rho_oas * F
    
    return Q_oas, rho_oas


def rblw_shrinkage_covariance(Q_sample, n_samples):
    """
    Rao-Blackwell Ledoit-Wolf (RBLW) shrinkage for covariance estimation.
    
    Alternative to OAS, provably dominates standard Ledoit-Wolf for Gaussian samples.
    Based on same paper as OAS.
    
    Args:
        Q_sample: Sample covariance matrix (NR x NR)
        n_samples: Number of samples
    
    Returns:
        Q_rblw: Shrunk covariance estimate
        rho_rblw: Shrinkage coefficient
    """
    NR = Q_sample.shape[0]
    n = n_samples
    
    # Compute traces
    tr_Q = np.trace(Q_sample).real
    tr_Q2 = np.trace(Q_sample @ Q_sample).real
    
    # RBLW shrinkage coefficient (equation 17 from paper)
    numerator = (n - 2) * tr_Q2 + tr_Q ** 2
    denominator = (n + 2) * tr_Q2 - tr_Q ** 2
    
    if denominator > 1e-10:
        rho_rblw = numerator / denominator
    else:
        rho_rblw = 1.0
    
    rho_rblw = np.clip(rho_rblw, 0.0, 1.0)
    
    # Shrinkage target
    F = (tr_Q / NR) * np.eye(NR, dtype=complex)
    
    # Shrunk covariance
    Q_rblw = (1 - rho_rblw) * Q_sample + rho_rblw * F
    
    return Q_rblw, rho_rblw


def estimate_noise_covariance_improved(
    receiveSignalsAntennas,
    knownTrainingSequence,
    channelEstimatesAntennas,
    shrinkageMethod='oas',   # 'oas', 'rblw', 'diagonal_loading', or 'none'
    shrinkageFactor=0.0,    # Legacy: only used if shrinkageMethod='diagonal_loading'
):
    """
    Estimate noise/interference covariance matrix Q with advanced shrinkage methods.
    
    Args:
        receiveSignalsAntennas: List of received signals per antenna
        knownTrainingSequence: Known training symbols
        channelEstimatesAntennas: List of channel estimates per antenna
        shrinkageMethod: Method for covariance regularization
            - 'oas': Oracle Approximating Shrinkage
            - 'rblw': Rao-Blackwell Ledoit-Wolf
            - 'diagonal_loading': Legacy fixed shrinkage
            - 'none': No shrinkage (sample covariance only)
        shrinkageFactor: Only used for 'diagonal_loading' method
    
    Returns:
        Q: Regularized covariance matrix (NR x NR)
    """
    NR = len(receiveSignalsAntennas)
    len_ts = len(knownTrainingSequence)
    
    # Step 1: Compute residuals e(t) = r(t) - sum_m h_m * s(t-m)
    noiseInterferenceEstimates = []
    
    for ant_idx in range(NR):
        h_est = channelEstimatesAntennas[ant_idx]
        r_ts = receiveSignalsAntennas[ant_idx][:len_ts]
        
        # Expected signal: h * training_sequence
        expected_signal = np.convolve(h_est, knownTrainingSequence, mode='same')[:len_ts]
        
        # Residual
        e_t = r_ts - expected_signal
        noiseInterferenceEstimates.append(e_t)
    
    # Step 2: Sample covariance matrix Q_sample
    noise_matrix = np.array(noiseInterferenceEstimates)  # (NR x len_ts)
    Q_sample = (noise_matrix @ noise_matrix.conj().T) / len_ts
    
    # Step 3: Apply shrinkage method
    if shrinkageMethod == 'oas':
        Q, rho = oas_shrinkage_covariance(Q_sample, n_samples=len_ts)
        # Optional: log shrinkage coefficient for analysis
        # print(f"[OAS] rho={rho:.4f}, cond(Q_sample)={np.linalg.cond(Q_sample):.2f}, cond(Q_oas)={np.linalg.cond(Q):.2f}")
        
    elif shrinkageMethod == 'rblw':
        Q, rho = rblw_shrinkage_covariance(Q_sample, n_samples=len_ts)
        # print(f"[RBLW] rho={rho:.4f}")
        
    elif shrinkageMethod == 'diagonal_loading':
        # Legacy method: Q = (1-α)*Q_sample + α*F, then diagonal loading
        tr_Q = np.trace(Q_sample).real
        F = (tr_Q / NR) * np.eye(NR, dtype=complex)
        
        # Shrinkage
        Q_shrink = (1 - shrinkageFactor) * Q_sample + shrinkageFactor * F
        
        # Additional diagonal loading (legacy)
        loadingFactor = 0.2  # Fixed value from previous implementation
        Q = (1 - loadingFactor) * Q_shrink + loadingFactor * F
        
    elif shrinkageMethod == 'none':
        Q = Q_sample
        
    else:
        raise ValueError(f"Unknown shrinkageMethod: {shrinkageMethod}")
    
    return Q

def irc_corrected_process(
    received_signals_antennas,
    channel_estimates_antennas,
    known_training_sequence,
    shrinkageMethod='oas',
    shrinkage=0.1,
    loading_factor=0.2
):
    N_R = len(received_signals_antennas)
    L = len(channel_estimates_antennas[0])
    
    Q = estimate_noise_covariance_improved(
        received_signals_antennas,
        known_training_sequence,
        channel_estimates_antennas,
        shrinkageMethod=shrinkageMethod,
        shrinkageFactor=shrinkage
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