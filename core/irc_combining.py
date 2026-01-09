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
        Q = (1 - shrinkageFactor) * Q_sample + shrinkageFactor * F
        
    elif shrinkageMethod == 'none':
        Q = Q_sample
        
    else:
        raise ValueError(f"Unknown shrinkageMethod: {shrinkageMethod}")
    
    return Q

def adaptive_diagonal_loading(Q_sample, target_condition_number=10.0):
    """
    Compute optimal loading factor to achieve target condition number.
    
    Args:
        Q_sample: Sample covariance matrix (NR x NR)
        target_condition_number: Desired cond(Q) after regularization (default: 10.0)
    
    Returns:
        Q_reg: Regularized covariance matrix
        alpha: Loading factor used
        metrics: Dict with diagnostics
    """
    NR = Q_sample.shape[0]
    
    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(Q_sample)
    lambda_max = eigvals[-1].real
    lambda_min = eigvals[0].real
    current_cond = lambda_max / (lambda_min + 1e-10)
    
    # If already well-conditioned, no loading needed
    if current_cond <= target_condition_number:
        metrics = {
            'cond_before': current_cond,
            'cond_after': current_cond,
            'alpha': 0.0,
            'method': 'no_loading_needed'
        }
        return Q_sample, 0.0, metrics
    
    # Shrinkage target: scaled identity
    tr_Q = np.trace(Q_sample).real
    avg_eigval = tr_Q / NR
    
    # Binary search for optimal alpha
    alpha_low, alpha_high = 0.0, 1.0
    alpha_opt = 0.0
    
    for iteration in range(20):
        alpha = (alpha_low + alpha_high) / 2
        
        # Regularized eigenvalues: lambda_reg = (1-alpha)*lambda + alpha*avg
        lambda_reg_min = (1 - alpha) * lambda_min + alpha * avg_eigval
        lambda_reg_max = (1 - alpha) * lambda_max + alpha * avg_eigval
        cond_reg = lambda_reg_max / (lambda_reg_min + 1e-10)
        
        if cond_reg > target_condition_number:
            alpha_low = alpha  # Need more regularization
        else:
            alpha_high = alpha
        
        alpha_opt = alpha
        
        # Early stopping if close enough
        if abs(cond_reg - target_condition_number) < 0.5:
            break
    
    # Apply loading
    F = avg_eigval * np.eye(NR, dtype=complex)
    Q_reg = (1 - alpha_opt) * Q_sample + alpha_opt * F
    
    # Final metrics
    cond_final = np.linalg.cond(Q_reg)
    metrics = {
        'cond_before': current_cond,
        'cond_after': cond_final,
        'alpha': alpha_opt,
        'method': 'adaptive',
        'eigvals_before': eigvals.tolist(),
        'eigvals_after': np.linalg.eigvalsh(Q_reg).tolist()
    }
    
    return Q_reg, alpha_opt, metrics

def irc_corrected_process(
    received_signals_antennas,
    channel_estimates_antennas,
    known_training_sequence,
    shrinkageMethod='oas',           # Stage 1: 'oas', 'rblw', 'diagonal_loading', 'none'
    loadingMethod='adaptive',        # Stage 2: 'adaptive', 'legacy', 'none'
    target_condition_number=10.0,    # For adaptive loading
    shrinkage=0.1,                   # For diagonal_loading (Stage 1)
    loading_factor=0.2,              # For legacy loading (Stage 2)
    verbose=False
):
    """
    IRC combining with two-stage regularization.
    
    Stage 1 (Shrinkage): Statistical correction via shrinkageMethod
    Stage 2 (Loading): Numerical stabilization via loadingMethod
    
    Args:
        shrinkageMethod: 'oas', 'rblw', 'diagonal_loading', 'none'
        loadingMethod: 'adaptive', 'legacy', 'none'
        target_condition_number: Target cond(Q) for adaptive loading
        shrinkage: Shrinkage factor for 'diagonal_loading' method
        loading_factor: Loading factor for 'legacy' loading
        verbose: Print diagnostics
    """
    N_R = len(received_signals_antennas)
    L = len(channel_estimates_antennas[0])
    
    # Shrinkage
    Q_shrunk = estimate_noise_covariance_improved(
        received_signals_antennas,
        known_training_sequence,
        channel_estimates_antennas,
        shrinkageMethod=shrinkageMethod,
        shrinkageFactor=shrinkage
    )
    
    if verbose:
        cond_shrunk = np.linalg.cond(Q_shrunk)
        print(f"[IRC Stage 1 - {shrinkageMethod}] cond(Q) = {cond_shrunk:.2f}")
    
    # Loading
    if loadingMethod == 'adaptive':
        current_cond = np.linalg.cond(Q_shrunk)
        
        if current_cond > target_condition_number:
            Q_reg, alpha_opt, metrics = adaptive_diagonal_loading(
                Q_shrunk, 
                target_condition_number=target_condition_number
            )
            if verbose:
                print(f"[IRC Stage 2 - Adaptive] Applied loading: alpha={alpha_opt:.4f}, "
                      f"cond {metrics['cond_before']:.1f} → {metrics['cond_after']:.1f}")
        else:
            Q_reg = Q_shrunk
            if verbose:
                print(f"[IRC Stage 2 - Adaptive] No loading needed (cond={current_cond:.1f} < target={target_condition_number:.1f})")
    
    elif loadingMethod == 'legacy':
        avg_noise_power = np.trace(Q_shrunk).real / N_R
        Q_reg = (1 - loading_factor) * Q_shrunk + loading_factor * avg_noise_power * np.eye(N_R, dtype=complex)
        
        if verbose:
            cond_after = np.linalg.cond(Q_reg)
            print(f"[IRC Stage 2 - Legacy] Loading factor={loading_factor:.2f}, "
                  f"cond {cond_shrunk:.1f} → {cond_after:.1f}")
    
    elif loadingMethod == 'none':
        Q_reg = Q_shrunk
        if verbose:
            print(f"[IRC Stage 2 - None] No loading applied")
    
    else:
        raise ValueError(f"Unknown loadingMethod: {loadingMethod}")
    
    try:
        Q_inv = np.linalg.inv(Q_reg)
    except np.linalg.LinAlgError:
        if verbose:
            print("[IRC Warning] Q_reg singular, using pseudoinverse")
        Q_inv = np.linalg.pinv(Q_reg)
    
    # Compute IRC filters
    irc_filters = []
    for i in range(N_R):
        g_i = np.zeros(L, dtype=complex)
        for j in range(N_R):
            v_ij = Q_inv[i, j]
            h_j_matched = channel_estimates_antennas[j][::-1].conj()
            g_i += v_ij * h_j_matched
        irc_filters.append(g_i)
    
    # Apply IRC
    signal_length = len(received_signals_antennas[0])
    combined = np.zeros(signal_length, dtype=complex)
    
    for i in range(N_R):
        filtered = np.convolve(received_signals_antennas[i], irc_filters[i], mode='same')
        combined += filtered
    
    # Effective channel
    conv_len = 2 * L - 1
    h_eff_full = np.zeros(conv_len, dtype=complex)
    for i in range(N_R):
        conv = np.convolve(channel_estimates_antennas[i], irc_filters[i], mode='full')
        h_eff_full += conv
    
    start_idx = L // 2
    h_eff = h_eff_full[start_idx : start_idx + L]
    
    return combined, h_eff
