import numpy as np

def build_correlation_matrix(L, decay_factor=0.95):
    power_profile = np.array([decay_factor**i for i in range(L)])
    power_profile = power_profile / np.sum(power_profile)
    R_hh = np.diag(power_profile)
    return R_hh

def estimate_channel_ls(received_ts, known_ts, L):
    len_ts = len(known_ts)

    S = np.zeros((len_ts, L), dtype=complex)
    for i in range(len_ts):
        for j in range(L):
            if i - j >= 0:
                S[i, j] = known_ts[i - j]

    try:
        inv_STS = np.linalg.inv(S.conj().T @ S)
    except np.linalg.LinAlgError:
        inv_STS = np.linalg.pinv(S.conj().T @ S)
        
    h_ls = inv_STS @ S.conj().T @ received_ts[:len_ts]
    
    return h_ls, inv_STS

def estimate_channel_lmmse(received_ts, known_ts, L, snr_db=10.0):
    h_ls, inv_STS = estimate_channel_ls(received_ts, known_ts, L)

    R_hh = build_correlation_matrix(L, decay_factor=0.95)
    
    snr_linear = 10**(snr_db / 10.0)
    sigma_n2 = 1.0 / (snr_linear + 1e-6)

    noise_covariance = inv_STS * sigma_n2

    try:
        correction_term = np.linalg.pinv(R_hh + noise_covariance)
        W = R_hh @ correction_term
    except np.linalg.LinAlgError:
        W = np.eye(L) # Fallback

    h_lmmse = W @ h_ls
    
    return h_lmmse

def calculate_mse(h_true, h_est):
    min_len = min(len(h_true), len(h_est))
    err = h_true[:min_len] - h_est[:min_len]
    return np.mean(np.abs(err)**2)

def estimate_interference_metric(r_ts, training_sequence, h_est, L):
    # Используем матрицу Toeplitz (Можно использовать просто свёртку)
    X = np.zeros((len(r_ts), L), dtype=complex)
    for i in range(L):
        X[i:i+len(training_sequence), i] = training_sequence
        
    y_hat = X @ h_est

    e = r_ts - y_hat

    P_signal = np.linalg.norm(y_hat)**2
    P_interf = np.linalg.norm(e)**2

    eta = P_interf / (P_signal + P_interf + 1e-12)

    return eta, P_signal, P_interf