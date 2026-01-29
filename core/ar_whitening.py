import numpy as np
from scipy.linalg import cholesky, solve_triangular
from core.irc_combining import estimate_noise_covariance_improved

def form_spacetime_noise_vector(residuals_list, M=1):
    st_noise = []
    st_noise.extend(residuals_list)
    
    for m in range(1, M + 1):
        for res in residuals_list:
            delayed = np.concatenate([np.zeros(m, dtype=complex), res[:-m]])
            st_noise.append(delayed)
            
    return st_noise

def ar_prewhitening_process(
    received_signals,
    channel_estimates,
    known_training_sequence,
    M_taps=1,
    loading_factor=0.2
):
    """
    ST-IRC AR model

    1. Estimate Noise Covariance R_nn.
    2. Compute Cholesky factor L (R_nn = L * L^H).
    3. Whitening filter W = L^-1.
    4. Apply W to Signal and Channel.
    5. Perform MRC on whitened data.
    """
    N_ant = len(received_signals)
    L_ch = len(channel_estimates[0])
    
    # 1
    noise_residuals = []
    len_ts = len(known_training_sequence)
    
    for i in range(N_ant):
        est_sig = np.convolve(channel_estimates[i], known_training_sequence, mode='same')
        est_sig = est_sig[:len_ts]
        noise = received_signals[i][:len_ts] - est_sig
        noise_residuals.append(noise)
        
    st_noise_vec = form_spacetime_noise_vector(noise_residuals, M_taps)
    
    E_matrix = np.array(st_noise_vec)
    R_nn = (E_matrix @ E_matrix.conj().T) / len_ts
    
    N_st = R_nn.shape[0]
    avg_pow = np.trace(R_nn).real / N_st
    R_nn_reg = (1 - loading_factor) * R_nn + loading_factor * avg_pow * np.eye(N_st, dtype=complex)
    
    # 2
    try:
        # R = L @ L.H 
        L_chol = cholesky(R_nn_reg, lower=True)
    except np.linalg.LinAlgError:
        L_chol = np.eye(N_st, dtype=complex) * np.sqrt(avg_pow)

    off_diag_energy = np.sum(np.abs(R_nn_reg)**2) - np.sum(np.abs(np.diag(R_nn_reg))**2)
    diag_energy = np.sum(np.abs(np.diag(R_nn_reg))**2)
    
    color_factor = off_diag_energy / (diag_energy + 1e-9)
    
    # If the interference is white
    if color_factor < 0.1:
        W_whitening = np.eye(N_st, dtype=complex)
    else:
        W_whitening = np.linalg.inv(L_chol)
    
    # 3
    from core.st_irc_combining import form_spacetime_vector, form_spacetime_channels
    
    st_signals = form_spacetime_vector(received_signals, M_taps)
    st_channels = form_spacetime_channels(channel_estimates, M_taps)
    
    st_signals_matrix = np.array(st_signals)
    whitened_signals_matrix = W_whitening @ st_signals_matrix
    
    st_channels_matrix = np.array(st_channels)
    whitened_channels_matrix = W_whitening @ st_channels_matrix
    
    # 4
    
    combined_signal = np.zeros(received_signals[0].shape[0], dtype=complex)
    h_eff_acc = np.zeros(whitened_channels_matrix.shape[1] * 2, dtype=complex)
    
    num_whitened_channels = whitened_channels_matrix.shape[0]
    
    for i in range(num_whitened_channels):
        w_sig = whitened_signals_matrix[i]
        w_h = whitened_channels_matrix[i]
        
        mf_response = w_h[::-1].conj()
        
        branch_out = np.convolve(w_sig, mf_response, mode='same')
        combined_signal += branch_out
        
        # h_eff = sum ( h_i * h_i_matched )
        branch_h_eff = np.convolve(w_h, mf_response, mode='full')
        h_eff_acc[:len(branch_h_eff)] += branch_h_eff
        
    target_len = L_ch + M_taps
    
    center = np.argmax(np.abs(h_eff_acc))
    start = max(0, center - target_len // 2)
    end = start + target_len
    
    h_eff_final = h_eff_acc[start:end]
    
    h_eff_final /= (np.max(np.abs(h_eff_final)) + 1e-9)
    
    return combined_signal, h_eff_final