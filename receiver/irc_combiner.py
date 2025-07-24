import numpy as np
from scipy.signal import convolve

def estimate_noise_covariance(rx_signal_matrix: np.ndarray, h_est_matrix: np.ndarray, train_seq: np.ndarray) -> np.ndarray:
    """
    Estimates the spatial noise/interference covariance matrix Q.
    Based on formulas (16) and (17) from the source paper.
    
    Args:
        rx_signal_matrix (np.ndarray): Matrix of received signals on TS part [N_ant, N_samples_ts].
        h_est_matrix (np.ndarray): Matrix of estimated channel responses [N_ant, L].
        train_seq (np.ndarray): The known training sequence.
        
    Returns:
        np.ndarray: The estimated covariance matrix Q [N_ant, N_ant].
    """
    num_antennas, L = h_est_matrix.shape
    
    # Reconstruct the "clean" signal from the desired user using the estimated channel
    reconstructed_ts_matrix = np.zeros_like(rx_signal_matrix, dtype=np.complex128)
    for i in range(num_antennas):
        reconstructed_ts_matrix[i, :] = convolve(train_seq, h_est_matrix[i, :], mode='full')

    # Estimate the noise + interference by subtracting the reconstructed signal
    error_matrix = rx_signal_matrix - reconstructed_ts_matrix
    
    # Calculate the covariance matrix Q
    Q = np.cov(error_matrix)
    
    return Q

def calculate_irc_parameters(rx_signal_matrix: np.ndarray, h_matrix: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the parameters ŷn and Sn for the MLSE using IRC.
    Based on formulas (24) and (25) from the source paper.
    
    Args:
        rx_signal_matrix (np.ndarray): Matrix of all received signals [N_ant, N_samples].
        h_matrix (np.ndarray): Matrix of CIRs [N_ant, L].
        Q (np.ndarray): The noise covariance matrix [N_ant, N_ant].
        
    Returns:
        tuple[np.ndarray, np.ndarray]:
            - y_hat (ŷn): The combined samples after passing through IRC filters.
            - S_n: The auto-correlation of the combined post-filtering channel.
    """
    num_antennas, L = h_matrix.shape
    
    Q_inv = np.linalg.inv(Q)
    
    # Form the IRC filters
    g_matrix = Q_inv @ h_matrix

    # Calculate y_hat (ŷn) by convolving each antenna's signal with its corresponding IRC filter
    y_hat = np.zeros(rx_signal_matrix.shape[1] + L - 1, dtype=np.complex128)
    for i in range(num_antennas):
        irc_filter = np.conj(np.flip(g_matrix[i, :]))
        y_hat += convolve(rx_signal_matrix[i, :], irc_filter, mode='full')
        
    # Calculate S_n, the effective channel autocorrelation after IRC
    S_n_full = np.zeros(2*L - 1, dtype=np.complex128)
    for i in range(num_antennas):
        irc_filter = np.conj(np.flip(g_matrix[i, :]))
        S_n_full += convolve(h_matrix[i, :], irc_filter, mode='full')

    S_n_out = S_n_full[L-1 : L-1 + L]

    return y_hat, S_n_out