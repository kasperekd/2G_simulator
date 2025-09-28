import numpy as np

def estimate_channel_ls(rx_signal_on_ts: np.ndarray, train_seq: np.ndarray, L: int) -> np.ndarray:
    """
    Estimates the channel impulse response (CIR) using the Least Squares (LS) method.
    This version uses the normal equations for a robust and stable solution.

    Args:
        rx_signal_on_ts (np.ndarray): The received signal segment corresponding to the
                                      transmission of the training sequence. 
                                      Should have length len(train_seq) + L - 1.
        train_seq (np.ndarray): The known training sequence (complex symbols).
        L (int): The expected length of the channel impulse response to estimate.
        
    Returns:
        np.ndarray: The estimated CIR (h_est).
    """
    N_ts = len(train_seq)
    
    # Ensure the received signal has the expected length
    expected_len = N_ts + L - 1
    if len(rx_signal_on_ts) != expected_len:
        raise ValueError(
            f"Length of rx_signal_on_ts is {len(rx_signal_on_ts)}, "
            f"but expected {expected_len} (N_ts + L - 1)."
        )

    # Create the convolution matrix 'A' from the training sequence
    A = np.zeros((expected_len, L), dtype=np.complex128)
    for i in range(L):
        A[i:i+N_ts, i] = train_seq

    # Solve the normal equations: h = (A^H * A)^-1 * A^H * y
    # This approach is numerically more stable than taking the pinv of A directly.
    A_H = A.conj().T
    
    # This is a small (L, L) matrix, e.g., 7x7
    A_H_A = A_H @ A
    
    # This is a vector of shape (L,)
    A_H_y = A_H @ rx_signal_on_ts
    
    # Invert the small (L, L) matrix and solve for h.
    # Using pinv here is safe and handles potential singularity if the TS has poor autocorrelation.
    h_est = np.linalg.pinv(A_H_A) @ A_H_y
    
    return h_est

def estimate_channel_corr(rx_signal_on_ts: np.ndarray, train_seq: np.ndarray, Lh: int, OSR: int) -> np.ndarray:
    """
    Estimates the channel impulse response (CIR) using the correlate method.
    This version uses the normal equations for a robust and stable solution.

    Args:
        rx_signal_on_ts (np.ndarray): The received signal segment corresponding to the
                                      transmission of the training sequence. 
                                      Should have length len(train_seq) + L - 1.
        train_seq (np.ndarray): The known training sequence (complex symbols).
        L (int): The expected length of the channel impulse response to estimate.
        OSR (int): The oversample rate fs/rb, where rb: sample rate, fs: frequency 
        
    Returns:
        np.ndarray: The estimated CIR (h_est).
        m_max(int): Where burst start 
    """
    N_ts = len(train_seq)
    
    # Ensure the received signal has the expected length
    expected_len = N_ts + Lh - 1
    if len(rx_signal_on_ts) != expected_len:
        raise ValueError(
            f"Length of rx_signal_on_ts is {len(rx_signal_on_ts)}, "
            f"but expected {expected_len} (N_ts + L - 1)."
        )

    center_start = len(train_seq) // 2 + 8
    center_end = center_start + 16
    tseq_c = train_seq[center_start : center_end]

    zeros = np.zeros(5, dtype=train_seq.dtype)
    tseq_e = np.concatenate([zeros, tseq_c, zeros])

    v = np.correlate(rx_signal_on_ts, np.conj(tseq_c[::-1]), mode='full')
                                # tseq_e
    energy = np.abs(v) ** 2

    L = OSR * 2 * Lh
    window_energy = np.array([np.sum(energy[m: m + L]) for m in range(len(energy) - L + 1)])

    m_max = np.argmax(window_energy)

    h_est = v[m_max : m_max + Lh]

    return h_est, m_max