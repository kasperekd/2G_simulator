import numpy as np

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Additive White Gaussian Noise (AWGN) to the input signal.
    
    Parameters:
    signal : np.ndarray
        Input signal (can be complex-valued)
    snr_db : float
        Desired signal-to-noise ratio in decibels (dB)
    
    Returns:
    np.ndarray
        Noisy signal with AWGN added
    
    Notes:
    For complex signals, the noise is added independently to real and imaginary parts.
    The SNR is calculated as the ratio of signal power to noise power.
    """
    snr_linear = 10 ** (snr_db / 10)
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / snr_linear
    
    if np.iscomplexobj(signal):
        noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 
                1j*np.random.randn(*signal.shape))
    else:
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        
    return signal + noise