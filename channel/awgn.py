import numpy as np

def add_noise(signal: np.array, snr_db: int):
    """
    Add AWGN with proper energy calculations for different modulation types.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (complex-valued)
    snr_db : float
        Signal-to-noise ratio in dB
    
    Returns:
        Noisy signal
    """
    snr = 10**(snr_db/10)
    power = np.mean(np.abs(signal)**2)
    noise_power = power / snr
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise