import numpy as np
from scipy.signal import convolve
from channel.awgn import add_awgn

def simulate_reception(
    modulated_signal_s1: np.ndarray,
    modulated_signal_s2: np.ndarray,
    h11: np.ndarray,
    h12: np.ndarray,
    h21: np.ndarray,
    h22: np.ndarray,
    SNR_db: float,
    num_tests: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """Simulates signal reception at a base station with two antennas in multipath 
    and interference conditions.

    Models the reception of desired and interfering signals through multipath channels,
    with AWGN added according to specified SNR. Each channel realization is processed
    separately to evaluate performance under different channel conditions.

    Args:
        modulated_signal_s1: Desired signal waveform (complex-valued array)
        modulated_signal_s2: Interfering signal waveform (complex-valued array)
        h11: Channel impulse responses between TX1 and RX1 (shape: 7 x 1000)
        h12: Channel impulse responses between TX1 and RX2 (shape: 7 x 1000)
        h21: Channel impulse responses between TX2 and RX1 (shape: 7 x 1000)
        h22: Channel impulse responses between TX2 and RX2 (shape: 7 x 1000)
        SNR_db: Desired signal-to-noise ratio in decibels
        num_tests: Number of channel realizations to simulate (default: 1000)

    Returns:
        tuple: Two numpy arrays containing:
            - received_antenna1: Signals received at antenna 1 (shape: num_tests x N+6)
            - received_antenna2: Signals received at antenna 2 (shape: num_tests x N+6)
            where N is length of input signals and 6 comes from 7-tap channel - 1

    Raises:
        AssertionError: If input dimensions are incompatible
        ValueError: If SNR is negative or other invalid parameters
    """
    # Validate input dimensions
    assert h11.shape == (7, num_tests), "h11 must have shape (7, num_tests)"
    assert modulated_signal_s1.shape == modulated_signal_s2.shape, "Input signals must have same length"
    
    signal_length = len(modulated_signal_s1)
    result_length = signal_length + 6  # 6 = 7 (channel length) - 1
    
    # Initialize output arrays
    received_antenna1 = np.zeros((num_tests, result_length), dtype=np.complex128)
    received_antenna2 = np.zeros((num_tests, result_length), dtype=np.complex128)
    
    # Process each channel realization
    for i in range(num_tests):
        # Convolve signals with channels
        s1_h11 = convolve(modulated_signal_s1, h11[:, i], mode='full')
        s2_h21 = convolve(modulated_signal_s2, h21[:, i], mode='full')
        
        s1_h12 = convolve(modulated_signal_s1, h12[:, i], mode='full')
        s2_h22 = convolve(modulated_signal_s2, h22[:, i], mode='full')
        
        # Combine signals and add noise
        received_antenna1[i] = add_awgn(s1_h11 + s2_h21, SNR_db)
        received_antenna2[i] = add_awgn(s1_h12 + s2_h22, SNR_db)
    
    return received_antenna1, received_antenna2