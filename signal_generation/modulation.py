import numpy as np

def modulation(signal_type, bit_sequence, sampling_rate):
    """
    Generates a modulated signal using the specified modulation type.

    Supported modulation types:
    - GMSK: Gaussian Minimum Shift Keying
    - QPSK: Quadrature Phase Shift Keying
    - QAM16: 16-level Quadrature Amplitude Modulation

    Args:
        signal_type (str): Type of modulation ("GMSK", "QPSK", or "QAM16").
        bit_sequence (np.ndarray): Input binary sequence as a 1D NumPy array of 0s and 1s.
        sampling_rate (int): Number of samples per symbol (oversampling factor). Must be ≥ 0.

    Returns:
        np.ndarray: Complex-valued modulated signal or real-valued (for GMSK).

    Raises:
        Exception: If sampling_rate < 0 or if signal_type is invalid.
    """
    if sampling_rate < 0:
        raise Exception("Signal generation - incorrect value of sampling rate < 0!")

    modulated_signal = []
    match signal_type:
        case "GMSK":
            modulated_signal = 2 * np.array(bit_sequence) - 1
            upsample_modulated_signal = np.repeat(modulated_signal, sampling_rate)
            return upsample_modulated_signal
        
        case "QPSK":
            modulated_signal = 2 * np.array(bit_sequence) - 1
            
            filter = np.ones(sampling_rate)

            i_signal, q_signal = modulated_signal[0::2], modulated_signal[1::2]

            i_oversampling = np.zeros(len(i_signal) * sampling_rate)
            q_oversampling = np.zeros(len(q_signal) * sampling_rate)

            i_oversampling[::sampling_rate] = i_signal
            q_oversampling[::sampling_rate] = q_signal

            i_filtered = np.convolve(i_oversampling, filter, mode='same')
            q_filtered = np.convolve(q_oversampling, filter, mode='same')

            qpsk_signal = i_filtered + 1j * q_filtered
            return qpsk_signal
        
        case "QAM16":
            filter = np.ones(sampling_rate)

            symbols = bit_sequence.reshape(-1, 4)

            i_bits = symbols[:, :2]
            q_bits = symbols[:, 2:]

            amplitude = {
                (0, 0): -3,
                (0, 1): -1,
                (1, 0): 1,
                (1, 1): 3
            }

            i_signal = np.array([amplitude[tuple(bits)] for bits in i_bits])
            q_signal = np.array([amplitude[tuple(bits)] for bits in q_bits])

            i_oversampling = np.zeros(len(i_signal) * sampling_rate)
            q_oversampling = np.zeros(len(q_signal) * sampling_rate)

            i_oversampling[::sampling_rate] = i_signal
            q_oversampling[::sampling_rate] = q_signal

            i_filtered = np.convolve(i_oversampling, filter, mode='same')
            q_filtered = np.convolve(q_oversampling, filter, mode='same')

            qam16_signal = i_filtered + 1j * q_filtered
            print(qam16_signal)
            return qam16_signal
        
        case _:
            raise Exception("Signal generation - type of modulation is incorrectly specified!")