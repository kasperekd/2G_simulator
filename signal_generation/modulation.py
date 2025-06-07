import numpy as np


def modulation(signal_type, bit_sequence, sampling_rate):
    """
    Generates a modulated signal using the specified modulation type.

    Supported modulation types:
    - GMSK: Gaussian Minimum Shift Keying (with Gaussian filter and phase integration)
    - 8PSK: 8-Phase Shift Keying - uses 8 equally spaced constellation points on a circle, 
            encoding 3 bits per symbol with phase shifts of π/4 (45°) between points
    - QPSK: Quadrature Phase Shift Keying
    - QAM16: 16-level Quadrature Amplitude Modulation

    Args:
        signal_type (str): Type of modulation ("GMSK", "QPSK", or "QAM16").
        bit_sequence (np.ndarray): Input binary sequence as a 1D NumPy array of 0s and 1s.
        sampling_rate (int): Number of samples per symbol (oversampling factor). Must be ≥ 0.


    Returns:
        np.ndarray: Modulated signal (complex-valued for QPSK/QAM16, real-valued for GMSK).

    Raises:
        Exception: If sampling_rate < 0 or if signal_type is invalid.
    """
    if sampling_rate < 0:
        raise Exception("Signal generation - incorrect value of sampling rate < 0!")

    modulated_signal = []
    match signal_type:
        case "GMSK":
            """
            bt (float): Bandwidth-time product for GMSK (default: 0.3).
            filter_length (int): Length of Gaussian filter in symbols (default: 4).
            """
            bandwidth_time=0.3
            filter_length=4

            modulated_signal = 2 * np.array(bit_sequence) - 1

            upsampled_signal = np.repeat(modulated_signal, sampling_rate)

            t = np.linspace(-filter_length / 2, filter_length / 2, filter_length * sampling_rate)
            gaussian_filter = np.exp(-(t ** 2) / (2 * (bandwidth_time ** 2)))
            gaussian_filter /= np.sum(gaussian_filter)

            filtered_signal = np.convolve(upsampled_signal, gaussian_filter, mode = "same")

            phase = np.cumsum(filtered_signal) * (np.pi / (2 * sampling_rate))

            gmsk_signal = np.exp(1j * phase).real

            return gmsk_signal
        
        case "8PSK":
            
            symbols = np.array([bit_sequence]).reshape(-1, 3)

            symbol_mapping = {
                (0,0,0): np.exp(1j * 0),
                (0,0,1): np.exp(1j * np.pi/4),
                (0,1,1): np.exp(1j * np.pi/2),
                (0,1,0): np.exp(1j * 3*np.pi/4),
                (1,1,0): np.exp(1j * np.pi),
                (1,1,1): np.exp(1j * 5*np.pi/4),
                (1,0,1): np.exp(1j * 3*np.pi/2),
                (1,0,0): np.exp(1j * 7*np.pi/4)
            }
            
            signal = np.array([symbol_mapping[tuple(symbol)] for symbol in symbols])
            
            upsampled_signal = np.zeros(len(signal) * sampling_rate, dtype=complex)
            upsampled_signal[::sampling_rate] = signal
            
            pulse_shape = np.ones(sampling_rate) 
            
            filtered_signal = np.convolve(upsampled_signal, pulse_shape, mode='same')
            
            return filtered_signal

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
            bits = np.array(bit_sequence)
            symbols = bits.reshape(-1, 4)

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
            return qam16_signal
        
        case _:
            raise Exception("Signal generation - type of modulation is incorrectly specified!")