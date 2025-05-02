import numpy as np

def modulation(signal_type, bit_sequence, sampling_rate):
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