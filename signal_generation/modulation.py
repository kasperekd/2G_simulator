import numpy as np

def get_constellation(signal_type):
    if signal_type == "GMSK" or signal_type == "BPSK":
        return np.float64(np.array([-1, 1]))

    # elif signal_type == "8PSK":
    #     # 8 точек равномерно на окружности с шагом π/4
    #     angles = np.arange(0, 2*np.pi, 2*np.pi/8)
    #     return np.exp(1j * angles)

    # elif signal_type == "QPSK":
    #     levels = np.array([-1, 1])
    #     constellation = np.array([complex(i, q) for i in levels for q in levels])
    #     return constellation

    # elif signal_type == "QAM16":
    #     # Созвездие 16-QAM - решетка из 16 точек
    #     levels = np.array([-3, -1, 1, 3])
    #     constellation = np.array([complex(i, q) for i in levels for q in levels])
    #     return constellation

    else:
        raise Exception("Созвездие для данного типа модуляции не определено")

# class GMSK:
#     def __init__(self, sampling_rate, bt=0.3, filter_length=4):
#         self.sampling_rate = sampling_rate
#         self.bt = bt
#         self.filter_length = filter_length
#         self.constellation = get_constellation("GMSK")

#     def modulate(self, bit_sequence):
#         modulated_signal = 2 * np.array(bit_sequence) - 1
#         upsampled_signal = np.repeat(modulated_signal, self.sampling_rate)
#         t = np.linspace(-self.filter_length / 2, self.filter_length / 2, self.filter_length * self.sampling_rate)
#         gaussian_filter = np.exp(-(t ** 2) / (2 * (self.bt ** 2)))
#         gaussian_filter /= np.sum(gaussian_filter)
#         filtered_signal = np.convolve(upsampled_signal, gaussian_filter, mode="full")
#         phase = np.cumsum(filtered_signal) * (np.pi / (2 * self.sampling_rate))
#         gmsk_signal = np.exp(1j * phase).real
#         return gmsk_signal

#     def demodulate(self, received_signal):
#         # Простая демодуляция по знаку
#         return (received_signal > 0).astype(int)

class BPSK:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.constellation = get_constellation("BPSK")

    def modulate(self, bit_sequence):
        modulated_signal = 2 * np.array(bit_sequence) - 1
        upsampled_signal = np.repeat(modulated_signal, self.sampling_rate)
        pulse_shape = np.ones(self.sampling_rate) / np.sqrt(self.sampling_rate)
        bpsk_signal = np.convolve(upsampled_signal, pulse_shape, mode='same')

        # Индексы символов в созвездии (0 для -1, 1 для 1)
        symbol_indices = (modulated_signal > 0).astype(int)
        return bpsk_signal, symbol_indices


    def demodulate(self, received_signal):
        # Демодуляция по знаку реальной части
        return (received_signal > 0).astype(int)

# class PSK8:
#     def __init__(self, sampling_rate):
#         self.sampling_rate = sampling_rate
#         self.constellation = get_constellation("8PSK")
#         self.symbol_mapping = {tuple(map(int, np.binary_repr(i, 3))): self.constellation[i] for i in range(8)}
#         self.mapping_bits = {v: k for k, v in self.symbol_mapping.items()}

#     def modulate(self, bit_sequence):
#         bits = np.array(bit_sequence)
#         if len(bits) % 3 != 0:
#             raise ValueError("Длина бит должна быть кратна 3")
#         symbols_bits = bits.reshape(-1, 3)
#         symbols = np.array([self.symbol_mapping[tuple(b)] for b in symbols_bits])
#         # Апсемплирование
#         upsampled = np.repeat(symbols, self.sampling_rate)
#         pulse = np.ones(self.sampling_rate) / np.sqrt(self.sampling_rate)
#         signal = np.convolve(upsampled, pulse, mode='same')
        
#         # Для индексов символов - ищем позицию символа в созвездии
#         symbol_indices = np.array([np.argmin(np.abs(sym - self.constellation)) for sym in symbols])
        
#         return signal, symbol_indices

#     def demodulate(self, received_signal):
#         sampled = received_signal[::self.sampling_rate]
#         bits = []
#         for sym in sampled:
#             idx = np.argmin(np.abs(sym - self.constellation))
#             bits.extend(self.mapping_bits[self.constellation[idx]])
#         return np.array(bits, dtype=int)


# class QPSK:
#     def __init__(self, sampling_rate):
#         self.sampling_rate = sampling_rate
#         self.sampling_rate = sampling_rate
#         self.constellation = get_constellation("QPSK")

#     def modulate(self, bit_sequence):
#         modulated_signal = 2 * np.array(bit_sequence) - 1
#         filter = np.ones(self.sampling_rate)
#         i_signal, q_signal = modulated_signal[0::2], modulated_signal[1::2]
#         i_oversampling = np.zeros(len(i_signal) * self.sampling_rate)
#         q_oversampling = np.zeros(len(q_signal) * self.sampling_rate)
#         i_oversampling[::self.sampling_rate] = i_signal
#         q_oversampling[::self.sampling_rate] = q_signal
#         i_filtered = np.convolve(i_oversampling, filter, mode='full')
#         q_filtered = np.convolve(q_oversampling, filter, mode='full')
#         qpsk_signal = i_filtered + 1j * q_filtered
#         return qpsk_signal

#     def demodulate(self, received_signal):
#         sampled = received_signal[::self.sampling_rate]
#         bits = []
#         for sym in sampled:
#             distances = np.abs(sym - self.constellation)
#             idx = distances.argmin()
#             bits.extend([int(b) for b in format(idx, '02b')])
#         return np.array(bits, dtype=int)

# class QAM16:
#     def __init__(self, sampling_rate):
#         self.sampling_rate = sampling_rate
#         self.constellation = get_constellation("QAM16")
#         self.gray_codes = gray_code(2)
#         self.level_map = {'00': -3, '01': -1, '11': 1, '10': 3}
#         self.level_inv_map = {v: k for k, v in self.level_map.items()}

#     def modulate(self, bit_sequence):
#         bits = np.array(bit_sequence)
#         if len(bits) % 4 != 0:
#             raise ValueError("Длина бит должна быть кратна 4")
#         symbols_bits = bits.reshape(-1,4)
#         i_bits = symbols_bits[:, :2]
#         q_bits = symbols_bits[:, 2:]
#         i_signal = np.array([self.level_map[''.join(str(b) for b in bts)] for bts in i_bits])
#         q_signal = np.array([self.level_map[''.join(str(b) for b in bts)] for bts in q_bits])
#         symbols = i_signal + 1j * q_signal
#         upsampled = np.repeat(symbols, self.sampling_rate)
#         pulse = np.ones(self.sampling_rate)
#         return np.convolve(upsampled, pulse, mode='same')

#     def demodulate(self, received_signal):
#         sampled = received_signal[::self.sampling_rate]
#         bits = []
#         for sym in sampled:
#             idx = np.argmin(np.abs(sym - self.constellation))
#             i = np.real(self.constellation[idx])
#             q = np.imag(self.constellation[idx])
#             bits.extend([int(b) for b in self.level_inv_map[i]])
#             bits.extend([int(b) for b in self.level_inv_map[q]])
#         return np.array(bits, dtype=int)
    
def modulate_bits(signal_type, bit_sequence, sampling_rate):
    if sampling_rate < 0:
        raise Exception("Signal generation - incorrect value of sampling rate < 0!")
    
    modulators = {
        # "GMSK": GMSK,
        "BPSK": BPSK,
        # "8PSK": PSK8,
        # "QPSK": QPSK,
        # "QAM16": QAM16
    }
    
    if signal_type not in modulators:
        raise Exception("Signal generation - type of modulation is incorrectly specified!")
    
    ModulatorClass = modulators[signal_type]
    
    if signal_type == "GMSK":
        modulator = ModulatorClass(sampling_rate, bt=0.3, filter_length=4)
    else:
        modulator = ModulatorClass(sampling_rate)
    
    return modulator.modulate(bit_sequence)

def demodulate_symbols(signal_type, received_signal, sampling_rate):
    demodulators = {
        # "GMSK": GMSK,
        "BPSK": BPSK,
        # "8PSK": PSK8,
        # "QPSK": QPSK,
        # "QAM16": QAM16
    }

    if signal_type not in demodulators:
        raise Exception("Signal demodulation - type of modulation is incorrectly specified!")

    DemodulatorClass = demodulators[signal_type]

    if signal_type == "GMSK":
        demodulator = DemodulatorClass(sampling_rate, bt=0.3, filter_length=4)
    else:
        demodulator = DemodulatorClass(sampling_rate)

    # received_signal — комплексный сигнал после демодуляции/фильтрации, сэмплированный по символам
    bits = demodulator.demodulate(received_signal)
    return bits

