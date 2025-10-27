# ==============================================================================
# 1. MODULATION CLASS
# ==============================================================================
import numpy as np

class Modulator:
    def __init__(self, modulation_type='QPSK'):
        self.type = modulation_type
        # TODO: Swap BPSK for GMSK
        if self.type == 'BPSK':
            self.bits_per_symbol = 1
            self.constellation = np.array([-1, 1])
        elif self.type == 'QPSK':
            self.bits_per_symbol = 2
            self.constellation = np.array([
                1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j
            ]) / np.sqrt(2)
        elif self.type == '8PSK':
            self.bits_per_symbol = 3
            self.constellation = np.exp(1j * np.pi / 4 * np.arange(8))
            print(self.constellation)
        elif self.type == 'QAM16':
            self.bits_per_symbol = 4
            self.constellation = np.array([
                -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
                -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
                 1 - 3j,  1 - 1j,  1 + 1j,  1 + 3j,
                 3 - 3j,  3 - 1j,  3 + 1j,  3 + 3j
            ]) / np.sqrt(10)
        # TODO: Add 32QAM and a proper GMSK implementation.
        else:
            raise ValueError(f"Unsupported modulation type: {self.type}")
        self.num_symbols = len(self.constellation)
        self._map_bits_to_idx = {
            tuple(map(int, np.binary_repr(i, width=self.bits_per_symbol))): i
            for i in range(self.num_symbols)
        }
        self._map_idx_to_bits = {v: k for k, v in self._map_bits_to_idx.items()}

    def modulate(self, bits):
        if len(bits) % self.bits_per_symbol != 0:
            bits = np.append(
                bits,
                np.zeros(
                    self.bits_per_symbol - (len(bits) % self.bits_per_symbol),
                    dtype=int
                )
            )
        symbols = [
            self.constellation[
                self._map_bits_to_idx[tuple(bits[i:i + self.bits_per_symbol])]
            ]
            for i in range(0, len(bits), self.bits_per_symbol)
        ]
        return np.array(symbols)

    def demodulate_indices(self, indices):
        bit_list = [
            bit for idx in indices for bit in self._map_idx_to_bits[idx]
        ]
        return np.array(bit_list)
