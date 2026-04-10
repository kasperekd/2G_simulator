# ==============================================================================
# 1. MODULATION CLASS
# ==============================================================================
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc

def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))

def c0_generate(oversamp, L, BT, f=270.83333e3):
    T = 1 / f
    h = 0.5
    B = BT / T

    t = np.arange(-(L*T/2), (L*T/2), T/oversamp)
    t = t + (T/oversamp)/2

    g = (
        qfunc(2*np.pi*B*(t - T/2) / np.sqrt(np.log(2))) -
        qfunc(2*np.pi*B*(t + T/2) / np.sqrt(np.log(2)))
    )

    g = g / np.sum(g) * (np.pi / 2)
    g = np.concatenate(([0.0], g))

    q = np.cumsum(g)

    s = np.zeros(2*len(g) - 1)
    for i in range(len(g)):
        s[i] = np.sin(q[i]) / np.sin(np.pi*h)

    for i in range(len(g), 2*len(g) - 1):
        s[i] = np.sin(np.pi*h - q[i - (len(g) - 1)]) / np.sin(np.pi*h)

    c0 = s[:len(s) - oversamp*(L-1)]
    for i in range(1, L):
        c0 = c0 * s[i*oversamp : len(s) - oversamp*(L-1-i)]

    return c0


def gmsk_laurent_tx(bits, c0, oversamp):
    bits = np.asarray(bits)
    bits = 2*bits - 1 + 0j
    b_up = np.zeros(len(bits)*oversamp, dtype=complex)
    b_up[::oversamp] = bits
    s = np.convolve(b_up, c0, mode="same")

    s = s[::oversamp]

    
    if np.max(s) != 0:
        s /= np.max(s)
        s /= np.sqrt(2)

    return np.round(s,2)

oversamp = 16
L = 1
BT = 0.3
N = 1000
f = 270.83333e3
T = 1/f
dt = T/oversamp

class Modulator:
    def __init__(self, modulation_type='GMSK'):
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
        elif self.type == "GMSK":            
            self.bits_per_symbol = 1
            self.constellation = np.array([-1, 1] / np.sqrt(2))
        else:
            raise ValueError(f"Unsupported modulation type: {self.type}")
        self.num_symbols = len(self.constellation)
        self._map_bits_to_idx = {
            tuple(map(int, np.binary_repr(i, width=self.bits_per_symbol))): i
            for i in range(self.num_symbols)
        }
        self._map_idx_to_bits = {v: k for k, v in self._map_bits_to_idx.items()}

    def modulate(self, bits):
        if(self.type == "GMSK"):
            c0 = c0_generate(oversamp, L, BT, f)
            signal = gmsk_laurent_tx(bits, c0, oversamp)
            
            return np.array(signal)
        
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
