# ==============================================================================
# 1. MODULATION CLASS
# ==============================================================================
import numpy as np
from scipy.special import erfc

def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))

def gmsk_modulator(bits):
    #params
    oversamp = 16 # samples per bit
    L = 3 # filter memory
    f = 270.83333e3 
    T = 1/f #interval
    h = 0.5 #modulation coeff
    BT = 0.30 #coeff for GSM
    B = BT / T
    N = len(bits) # bits count

    # bits -> NRZ (1 = 1, 0 = -1)
    NRZ_bits = bits * 2 - 1
    # print(f"NRZ bits: {NRZ_bits}")

    # NRZ -> complex symbols
    a_prev = np.concatenate(([1], NRZ_bits[:-1]))
    alpha = NRZ_bits * a_prev
    
    b = np.empty(len(alpha), dtype=complex)
    prev = 1+0j
    for n, al in enumerate(alpha):
        prev = 1j * al * prev
        b[n] = prev

    # overasmpling symbols
    b_os = np.zeros(len(b) * oversamp, dtype=complex)
    b_os[::oversamp] = b

    #timeline
    t = np.arange(-(L*T/2), (L*T/2), T/oversamp)
    t = t + (T/oversamp)/2

    # gauss impulse
    g = (
        qfunc(2*np.pi*B*(t - T/2) / np.sqrt(np.log(2))) -
        qfunc(2*np.pi*B*(t + T/2) / np.sqrt(np.log(2)))
    )

    g = g / np.sum(g) * (np.pi / 2)
    g = np.concatenate(([0.0], g))

    # plt.plot(g)
    # plt.title("f(t)")
    # plt.xlabel("sample")
    # plt.ylabel("Hz")
    # plt.show()

    # integral (phase)
    q = np.cumsum(g)
    
    # plt.plot(q)
    # plt.title("phi(t)")
    # plt.xlabel("sample")
    # plt.ylabel("rad")
    # plt.show()

    s = np.zeros(2*len(g) - 1)

    for i in range(len(g)):
        s[i] = np.sin(q[i]) / np.sin(np.pi*h)

    for i in range(len(g), 2*len(g) - 1):
        s[i] = np.sin(np.pi*h - q[i - (len(g) - 1)]) / np.sin(np.pi*h)

    # plt.plot(s)
    # plt.title("Non-linear coeffs")
    # plt.show()

    # Compute C0 pulse: valid for all L values
    c0 = s[:len(s) - oversamp*(L-1)]
    for i in range(1, L):
        c0 = c0 * s[i*oversamp : len(s) - oversamp*(L-1-i)]

    # Generate OpenBTS pulse
    numSamples = len(c0)
    centerPoint = (numSamples - 1) / 2
    i = (np.arange(numSamples + 1) - centerPoint) / oversamp
    xP = 0.96 * np.exp(-1.1380*i**2 - 0.527*i**4)
    xP = xP / np.max(xP) * np.max(c0)

    h0 = 0.5 * (xP[oversamp:] - xP[:-oversamp])

    signal = np.convolve(b_os, h0, mode='full')
    
    return signal


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
            self.bits_per_symbol = 2
            self.constellation = np.array([
                1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j
            ]) / np.sqrt(2)
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
            symbols = gmsk_modulator(np.array(bits))
            return np.array(symbols)
        
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
