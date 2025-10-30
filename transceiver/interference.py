from .generate_data import generate_data_bits
from scipy.signal import convolve
import numpy as np

def interference_generation(s1_rx_ant1, num_interferers, h21, h22, L, channel_idx, modem, len_tx_burst):
    max_len = len(s1_rx_ant1)
    total_interf_rx_ant1 = np.zeros(max_len, dtype=complex)
    total_interf_rx_ant2 = np.zeros(max_len, dtype=complex)
    for _ in range(num_interferers):
        interf_bits = generate_data_bits(len_tx_burst)
        interf_syms = modem.modulate(interf_bits)

        h_interf_ant1 = h21[:L, channel_idx]
        h_interf_ant2 = h22[:L, channel_idx]

        interf_conv1 = convolve(interf_syms, h_interf_ant1, 'full')
        interf_conv2 = convolve(interf_syms, h_interf_ant2, 'full')
        
        total_interf_rx_ant1[:len(interf_conv1)] += interf_conv1
        total_interf_rx_ant2[:len(interf_conv2)] += interf_conv2

    return total_interf_rx_ant1, total_interf_rx_ant2