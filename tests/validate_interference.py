import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transceiver.modulator import Modulator
from transceiver.interference import interference_generation
from receiver.scaling_and_combing import scaling_and_combing

def validate_power_scaling():
    print("TEST 1: Power Scaling Validation")
    
    L = 3 
    len_tx_burst = 156
    num_interferers = 1
    modem = Modulator('BPSK')
    
    # Создаем фиктивный полезный сигнал
    s1_rx_ant1 = np.random.randn(len_tx_burst + L - 1) + 1j * np.random.randn(len_tx_burst + L - 1)
    s1_rx_ant2 = np.random.randn(len_tx_burst + L - 1) + 1j * np.random.randn(len_tx_burst + L - 1)
    
    # Создаем фиктивные каналы для помехи
    num_ch_realizations = 10
    h21 = (np.random.randn(L, num_ch_realizations) + 1j * np.random.randn(L, num_ch_realizations)) / np.sqrt(2)
    h22 = (np.random.randn(L, num_ch_realizations) + 1j * np.random.randn(L, num_ch_realizations)) / np.sqrt(2)
    
    channel_idx = 0
    
    # Interference
    int_ant1, int_ant2 = interference_generation(
        s1_rx_ant1, num_interferers, h21, h22, L, channel_idx, modem, len_tx_burst
    )
    
    # Scaling
    target_ci_db = 0.0
    bs_nf_db = 0
    fs_hz = 1e6
    temp_k = 300
    
    rx1, rx2 = scaling_and_combing(
        s1_rx_ant1, s1_rx_ant2, target_ci_db, 'CI', bs_nf_db, fs_hz, temp_k,
        int_ant1.copy(), int_ant2.copy()
    )
    
    # rx = s + i_scaled => i_scaled = rx - s
    i_scaled_1 = rx1 - s1_rx_ant1
    i_scaled_2 = rx2 - s1_rx_ant2
    
    # Measure
    pow_signal = np.mean(np.abs(s1_rx_ant1)**2 + np.abs(s1_rx_ant2)**2)
    pow_interf = np.mean(np.abs(i_scaled_1)**2 + np.abs(i_scaled_2)**2)
    
    measured_ci = 10 * np.log10(pow_signal / pow_interf)
    
    print(f"Target C/I: {target_ci_db} dB")
    print(f"Measured C/I: {measured_ci:.4f} dB")
    
    if np.abs(measured_ci - target_ci_db) < 0.1:
        print("[PASS] Power scaling is correct.")
    else:
        print("[FAIL] Power scaling error is too large!")

def validate_correlation_fixed():
    print("\nTEST 2: Signal-Interference Independence")

    L = 5
    len_tx_burst = 1000 
    modem = Modulator('BPSK')
    
    np.random.seed(1111)
    s_bits = np.random.randint(0, 2, len_tx_burst)
    s_syms = modem.modulate(s_bits)
    
    h_flat = np.zeros((L, 1), dtype=complex); h_flat[0,0] = 1.0
    
    s1_rx = np.concatenate([s_syms, np.zeros(L-1)])
    
    int_rx, _ = interference_generation(
        s1_rx, 1, h_flat, h_flat, L, 0, modem, len_tx_burst
    )
    
    min_len = min(len(s1_rx), len(int_rx))
    
    # огибающая
    corr_mag = np.corrcoef(np.abs(s1_rx[:min_len]), np.abs(int_rx[:min_len]))[0, 1]
    
    # фаза
    corr_real = np.corrcoef(np.real(s1_rx[:min_len]), np.real(int_rx[:min_len]))[0, 1]
    
    print(f"Magnitude Correlation: {corr_mag:.4f}")
    print(f"Real Part Correlation: {corr_real:.4f}")
    
    diff = np.abs(s1_rx[:min_len] - int_rx[:min_len])
    is_identical = np.all(diff < 1e-10)
    
    if is_identical:
        print("[FAIL] Signals are IDENTICAL!")
    elif corr_real > 0.5:
        print("[FAIL] High correlation!")
    else:
        print("[PASS] Signal and Interference are uncorrelated.")

if __name__ == "__main__":
    validate_power_scaling()
    validate_correlation_fixed()