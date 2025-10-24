# Import functions from modules 
from config import validator, loader, extract_parameters
from transceiver.modulator import Modulator
from transceiver.burst import create_burst
from transceiver.generate_data import generate_data_bits
from channel.noise import add_thermal_noise
from channel.quadriga import load_quadriga_channel
from receiver.channel_estimation import estimate_channel_ls
from receiver.viterbi import mlse_viterbi_decode
from visualisation.result import plot_results

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from typing import Dict
from multiprocessing import Pool, cpu_count
import time

# ==============================================================================
# 2. HELPER AND SIMULATION FUNCTIONS
# ==============================================================================


def single_iteration(param):
    (target_ratio_db, num_interferers, h11, h12, h21, h22, L, modem,
     training_sequence, num_bursts, bs_nf_db, temp_k, fs_hz, calculation_mode,
     channel_estimation_method, training_sequence_len, traceback_depth,
     config, num_data_symbols_per_burst) = param
    
    total_errors, total_bits = 0, 0
    for _ in range(num_bursts):
        # 1. TRANSMITTER SIDE
        data_bits = generate_data_bits(num_data_symbols_per_burst * modem.bits_per_symbol)
        tx_burst, original_data_symbols = create_burst(data_bits, modem, training_sequence, L)

        # 2. CHANNEL PROPAGATION
        channel_idx = np.random.randint(0, h11.shape[1])
        # channel_idx = 0
        h_true_ant1 = h11[:L, channel_idx]
        h_true_ant2 = h12[:L, channel_idx]
        s1_rx_ant1 = convolve(tx_burst, h_true_ant1, 'full')
        s1_rx_ant2 = convolve(tx_burst, h_true_ant2, 'full')

        # 3. INTERFERENCE GENERATION
        max_len = len(s1_rx_ant1)
        total_interf_rx_ant1 = np.zeros(max_len, dtype=complex)
        total_interf_rx_ant2 = np.zeros(max_len, dtype=complex)
        for _ in range(num_interferers):
            interf_bits = generate_data_bits(len(tx_burst) * modem.bits_per_symbol)
            interf_syms = modem.modulate(interf_bits)
            h_interf_ant1 = h21[:L, channel_idx]
            h_interf_ant2 = h22[:L, channel_idx]
            interf_conv1 = convolve(interf_syms, h_interf_ant1, 'full')
            interf_conv2 = convolve(interf_syms, h_interf_ant2, 'full')
            total_interf_rx_ant1[:len(interf_conv1)] += interf_conv1
            total_interf_rx_ant2[:len(interf_conv2)] += interf_conv2

        # 4. SCALING AND COMBINING
        signal_power = np.mean(np.abs(s1_rx_ant1) ** 2 + np.abs(s1_rx_ant2) ** 2)
        current_interf_power = np.mean(np.abs(total_interf_rx_ant1) ** 2 + np.abs(total_interf_rx_ant2) ** 2)
        if current_interf_power < 1e-20:
            current_interf_power = 1e-20

        target_ratio_linear = 10 ** (target_ratio_db / 10)

        if calculation_mode == 'CI':
            required_interf_power = signal_power / target_ratio_linear
        else:  # SINR
            k_b = 1.380649e-23
            nf_lin = 10 ** (bs_nf_db / 10)
            noise_power = k_b * temp_k * fs_hz * nf_lin * 2
            required_interf_power = (signal_power / target_ratio_linear) - noise_power
            if required_interf_power < 0:
                required_interf_power = 1e-20

        scaling_factor = np.sqrt(required_interf_power / current_interf_power)
        total_interf_rx_ant1 *= scaling_factor
        total_interf_rx_ant2 *= scaling_factor

        rx_ant1 = s1_rx_ant1 + total_interf_rx_ant1
        rx_ant2 = s1_rx_ant2 + total_interf_rx_ant2

        # 5. RECEIVER: Add noise
        rx_ant1_noisy = add_thermal_noise(rx_ant1, config)
        rx_ant2_noisy = add_thermal_noise(rx_ant2, config)

        # 6. RECEIVER: Channel Estimation
        if channel_estimation_method == 'true':
            h_est_ant1, h_est_ant2 = h_true_ant1, h_true_ant2
        else:  # 'ls'
            ts_start_idx = len(original_data_symbols) // 2 + L
            ts_end_idx = ts_start_idx + training_sequence_len
            h_est_ant1 = estimate_channel_ls(rx_ant1_noisy[ts_start_idx: ts_end_idx + L - 1], training_sequence, L)
            h_est_ant2 = estimate_channel_ls(rx_ant2_noisy[ts_start_idx: ts_end_idx + L - 1], training_sequence, L)

        # 7. RECEIVER: Equalization and Decoding
        rx_combined = (rx_ant1_noisy + rx_ant2_noisy) / 2
        h_est_avg = (h_est_ant1 + h_est_ant2) / 2

        mlse_input = rx_combined[:len(tx_burst) + L - 1]
        decoded_indices = mlse_viterbi_decode(mlse_input, h_est_avg, modem.constellation, traceback_depth)

        # 8. BER CALCULATION
        d1_len = len(original_data_symbols) // 2
        d2_len = len(original_data_symbols) - d1_len
        decoded_d1_indices = decoded_indices[L: L + d1_len]
        d2_start_idx = L + d1_len + training_sequence_len
        d2_end_idx = d2_start_idx + d2_len
        decoded_d2_indices = decoded_indices[d2_start_idx: d2_end_idx]
        decoded_data_indices = np.concatenate([decoded_d1_indices, decoded_d2_indices])
        decoded_bits = modem.demodulate_indices(decoded_data_indices)

        total_errors += np.sum(data_bits != decoded_bits[:len(data_bits)])
        total_bits += len(data_bits)

    ber = total_errors / total_bits if total_bits > 0 else 0.5
    return target_ratio_db, ber

def simulate(config):
    np.random.seed(111)
    start_time = time.perf_counter()
    num_interferers, channel_mat_file, channel_memory, range_db, modulation_type, calculation_mode, channel_estimation_method, num_bursts, channel_model, num_data_symbols_per_burst, training_sequence_len, traceback_depth, training_sequence, bs_nf_db, temp_k, fs_hz = extract_parameters.extract_config_parameters(config)
    h11, h12, h21, h22 = load_quadriga_channel(channel_mat_file)
    L = channel_memory
    modem = Modulator(modulation_type)
    # FIXME: random TS for BPSK. Now we have GMSK TS for all modulation
    ts_indices = np.random.randint(
            0, len(modem.constellation), training_sequence_len
        )
    training_sequence = modem.constellation[ts_indices]
    target_ratio_range_db = np.arange(range_db[0], range_db[1], range_db[2])

    ratio_values, ber_values = np.zeros(len(target_ratio_range_db)), np.zeros(len(target_ratio_range_db))

    print("Starting simulation:")
    print(
        f"Modulation: {modulation_type}, "
        f"Mode: {calculation_mode}, "
        f"Estimation: {channel_estimation_method}"
    )    
    params_list = []
    for target_ratio_db in target_ratio_range_db:
        params_list.append((
            target_ratio_db, num_interferers, h11, h12, h21, h22, L, modem,
            training_sequence, num_bursts, bs_nf_db, temp_k, fs_hz, calculation_mode,
            channel_estimation_method, training_sequence_len, traceback_depth,
            config, num_data_symbols_per_burst
        ))

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(single_iteration, params_list)

    for i, (ratio, ber) in enumerate(sorted(results, key=lambda x: x[0])):
        ratio_values[i] = ratio
        ber_values[i] = ber
        print(f"  {calculation_mode} = {ratio:5.1f} dB, BER = {ber:.6f}")

    elapsed = time.perf_counter() - start_time
    print(f"Simulation completed with multiprocessing in {elapsed:.2f} seconds")

    return np.array(ratio_values), np.array(ber_values)

# TODO LIST:
# integrate config (+)
# adding parallel processing(+)
# allocation of functions to modules (+)
# adding burst types for others modulation(+-)
def main():
    config_path = "./config/settings.json"
    config = validator.validate_config(loader.ConfigLoader.load(config_path))
    # ----------------------------------------------------------------- ---

    ratio_values, ber_values = simulate(config)

    print('\nFinal BER Results:')
    for r, ber in zip(ratio_values, ber_values):
        print(f'{config.mode_selection.calculation_mode}={r:4.1f} dB => BER={ber:.6f}')
    plot_results(ratio_values, ber_values, config)


if __name__ == '__main__':
    main()