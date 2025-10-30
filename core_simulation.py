# Import functions from modules 
from config import validator, loader, extract_parameters
from transceiver.modulator import Modulator
from transceiver.burst import create_burst
from transceiver.generate_data import generate_data_bits
from transceiver.interference import interference_generation
from channel.quadriga import load_quadriga_channel
from receiver.noise import add_thermal_noise
from receiver.scaling_and_combing import scaling_and_combing
from receiver.channel_estimation import estimate_channel_ls
from receiver.viterbi import mlse_viterbi_decode
from receiver.DeMUX import extract_data_segments
from visualisation.result import plot_results

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from typing import Dict
from multiprocessing import Pool, cpu_count
import time

def single_iteration(param):
    (target_ratio_db, num_interferers, h11, h12, h21, h22, L, modem,
            training_sequence, num_bursts, bs_nf_db, temp_k, fs_hz, calculation_mode,
            channel_estimation_method, training_sequence_len, traceback_depth,
            num_data_bits_per_burst, tail_bits, guard_period, config, channel_model) = param
    
    # NOTE: for repeatability, uncomment the line below
    # np.random.seed(123)

    total_errors, total_bits = 0, 0
    for _ in range(num_bursts):
        # 1. TRANSMITTER SIDE
        data_bits = generate_data_bits(num_data_bits_per_burst)
        tx_burst, original_data_symbols, tail_symbols = create_burst(data_bits, modem, training_sequence, tail_bits, guard_period)

        # 2. CHANNEL PROPAGATION
        channel_idx = np.random.randint(0, h11.shape[1])
        h_true_ant1 = h11[:L, channel_idx]
        h_true_ant2 = h12[:L, channel_idx]
        s1_rx_ant1 = convolve(tx_burst, h_true_ant1, 'full')
        s1_rx_ant2 = convolve(tx_burst, h_true_ant2, 'full')

        # 3. INTERFERENCE GENERATION
        total_interf_rx_ant1, total_interf_rx_ant2 = interference_generation(s1_rx_ant1, num_interferers, h21, h22, L, channel_idx, modem, len(tx_burst))

        # 4. SCALING AND COMBINING
        rx_ant1, rx_ant2 = scaling_and_combing(s1_rx_ant1, s1_rx_ant2, target_ratio_db, calculation_mode, bs_nf_db, fs_hz, temp_k, total_interf_rx_ant1, total_interf_rx_ant2)

        # 5. RECEIVER: Add noise
        rx_ant1_noisy = add_thermal_noise(rx_ant1, bs_nf_db, fs_hz, temp_k)
        rx_ant2_noisy = add_thermal_noise(rx_ant2, bs_nf_db, fs_hz, temp_k)

        # 6. RECEIVER: Channel Estimation
        if channel_estimation_method == 'true':
            h_est_ant1, h_est_ant2 = h_true_ant1, h_true_ant2
        else:  # 'ls'
            ts_start_idx = len(original_data_symbols) // 2 + len(tail_symbols)
            ts_end_idx = ts_start_idx + len(training_sequence)
            h_est_ant1 = estimate_channel_ls(rx_ant1_noisy[ts_start_idx: ts_end_idx + L - 1], training_sequence, L)
            h_est_ant2 = estimate_channel_ls(rx_ant2_noisy[ts_start_idx: ts_end_idx + L - 1], training_sequence, L)

        # 7. RECEIVER: Equalization and Decoding
        # TODO: when it is implemented by another method, allocate it to the function
        rx_combined = (rx_ant1_noisy + rx_ant2_noisy) / 2
        h_est_avg = (h_est_ant1 + h_est_ant2) / 2

        mlse_input = rx_combined[:len(tx_burst) + L - 1]
        decoded_indices = mlse_viterbi_decode(mlse_input, h_est_avg, modem.constellation, traceback_depth)

        decoded_data_indices = extract_data_segments(original_data_symbols, decoded_indices, tail_symbols, training_sequence_len, modem, data_bits)
        decoded_bits = modem.demodulate_indices(decoded_data_indices)

        # 8. BER CALCULATION
        total_errors += np.sum(data_bits != decoded_bits[:len(data_bits)])
        total_bits += len(data_bits)

    ber = total_errors / total_bits if total_bits > 0 else 0.5
    return target_ratio_db, ber

def simulate(config):
    start_time = time.perf_counter()
    (num_interferers, channel_mat_file, channel_memory, 
            target_ratio_range_db, modulation_type, calculation_mode, 
            channel_estimation_method, num_bursts, channel_model, 
            num_data_bits_per_burst, training_sequence_len, traceback_depth,
            training_sequence, bs_nf_db, temp_k, fs_hz, tail_bits, guard_period) = extract_parameters.extract_config_parameters(config)
    h11, h12, h21, h22 = load_quadriga_channel(channel_mat_file)
    L = channel_memory
    modem = Modulator(modulation_type)
    training_sequence = modem.modulate(training_sequence)
    
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
            num_data_bits_per_burst, tail_bits, guard_period, config, channel_model
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

def main():
    config_path = "./config/settings.json"
    config = validator.validate_config(loader.ConfigLoader.load(config_path))
    # --------------------------------------------------------------------

    ratio_values, ber_values = simulate(config)

    print('\nFinal BER Results:')
    for r, ber in zip(ratio_values, ber_values):
        print(f'{config.mode_selection.calculation_mode}={r:4.1f} dB => BER={ber:.6f}')
    plot_results(ratio_values, ber_values, config)

if __name__ == '__main__':
    main()