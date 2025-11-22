# Import functions from modules 
from config import validator, loader, extract_parameters
from transceiver.modulator import Modulator
from channel.quadriga import load_quadriga_channel
from visualisation.result import plot_results
from core.ber import calculate_ber
from core.burst_info import get_burst_parameters

import numpy as np
import time

def simulate(config):
    start_time = time.perf_counter()
    (
        num_interferers, channel_mat_file, channel_memory, 
        target_ratio_range_db, modulation_type, calculation_mode, 
        channel_estimation_method, num_bursts, channel_model, 
        traceback_depth, bs_nf_db, temp_k, fs_hz, burst_symbol_rate,
        combining_mode, irc_regularization
    ) = extract_parameters.extract_config_parameters(config)
    (
        tail_bits, num_data_bits_per_burst, training_sequence_len, 
        training_sequence, guard_period
    ) = get_burst_parameters(burst_symbol_rate, modulation_type)

    h11, h12, h21, h22 = load_quadriga_channel(channel_mat_file)
    L = channel_memory
    modem = Modulator(modulation_type)
    training_sequence = modem.modulate(training_sequence)

    ratio_values, ber_values = np.zeros(len(target_ratio_range_db)), np.zeros(len(target_ratio_range_db))

    print("=" * 80)
    print("SIMULATION CONFIGURATION")
    print("=" * 80)
    print(f"Modulation Type:            {modulation_type}")
    print(f"Channel Model:              {channel_model}")
    print(f"Channel Memory:             {channel_memory}")
    print(f"Calculation Mode:           {calculation_mode}")
    print(f"Channel Estimation Method:  {channel_estimation_method}")
    print(f"Combining Mode:             {combining_mode}")
    if combining_mode == "IRC":
        print(f"IRC Regularization:         {irc_regularization}")
    print(f"Number of Interferers:      {num_interferers}")
    print(f"Number of Bursts:           {num_bursts}")
    print(f"Burst Symbol Rate:          {burst_symbol_rate}")
    print(f"Base Station Noise Figure:  {bs_nf_db} dB")
    print(f"Temperature:                {temp_k} K")
    print(f"Sampling Frequency:         {fs_hz:.2f} Hz")
    print(f"Target Ratio Range:         {target_ratio_range_db[0]:.1f} to {target_ratio_range_db[-1]:.1f} dB (step: {target_ratio_range_db[1] - target_ratio_range_db[0]:.1f} dB)")
    print("=" * 80)
    print("STARTING SIMULATION")
    print("=" * 80)

    for i, target_ratio_db in enumerate(target_ratio_range_db):
        base_args = (
            target_ratio_db, num_interferers, h11, h12, h21, h22, L, modem,
            training_sequence, bs_nf_db, temp_k, fs_hz, calculation_mode,
            channel_estimation_method, training_sequence_len, traceback_depth,
            num_data_bits_per_burst, tail_bits, guard_period, config, channel_model,
            combining_mode, irc_regularization
        )
        
        ber = calculate_ber(base_args, num_bursts, target_ratio_db)
        ratio_values[i] = target_ratio_db
        ber_values[i] = ber
        print(f"  {calculation_mode} = {target_ratio_db:5.1f} dB, BER = {ber:.6f}")


    elapsed = time.perf_counter() - start_time
    print("=" * 80)
    print(f"Simulation completed in {elapsed:.2f} seconds")
    print("=" * 80)

    return np.array(ratio_values), np.array(ber_values)


def main():
    config_path = "./config/settings.json"
    config = validator.validate_config(loader.ConfigLoader.load(config_path))
    # --------------------------------------------------------------------

    ratio_values, ber_values = simulate(config)

    print('\nFinal BER Results:')
    print("-" * 80)
    for r, ber in zip(ratio_values, ber_values):
        print(f'{config.mode_selection.calculation_mode}={r:4.1f} dB => BER={ber:.6f}')
    print("-" * 80)
    plot_results(ratio_values, ber_values, config)

if __name__ == '__main__':
    main()