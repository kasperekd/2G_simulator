# Import functions from modules 
from config import validator, loader, extract_parameters
from transceiver.modulator import Modulator
from channel.quadriga import load_quadriga_channel
from channel.gen_channel import generate_cir
from channel.extract_channel import extract_cir
from visualisation.result import plot_results, save_results_to_csv
from core.ber import calculate_ber
from core.burst_info import get_burst_parameters

import numpy as np
import time
import sys
import glob
from pathlib import Path

def simulate(config):
    start_time = time.perf_counter()
    (
        num_interferers, channel_mat_file, channel_memory, 
        target_ratio_range_db, modulation_type, calculation_mode, 
        channel_estimation_method, num_bursts, channel_model, 
        traceback_depth, bs_nf_db, temp_k, fs_hz, burst_symbol_rate,
        combining_mode, irc_regularization,
        apply_saic_preprocessing, saic_method, saic_regularization, saic_thermal_noise_variance
    ) = extract_parameters.extract_config_parameters(config)
    (
        tail_bits, num_data_bits_per_burst, training_sequence_len, 
        training_sequence, guard_period
    ) = get_burst_parameters(burst_symbol_rate, modulation_type)

    a, _, _ = generate_cir(channel_model=channel_model,
                           carrier_frequency=fs_hz,
                              num_rx_ant=2,
                              num_tx_ant=2)
    
    channels_cir = extract_cir(a)
    h11 = channels_cir['h11']
    h12 = channels_cir['h12']
    h21 = channels_cir['h21']
    h22 = channels_cir['h22']
    
    # uncomment for using channel from quadriga
    # h11, h12, h21, h22 = load_quadriga_channel(channel_mat_file)
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
    print(f"SAIC preprocessing enabled: {apply_saic_preprocessing}")
    if apply_saic_preprocessing:
        print(f"SAIC method:                {saic_method}")
        print(f"SAIC regularization:        {saic_regularization}")
        print(f"SAIC thermal noise var.:    {saic_thermal_noise_variance}")
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
            combining_mode, irc_regularization,
            apply_saic_preprocessing, saic_method, saic_regularization, saic_thermal_noise_variance
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
    # Check for command line arguments
    compare_mode = False
    comparison_files = []
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--compare':
            compare_mode = True
            comparison_files = sys.argv[2:]
            
            if not comparison_files:
                print("Error: --compare flag requires file paths or patterns")
                print("Usage: python core_simulation.py --compare file1.csv file2.csv")
                print("   or: python core_simulation.py --compare ./results/*.csv")
                print("   or: python core_simulation.py --compare ./*")
                sys.exit(1)
            
            # Expand glob patterns and collect CSV files
            expanded_files = []
            for filepath in comparison_files:
                # Check if it's a glob pattern
                if any(char in filepath for char in ['*', '?', '[']):
                    # Expand glob pattern
                    matches = glob.glob(filepath)
                    if not matches:
                        print(f"Warning: glob pattern '{filepath}' matched no files")
                        continue
                    # Filter to CSV files only
                    csv_matches = [f for f in matches if f.endswith('.csv')]
                    expanded_files.extend(csv_matches)
                else:
                    # Regular file path
                    if not Path(filepath).exists():
                        print(f"Error: File not found: {filepath}")
                        sys.exit(1)
                    if filepath.endswith('.csv'):
                        expanded_files.append(filepath)
            
            # Remove duplicates and sort
            expanded_files = sorted(list(set(expanded_files)))
            
            if not expanded_files:
                print("Error: No CSV files found to compare")
                sys.exit(1)
            
            print(f"Found {len(expanded_files)} CSV file(s) to compare:")
            for f in expanded_files:
                print(f"  - {f}")
            
            print("=" * 80)
            print("COMPARISON MODE - Loading Results")
            print("=" * 80)
            plot_results(comparison_files=expanded_files)
            return
    
    # Normal simulation mode
    config_path = "./config/settings.json"
    config = validator.validate_config(loader.ConfigLoader.load(config_path))
    # --------------------------------------------------------------------

    ratio_values, ber_values = simulate(config)

    print('\nFinal BER Results:')
    print("-" * 80)
    for r, ber in zip(ratio_values, ber_values):
        print(f'{config.mode_selection.calculation_mode}={r:4.1f} dB => BER={ber:.6f}')
    print("-" * 80)
    
    # Save results if enabled
    if config.results_output.save_results:
        output_path = config.results_output.output_directory
        csv_filepath = save_results_to_csv(ratio_values, ber_values, config, output_path)
        print(f"CSV file saved: {csv_filepath}")
    
    plot_results(ratio_values, ber_values, config)

if __name__ == '__main__':
    main()