# Import functions from modules 
import csv
from datetime import datetime
from config import validator, loader, extract_parameters
from transceiver.modulator import Modulator
from channel.quadriga import load_quadriga_channel
from channel.gen_channel_new import generate_multiple_cir
from channel.extract_channel import extract_cir
from visualisation.result import plot_results, save_results_to_csv
from core.ber import calculate_ber
from core.burst_info import get_burst_parameters
from receiver.power_calc import (
    calculate_received_power_dbm,
    calculate_thermal_noise_power_dbm
)
from multiprocessing import Pool, cpu_count

import numpy as np
import time
import sys
import glob
from pathlib import Path

def save_mse_comparison(ratios, mse_ls, mse_lmmse, config):
    output_dir = Path(config.results_output.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"mse_debug_{timestamp}.csv"
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['CI_dB', 'MSE_LS', 'MSE_LMMSE', 'Gain_dB'])
        for r, mls, mlmmse in zip(ratios, mse_ls, mse_lmmse):
            gain = 10 * np.log10(mls / (mlmmse + 1e-20)) if mlmmse > 0 else 0
            writer.writerow([r, mls, mlmmse, gain])
    print(f"MSE Debug stats saved to: {filepath}")

def simulate(config):
    start_time = time.perf_counter()
    (
        num_interferers, channel_mat_file, channel_memory, 
        target_ratio_range_db, modulation_type, calculation_mode, 
        channel_estimation_method, num_bursts, channel_model, 
        traceback_depth, bs_nf_db, temp_k, fs_hz, burst_symbol_rate,
        combining_mode, irc_regularization,
        apply_saic_preprocessing, saic_method, saic_regularization, saic_thermal_noise_variance,
        apply_temporal_whitening, temporal_method, temporal_regularization, temporal_thermal_noise_variance, temporal_full_burst,
        bs_tx_power_dbm, bs_antenna_gain_dbi, ms_antenna_gain_dbi, path_loss_db, channel_bandwidth_hz
    ) = extract_parameters.extract_config_parameters(config)

    (
        tail_bits, num_data_bits_per_burst, training_sequence_len, 
        training_sequence, guard_period
    ) = get_burst_parameters(burst_symbol_rate, modulation_type)

    # channel memory length used when constructing simple AWGN channels
    L = channel_memory

    # Generate channel impulse responses using the internal generator.
    # Special-case AWGN: treat as single-tap (no multipath) channel.
    if channel_model.upper() == 'AWGN':
        # create trivial single-tap channels with 1.0 gain on first tap
        # shape: (L, 1) so downstream code can select a time index
        h11 = np.zeros((L, 1), dtype=complex)
        h12 = np.zeros((L, 1), dtype=complex)
        h21 = np.zeros((L, 1), dtype=complex)
        h22 = np.zeros((L, 1), dtype=complex)
        h11[0, 0] = 1.0
        h12[0, 0] = 1.0
        h21[0, 0] = 1.0
        h22[0, 0] = 1.0
    else:
        # set channel memory size used for shaping AWGN channels if needed
        L = channel_memory

        H_list, tau = generate_multiple_cir(
            num_realizations=4,
            num_processes=4,
            channel_model=channel_model,
            channel_taps=channel_taps,
            frequency_s=fs_hz,
        )

        channels_cir = extract_cir(H_list)
        h11 = channels_cir['h11']
        h12 = channels_cir['h12']
        h21 = channels_cir['h21']
        h22 = channels_cir['h22']
    
    # uncomment for using channel from quadriga
    # h11, h12, h21, h22 = load_quadriga_channel(channel_mat_file)
    modem = Modulator(modulation_type)
    training_sequence = modem.modulate(training_sequence)

    ratio_values = np.zeros(len(target_ratio_range_db))
    ber_values = np.zeros(len(target_ratio_range_db))
    mse_ls_values = np.zeros(len(target_ratio_range_db))
    mse_lmmse_values = np.zeros(len(target_ratio_range_db))
    print("=" * 80)
    print(f"STARTING SIMULATION with {cpu_count()} processes")
    print("SIMULATION CONFIGURATION")
    print("=" * 80)
    print(f"Modulation Type:            {modulation_type}")
    print(f"Channel Model:              {channel_model}")
    print(f"Channel Taps:               {channel_taps}")
    print(f"Channel Memory:             {channel_memory}")
    print(f"Calculation Mode:           {calculation_mode}")
    print(f"Channel Estimation Method:  {channel_estimation_method}")
    print(f"Combining Mode:             {combining_mode}")
    if combining_mode == "IRC":
        print(f"IRC Regularization:         {irc_regularization}")
    if combining_mode == "ST-IRC":
        st_irc_method = getattr(config.mode_selection, 'st_irc_method', 'ar-prewhitening')
        print(f"ST-IRC Method:              {st_irc_method}")
        print(f"IRC Regularization:         {irc_regularization}")
    print(f"SAIC preprocessing enabled: {apply_saic_preprocessing}")
    if apply_saic_preprocessing:
        print(f"SAIC method:                {saic_method}")
        print(f"SAIC regularization:        {saic_regularization}")
        print(f"SAIC thermal noise var.:    {saic_thermal_noise_variance}")
    print(f"Temporal whitening enabled: {apply_temporal_whitening}")
    if apply_temporal_whitening: 
        print(f"Temporal method:            {temporal_method}")
        print(f"Temporal regularization:    {temporal_regularization}")
        print(f"Temporal thermal noise var.:{temporal_thermal_noise_variance}")
    print(f"Number of Interferers:      {num_interferers}")
    print(f"Number of Bursts:           {num_bursts}")
    print(f"Burst Symbol Rate:          {burst_symbol_rate}")
    print(f"Base Station Noise Figure:  {bs_nf_db} dB")
    print(f"Temperature:                {temp_k} K")
    print(f"Sampling Frequency:         {fs_hz:.2f} Hz")
    print(f"Target Ratio Range:         {target_ratio_range_db[0]:.1f} to {target_ratio_range_db[-1]:.1f} dB (step: {target_ratio_range_db[1] - target_ratio_range_db[0]:.1f} dB)")
    print("=" * 80)
    print("POWER PARAMETERS (dBm)")
    print("=" * 80)
    print(f"BS TX Power:                {bs_tx_power_dbm:.1f} dBm")
    print(f"BS Antenna Gain:            {bs_antenna_gain_dbi:.1f} dBi")
    print(f"MS Antenna Gain:            {ms_antenna_gain_dbi:.1f} dBi")
    print(f"Path Loss:                  {path_loss_db:.1f} dB")
    print(f"Channel Bandwidth:          {channel_bandwidth_hz/1000:.0f} kHz")
    
    # Calculate and display received power
    rx_power_dbm = calculate_received_power_dbm(
        bs_tx_power_dbm,
        bs_antenna_gain_dbi,
        ms_antenna_gain_dbi,
        path_loss_db
    )
    rx_noise_dbm = calculate_thermal_noise_power_dbm(
        channel_bandwidth_hz,
        temp_k,
        bs_nf_db
    )
    print(f"Received Signal Power:      {rx_power_dbm:.1f} dBm")
    print(f"Received Noise Power:       {rx_noise_dbm:.1f} dBm")
    result_snr_db = rx_power_dbm - rx_noise_dbm
    print(f"Resulting SNR (thermal):    {result_snr_db:.1f} dB")
    print("=" * 80)
    print("STARTING SIMULATION")
    print("=" * 80)

    num_workers = cpu_count()
    
    with Pool(processes=num_workers) as pool:
        for i, target_ratio_db in enumerate(target_ratio_range_db):
            base_args = (
                seed, snr, ci, target_ratio_db, num_interferers, h11, h12, h21, h22, L, modem,
                training_sequence, bs_nf_db, temp_k, fs_hz, calculation_mode,
                channel_estimation_method, training_sequence_len, traceback_depth,
                num_data_bits_per_burst, tail_bits, guard_period, config, channel_model,
                combining_mode, irc_regularization,
                apply_saic_preprocessing, saic_method, saic_regularization, saic_thermal_noise_variance,
                apply_temporal_whitening, temporal_method, temporal_regularization, temporal_thermal_noise_variance, temporal_full_burst,
                bs_tx_power_dbm, bs_antenna_gain_dbi, ms_antenna_gain_dbi, path_loss_db, channel_bandwidth_hz
            )
            
            ber, avg_mse_ls, avg_mse_lmmse = calculate_ber(pool, base_args, num_bursts, target_ratio_db)
            
            ratio_values[i] = target_ratio_db
            ber_values[i] = ber
            mse_ls_values[i] = avg_mse_ls
            mse_lmmse_values[i] = avg_mse_lmmse
            print(f"  {calculation_mode} = {target_ratio_db:5.1f} dB, BER = {ber:.6f} | MSE(LS)={avg_mse_ls:.4f}, MSE(LMMSE)={avg_mse_lmmse:.4f}")


    elapsed = time.perf_counter() - start_time
    print("=" * 80)
    print(f"Simulation completed in {elapsed:.2f} seconds")
    print("=" * 80)

    if config.results_output.save_mse_debug:
        save_mse_comparison(ratio_values, mse_ls_values, mse_lmmse_values, config)

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
        if sys.argv[1] == '--sweep-channels':
            # Load config first to read sweep settings
            config_path = "./config/settings.json"
            config = validator.validate_config(loader.ConfigLoader.load(config_path))

            # Allow an optional list of models after the flag; else use config
            sweep_args = [a for a in sys.argv[2:] if not a.startswith('-')]
            if sweep_args:
                models_to_run = sweep_args
            else:
                models_to_run = config.channel_sweep.models

            if not models_to_run:
                print("No channel models specified for sweep.")
                sys.exit(1)

            # If user supplied --sweep-plot anywhere, plot combined results at the end
            should_plot = '--sweep-plot' in sys.argv

            results_paths = []
            results_for_plot = []

            for model in models_to_run:
                print(f"Running sweep for model: {model}")

                cfg_local = config.copy(deep=True)
                cfg_local.core_simulation_parameters = cfg_local.core_simulation_parameters.copy(update={"channel_model": model})

                ratio_values, ber_values = simulate(cfg_local)

                if cfg_local.results_output.save_results:
                    csv_filepath = save_results_to_csv(ratio_values, ber_values, cfg_local, cfg_local.results_output.output_directory)
                    results_paths.append(csv_filepath)
                    print(f"CSV file saved: {csv_filepath}")
                if should_plot:
                    results_for_plot.append((ratio_values, ber_values, cfg_local))

            if should_plot and results_for_plot:
                # plot all results together
                for r, b, c in results_for_plot:
                    plot_results(r, b, c)

            print("Sweep completed.")
            if results_paths:
                print("Saved CSV files:")
                for p in results_paths:
                    print(f"  - {p}")
            return
    
    # Normal simulation mode
    config_path = "./config/settings.json"
    config = validator.validate_config(loader.ConfigLoader.load(config_path))

    # If channel sweep is enabled in config, run sweep automatically (no plotting)
    if getattr(config, 'channel_sweep', None) and getattr(config.channel_sweep, 'enabled', False):
        models_to_run = config.channel_sweep.models
        results_paths = []

        print("Automatic channel sweep enabled in config. Running sweep for models:")
        for model in models_to_run:
            print(f" - {model}")
            cfg_local = config.copy(deep=True)
            cfg_local.core_simulation_parameters = cfg_local.core_simulation_parameters.copy(update={"channel_model": model})

            ratio_values, ber_values = simulate(cfg_local)
            if cfg_local.results_output.save_results:
                csv_filepath = save_results_to_csv(ratio_values, ber_values, cfg_local, cfg_local.results_output.output_directory)
                results_paths.append(csv_filepath)

        print("Sweep finished.")
        if results_paths:
            print("Saved CSV files:")
            for p in results_paths:
                print(f"  - {p}")

        return
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