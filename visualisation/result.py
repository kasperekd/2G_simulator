import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from receiver.power_calc import (
    ratio_to_effective_signal_power_dbm,
    calculate_received_power_dbm,
    calculate_thermal_noise_power_dbm
)

def save_results_to_csv(ratio_values: np.ndarray, ber_values: np.ndarray, config: Any, output_path: str) -> str:
    """
    Save simulation results and configuration parameters to CSV file.
    
    Args:
        ratio_values: Array of target ratio values (dB)
        ber_values: Array of BER values
        config: SystemConfig object with simulation parameters
        output_path: Directory path where the CSV file will be saved
        
    Returns:
        str: Full path to the saved CSV file
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp and parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modulation = config.core_simulation_parameters.modulation_type
    combining = config.mode_selection.combining_mode
    channel = config.core_simulation_parameters.channel_model
    filename = f"results_{modulation}_{combining}_{channel}_{timestamp}.csv"
    filepath = output_dir / filename
    
    # Extract all configuration parameters for metadata
    core_params = config.core_simulation_parameters
    mode_params = config.mode_selection
    phy_params = config.physical_layer_parameters
    burst_params = config.burst_structure_parameters
    
    # Write CSV with metadata header
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Metadata section
        writer.writerow(['# SIMULATION METADATA'])
        writer.writerow(['Timestamp', timestamp])
        writer.writerow(['Modulation Type', core_params.modulation_type])
        writer.writerow(['Channel Model', core_params.channel_model])
        writer.writerow(['Channel Memory', core_params.channel_memory])
        writer.writerow(['Calculation Mode', mode_params.calculation_mode])
        writer.writerow(['Channel Estimation Method', mode_params.channel_estimation_method])
        writer.writerow(['Combining Mode', mode_params.combining_mode])
        if mode_params.combining_mode == "IRC":
            writer.writerow(['IRC Regularization', mode_params.irc_regularization])
        writer.writerow(['Number of Interferers', config.num_interferers])
        writer.writerow(['Number of Bursts', burst_params.num_bursts])
        writer.writerow(['Burst Symbol Rate', burst_params.burst_symbol_rate])
        writer.writerow(['BS Noise Figure (dB)', phy_params.bs_nf_db])
        writer.writerow(['Temperature (K)', phy_params.temp_k])
        writer.writerow(['Sampling Frequency (Hz)', phy_params.fs_hz])
        writer.writerow(['Traceback Depth', core_params.traceback_depth])
        # Power parameters (dBm mode)
        use_dbm_mode = getattr(config, 'use_dbm_mode', False)
        power_params = config.power_parameters
        writer.writerow(['Use dBm Mode', use_dbm_mode])

        if use_dbm_mode:
            # Calculate signal and noise power for dBm mode
            try:
                rx_signal_dbm = calculate_received_power_dbm(
                    power_params.bs_tx_power_dbm,
                    power_params.bs_antenna_gain_dbi,
                    power_params.ms_antenna_gain_dbi,
                    power_params.path_loss_db
                )
                rx_noise_dbm = calculate_thermal_noise_power_dbm(
                    power_params.channel_bandwidth_hz,
                    phy_params.temp_k,
                    phy_params.bs_nf_db
                )
                writer.writerow(['Received Signal Power (dBm)', f'{rx_signal_dbm:.2f}'])
                writer.writerow(['Received Noise Power (dBm)', f'{rx_noise_dbm:.2f}'])
            except Exception as e:
                print(f"Warning: Could not calculate power for CSV: {e}")

        writer.writerow(['BS TX Power (dBm)', power_params.bs_tx_power_dbm])
        writer.writerow(['BS Antenna Gain (dBi)', power_params.bs_antenna_gain_dbi])
        writer.writerow(['MS Antenna Gain (dBi)', power_params.ms_antenna_gain_dbi])
        writer.writerow(['Path Loss (dB)', power_params.path_loss_db])
        writer.writerow(['Channel Bandwidth (kHz)', power_params.channel_bandwidth_hz / 1000])
        writer.writerow([])  # Empty line for readability
        
        # Data section header
        if use_dbm_mode:
            # In dBm mode, show interference power instead of ratio
            writer.writerow(['# RESULTS DATA'])
            # Convert ratio values to interference power in dBm
            rx_signal_dbm = calculate_received_power_dbm(
                power_params.bs_tx_power_dbm,
                power_params.bs_antenna_gain_dbi,
                power_params.ms_antenna_gain_dbi,
                power_params.path_loss_db
            )
            rx_noise_dbm = calculate_thermal_noise_power_dbm(
                power_params.channel_bandwidth_hz,
                phy_params.temp_k,
                phy_params.bs_nf_db
            )

            # Convert each ratio to effective signal power and create pairs
            data_pairs = []
            for ratio, ber in zip(ratio_values, ber_values):
                effective_dbm = ratio_to_effective_signal_power_dbm(
                    ratio, rx_signal_dbm, rx_noise_dbm, mode_params.calculation_mode
                )
                data_pairs.append((effective_dbm, ber))

            # Sort by INCREASING effective signal power (low on left, high on right)
            data_pairs.sort(key=lambda x: x[0])
            writer.writerow(['Effective Signal Power (dBm)', 'BER'])
            for effective_dbm, ber in data_pairs:
                writer.writerow([f'{effective_dbm:.2f}', f'{ber:.6e}'])
        else:
            writer.writerow(['# RESULTS DATA'])
            writer.writerow([f'{mode_params.calculation_mode} (dB)', 'BER'])

            # Write data
            for ratio, ber in zip(ratio_values, ber_values):
                writer.writerow([f'{ratio:.2f}', f'{ber:.6e}'])

    print(f"Results saved to: {filepath}")
    return str(filepath)


def load_results_from_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load simulation results and metadata from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Tuple of (x_values, ber_values, metadata_dict)
        x_values can be ratio values (dB) or interference power (dBm) depending on mode
    """
    metadata = {}
    x_values = []
    ber_values = []
    reading_data = False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            
            if row[0] == '# RESULTS DATA':
                reading_data = True
                continue
            
            if row[0].startswith('#'):
                continue
            
            if reading_data:
                if row[0] in ['CI', 'SINR', 'Effective Signal'] or row[0].endswith('(dB)') or row[0].endswith('(dBm)'):
                    # This is the header, skip it
                    continue
                try:
                    x = float(row[0])
                    ber = float(row[1])
                    x_values.append(x)
                    ber_values.append(ber)
                except (ValueError, IndexError):
                    continue
            else:
                # Metadata section
                if len(row) >= 2:
                    key = row[0]
                    value = row[1]
                    metadata[key] = value
    
    return np.array(x_values), np.array(ber_values), metadata


def plot_results(ratio_values: np.ndarray = None, ber_values: np.ndarray = None, config: Any = None,
                 comparison_files: Optional[List[str]] = None):
    """
    Plot BER results. Can plot a single result or compare multiple result files.
    
    Args:
        ratio_values: Array of target ratio values (dB) for single plot
        ber_values: Array of BER values for single plot
        config: SystemConfig object for single plot
        comparison_files: List of CSV file paths to compare
    """
    plt.figure(figsize=(12, 7))
    colors = [
        'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown',
        'tab:pink','tab:gray','tab:olive','tab:cyan','black','orange','purple',
        '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b'
    ]

    markers = [
        'o','s','^','v','D','p','*','h','+','x','1','2','3','4','|','_' 
    ]
    linestyles = ['-', '--', '-.', ':']
    
    if comparison_files:
        # Comparison mode: load and plot multiple CSV files
        from itertools import product
        combos = list(product(colors, markers, linestyles))

        # Determine if all files use dBm mode
        metadata_list = []
        for filepath in comparison_files:
            try:
                _, _, metadata = load_results_from_csv(filepath)
                metadata_list.append(metadata)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

        # Check if using dBm mode
        use_dbm_mode_list = [m.get('Use dBm Mode', 'False') for m in metadata_list]
        use_dbm_mode = any(v.lower() in ['true', '1', 'yes'] for v in use_dbm_mode_list)
        calc_mode = metadata_list[0].get('Calculation Mode', 'CI') if metadata_list else 'CI'

        # Set xlabel based on mode
        if use_dbm_mode:
                xlabel = 'Effective Signal Power (dBm)'
        else:
            xlabel = f'{calc_mode} (dB)'

        for idx, filepath in enumerate(comparison_files):
            try:
                x_data, ber, metadata = load_results_from_csv(filepath)

                # Create label from metadata
                modulation = metadata.get('Modulation Type', 'Unknown')
                combining = metadata.get('Combining Mode', 'Unknown')
                est_method = metadata.get('Channel Estimation Method', 'Unknown')
                channel = metadata.get('Channel Model', 'Unknown')

                # Include channel model in the legend label to aid comparisons
                label = f"{modulation} ({combining}, {est_method}, {channel})"
                
                # Avoid log(0)
                ber_plot = np.where(ber == 0, 1e-6, ber)
                
                # Sort dBm data by INCREASING effective signal power for intuitive reading
                is_dbm_mode_file = metadata.get('Use dBm Mode', 'False').lower() in ['true', '1', 'yes']
                if is_dbm_mode_file:
                    sort_idx = np.argsort(x_data)
                    x_data = x_data[sort_idx]
                    ber_plot = ber_plot[sort_idx]
                    ber = ber[sort_idx] # Update raw ber for threshold finding

                # select style so color changes fastest, then marker, then linestyle
                color = colors[idx % len(colors)]
                marker = markers[(idx // len(colors)) % len(markers)]
                linestyle = linestyles[(idx // (len(colors) * len(markers))) % len(linestyles)]
                plt.semilogy(x_data, ber_plot, marker=marker, linestyle=linestyle,
                           linewidth=2, markersize=6, label=label, color=color, alpha=0.85)
                
                # --- Добавление вертикальной линии при использовании dBm ---
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        if use_dbm_mode:
            target_ber = 0.06
            # idx_6_percent = np.argmin(np.abs(ber - target_ber))
            plt.axhline(y=target_ber, color="black", linestyle='--', linewidth=1, alpha=0.6)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('BER', fontsize=12)
        plt.title('BER Comparison - Multiple Configurations', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.ylim([1e-5, 1])
        
    else:
        # Single plot mode
        if ratio_values is None or ber_values is None or config is None:
            print("Error: For single plot, provide ratio_values, ber_values, and config")
            return
        
        # Check if dBm mode is enabled
        use_dbm_mode = getattr(config, 'use_dbm_mode', False)

        if use_dbm_mode:
            # Convert ratio values to effective signal power in dBm for plotting
            power_params = config.power_parameters
            phy_params = config.physical_layer_parameters
            mode_params = config.mode_selection

            rx_signal_dbm = calculate_received_power_dbm(
                power_params.bs_tx_power_dbm,
                power_params.bs_antenna_gain_dbi,
                power_params.ms_antenna_gain_dbi,
                power_params.path_loss_db
            )
            rx_noise_dbm = calculate_thermal_noise_power_dbm(
                power_params.channel_bandwidth_hz,
                phy_params.temp_k,
                phy_params.bs_nf_db
            )

            # Convert each ratio to effective signal power
            x_values = np.array([
                ratio_to_effective_signal_power_dbm(ratio, rx_signal_dbm, rx_noise_dbm, mode_params.calculation_mode)
                for ratio in ratio_values
            ])

            xlabel = 'Effective Signal Power (dBm)'

            # Sort by INCREASING effective signal power (low on left, high on right)
            sort_idx = np.argsort(x_values)
            x_values = x_values[sort_idx]
            ber_values_sorted = ber_values[sort_idx]
            ber_plot = np.where(ber_values_sorted == 0, 1e-6, ber_values_sorted)
            ber_for_line = ber_values_sorted
        else:
            # Traditional mode: use ratio values (dB)
            x_values = ratio_values
            ber_plot = np.where(ber_values == 0, 1e-6, ber_values)
            ber_for_line = ber_values
            xlabel = f'{config.mode_selection.calculation_mode} (dB)'

        plt.semilogy(x_values, ber_plot, 'bo-', linewidth=2, markersize=6, alpha=0.8)

        # --- Добавление вертикальной линии при использовании dBm ---
        if use_dbm_mode:
            target_ber = 0.06
            # idx_6_percent = np.argmin(np.abs(ber_for_line - target_ber))
            # x_at_6_percent = x_values[idx_6_percent]
            plt.axhline(y=target_ber, color='blue', linestyle='--', linewidth=1.5, label='6% BER Threshold')
            plt.legend(loc='best')

        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        title = f"BER vs {config.mode_selection.calculation_mode} for {config.core_simulation_parameters.modulation_type} (channel: {config.core_simulation_parameters.channel_model})\n"
        title += (
            f"(Estimation: {config.mode_selection.channel_estimation_method}, "
            f"Combining: {config.mode_selection.combining_mode}, "
            f"Interferers: {config.num_interferers})"
        )
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('BER', fontsize=12)
        plt.ylim([1e-5, 1])
    
    plt.tight_layout()
    plt.show()
