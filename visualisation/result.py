import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

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
        power_params = config.power_parameters
        writer.writerow(['Use dBm Mode', getattr(config, 'use_dbm_mode', False)])
        writer.writerow(['BS TX Power (dBm)', power_params.bs_tx_power_dbm])
        writer.writerow(['BS Antenna Gain (dBi)', power_params.bs_antenna_gain_dbi])
        writer.writerow(['MS Antenna Gain (dBi)', power_params.ms_antenna_gain_dbi])
        writer.writerow(['Path Loss (dB)', power_params.path_loss_db])
        writer.writerow(['Channel Bandwidth (kHz)', power_params.channel_bandwidth_hz / 1000])
        writer.writerow([])  # Empty line for readability
        
        # Data section header
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
        Tuple of (ratio_values, ber_values, metadata_dict)
    """
    metadata = {}
    ratio_values = []
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
                if row[0] in ['CI', 'SINR'] or row[0].endswith('(dB)'):
                    # This is the header, skip it
                    continue
                try:
                    ratio = float(row[0])
                    ber = float(row[1])
                    ratio_values.append(ratio)
                    ber_values.append(ber)
                except (ValueError, IndexError):
                    continue
            else:
                # Metadata section
                if len(row) >= 2:
                    key = row[0]
                    value = row[1]
                    metadata[key] = value
    
    return np.array(ratio_values), np.array(ber_values), metadata


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

        for idx, filepath in enumerate(comparison_files):
            try:
                ratio, ber, metadata = load_results_from_csv(filepath)
                
                # Create label from metadata
                modulation = metadata.get('Modulation Type', 'Unknown')
                combining = metadata.get('Combining Mode', 'Unknown')
                est_method = metadata.get('Channel Estimation Method', 'Unknown')
                channel = metadata.get('Channel Model', 'Unknown')

                # Include channel model in the legend label to aid comparisons
                label = f"{modulation} ({combining}, {est_method}, {channel})"
                
                # Avoid log(0)
                ber_plot = np.where(ber == 0, 1e-6, ber)
                
                # select style so color changes fastest, then marker, then linestyle
                color = colors[idx % len(colors)]
                marker = markers[(idx // len(colors)) % len(markers)]
                linestyle = linestyles[(idx // (len(colors) * len(markers))) % len(linestyles)]
                plt.semilogy(ratio, ber_plot, marker=marker, linestyle=linestyle,
                           linewidth=2, markersize=6, label=label, color=color, alpha=0.85)
                
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        plt.xlabel('CI (dB)', fontsize=12)
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
        
        ber_plot = np.where(ber_values == 0, 1e-6, ber_values)
        plt.semilogy(ratio_values, ber_plot, 'bo-', linewidth=2, markersize=6, alpha=0.8)
        
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        title = f"BER vs {config.mode_selection.calculation_mode} for {config.core_simulation_parameters.modulation_type} (channel: {config.core_simulation_parameters.channel_model})\n"
        title += (
            f"(Estimation: {config.mode_selection.channel_estimation_method}, "
            f"Combining: {config.mode_selection.combining_mode}, "
            f"Interferers: {config.num_interferers})"
        )
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel(f'{config.mode_selection.calculation_mode} (dB)', fontsize=12)
        plt.ylabel('BER', fontsize=12)
        plt.ylim([1e-5, 1])
    
    plt.tight_layout()
    plt.show()
