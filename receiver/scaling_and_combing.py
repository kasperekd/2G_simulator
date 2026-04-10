import numpy as np
from receiver.power_calc import (
    calculate_received_power_dbm,
    calculate_thermal_noise_power_dbm,
    dbm_to_watts
)

def scaling_combining_and_noise(
    s1_rx_ant1, s1_rx_ant2, 
    interf_rx_ant1, interf_rx_ant2, 
    target_ratio_db, 
    calculation_mode, 
    constant_snr_db=15.0, 
    constant_ci_db=20.0   
):
    
    signal_power = np.mean(np.abs(s1_rx_ant1)**2 + np.abs(s1_rx_ant2)**2)
    if signal_power < 1e-20: signal_power = 1e-20

    current_interf_power = np.mean(np.abs(interf_rx_ant1)**2 + np.abs(interf_rx_ant2)**2)
    if current_interf_power < 1e-20: current_interf_power = 1e-20

    sig_ant1, sig_ant2 = s1_rx_ant1, s1_rx_ant2
    inf_ant1, inf_ant2 = interf_rx_ant1, interf_rx_ant2

    target_ratio_linear = 10**(target_ratio_db / 10)

    constant_snr_linear = 10**(constant_snr_db / 10)
    constant_noise_power = signal_power / constant_snr_linear

    if calculation_mode == 'CI':
        required_interf_power = signal_power / target_ratio_linear
        scaling_factor_inf = np.sqrt(required_interf_power / current_interf_power)
        inf_ant1 = inf_ant1 * scaling_factor_inf
        inf_ant2 = inf_ant2 * scaling_factor_inf
        
        final_noise_power = constant_noise_power

    elif calculation_mode == 'SINR':
        required_interf_power = (signal_power / target_ratio_linear) - constant_noise_power
        if required_interf_power < 0:
            required_interf_power = 1e-20
            
        scaling_factor_inf = np.sqrt(required_interf_power / current_interf_power)
        inf_ant1 = inf_ant1 * scaling_factor_inf
        inf_ant2 = inf_ant2 * scaling_factor_inf
        
        final_noise_power = constant_noise_power

    elif calculation_mode == 'SNR':
        constant_ci_linear = 10**(constant_ci_db / 10)
        required_interf_power = signal_power / constant_ci_linear
        scaling_factor_inf = np.sqrt(required_interf_power / current_interf_power)
        inf_ant1 = inf_ant1 * scaling_factor_inf
        inf_ant2 = inf_ant2 * scaling_factor_inf

        final_noise_power = signal_power / target_ratio_linear

    else:
        raise ValueError("Must be 'CI', 'SNR' or 'SINR'")

    noise_std = np.sqrt(final_noise_power / 4)

    noise_ant1 = noise_std * (np.random.randn(*sig_ant1.shape) + 1j * np.random.randn(*sig_ant1.shape))
    noise_ant2 = noise_std * (np.random.randn(*sig_ant2.shape) + 1j * np.random.randn(*sig_ant2.shape))

    rx_ant1_noisy = sig_ant1 + inf_ant1 + noise_ant1
    rx_ant2_noisy = sig_ant2 + inf_ant2 + noise_ant2

    return rx_ant1_noisy, rx_ant2_noisy


def scaling_combining_and_noise_dbm(
    s1_rx_ant1, s1_rx_ant2,
    interf_rx_ant1, interf_rx_ant2,
    target_ratio_db,
    calculation_mode,
    # dBm parameters
    tx_power_dbm,
    antenna_gain_dbi,
    path_loss_db,
    noise_figure_db,
    bandwidth_hz,
    temperature_k=300.0,
    constant_snr_db=15.0,
    constant_ci_db=20.0
):
    """
    Alternative implementation using absolute power values in dBm.

    This function:
    1. Calculates received power using link budget (tx power, antenna gains, path loss)
    2. Calculates thermal noise power based on physical parameters
    3. Scales signal/interference/noise to achieve target ratio

    Args:
        s1_rx_ant1, s1_rx_ant2: Desired signal at antennas (complex baseband)
        interf_rx_ant1, interf_rx_ant2: Interference signal at antennas
        target_ratio_db: Target CI/SINR/SNR ratio in dB
        calculation_mode: 'CI', 'SINR', or 'SNR'
        tx_power_dbm: Transmit power in dBm
        antenna_gain_dbi: Antenna gain in dBi
        path_loss_db: Path loss in dB
        noise_figure_db: Noise figure in dB
        bandwidth_hz: Channel bandwidth in Hz
        temperature_k: Temperature in Kelvin (default 290)
        constant_snr_db: Constant SNR for CI/SINR modes (dB)
        constant_ci_db: Constant CI for SNR mode (dB)

    Returns:
        rx_ant1_noisy, rx_ant2_noisy: Noisy receiver signals
        power_info_dict: Dictionary with power information in dBm
    """
    # Calculate received signal power using link budget
    rx_signal_power_dbm = calculate_received_power_dbm(
        tx_power_dbm,
        tx_antenna_gain_dbi=antenna_gain_dbi,
        path_loss_db=path_loss_db
    )

    # Calculate thermal noise power
    rx_noise_power_dbm = calculate_thermal_noise_power_dbm(
        bandwidth_hz,
        temperature_k=temperature_k,
        noise_figure_db=noise_figure_db
    )

    signal_power = np.mean(np.abs(s1_rx_ant1)**2 + np.abs(s1_rx_ant2)**2)
    if signal_power < 1e-20:
        signal_power = 1e-20

    current_interf_power = np.mean(np.abs(interf_rx_ant1)**2 + np.abs(interf_rx_ant2)**2)
    if current_interf_power < 1e-20:
        current_interf_power = 1e-20

    target_ratio_linear = 10**(target_ratio_db / 10)

    sig_ant1, sig_ant2 = s1_rx_ant1, s1_rx_ant2
    inf_ant1, inf_ant2 = interf_rx_ant1, interf_rx_ant2

    if calculation_mode == 'CI':
        # CI mode: target Carrier-to-Interference ratio
        required_interf_power = signal_power / target_ratio_linear
        scaling_factor_inf = np.sqrt(required_interf_power / current_interf_power)
        inf_ant1 = inf_ant1 * scaling_factor_inf
        inf_ant2 = inf_ant2 * scaling_factor_inf

        # Noise remains at physical level
        final_noise_power = signal_power * dbm_to_watts(rx_noise_power_dbm) / dbm_to_watts(rx_signal_power_dbm)

    elif calculation_mode == 'SINR':
        # SINR mode: target Signal-to-Interference-plus-Noise ratio
        # Use actual physical noise power from dBm calculation
        signal_power_watts = dbm_to_watts(rx_signal_power_dbm)
        noise_power_watts = dbm_to_watts(rx_noise_power_dbm)

        # Scale noise to match current signal power level
        noise_power_scaled = (noise_power_watts / signal_power_watts) * signal_power

        required_interf_power = (signal_power / target_ratio_linear) - noise_power_scaled
        if required_interf_power < 0:
            required_interf_power = 1e-20

        scaling_factor_inf = np.sqrt(required_interf_power / current_interf_power)
        inf_ant1 = inf_ant1 * scaling_factor_inf
        inf_ant2 = inf_ant2 * scaling_factor_inf

        final_noise_power = noise_power_scaled

    elif calculation_mode == 'SNR':
        # SNR mode: target Signal-to-Noise ratio
        noise_power_watts = dbm_to_watts(rx_noise_power_dbm)
        signal_power_watts = dbm_to_watts(rx_signal_power_dbm)

        # Scale noise to match current signal power level
        noise_power_scaled = (noise_power_watts / signal_power_watts) * signal_power

        required_noise_power = signal_power / target_ratio_linear
        scaling_factor_noise = np.sqrt(required_noise_power / noise_power_scaled)

        # Interference at constant level
        constant_ci_linear = 10**(constant_ci_db / 10)
        required_interf_power = signal_power / constant_ci_linear
        scaling_factor_inf = np.sqrt(required_interf_power / current_interf_power)
        inf_ant1 = inf_ant1 * scaling_factor_inf
        inf_ant2 = inf_ant2 * scaling_factor_inf

        final_noise_power = required_noise_power

    else:
        raise ValueError("calculation_mode must be 'CI', 'SNR' or 'SINR'")

    # Generate noise
    noise_std = np.sqrt(final_noise_power / 4)
    noise_ant1 = noise_std * (np.random.randn(*sig_ant1.shape) + 1j * np.random.randn(*sig_ant1.shape))
    noise_ant2 = noise_std * (np.random.randn(*sig_ant2.shape) + 1j * np.random.randn(*sig_ant2.shape))

    rx_ant1_noisy = sig_ant1 + inf_ant1 + noise_ant1
    rx_ant2_noisy = sig_ant2 + inf_ant2 + noise_ant2

    # Power information dictionary for logging
    power_info_dict = {
        'signal_power_dbm': rx_signal_power_dbm,
        'noise_power_dbm': rx_noise_power_dbm,
        'antenna_gain_dbi': antenna_gain_dbi,
        'path_loss_db': path_loss_db
    }

    return rx_ant1_noisy, rx_ant2_noisy, power_info_dict

