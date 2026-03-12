import numpy as np

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