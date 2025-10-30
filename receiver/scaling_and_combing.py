import numpy as np
def scaling_and_combing(s1_rx_ant1, s1_rx_ant2, target_ratio_db, calculation_mode, bs_nf_db, fs_hz, temp_k, total_interf_rx_ant1, total_interf_rx_ant2):
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
    return rx_ant1, rx_ant2