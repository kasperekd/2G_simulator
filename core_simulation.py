from config import loader, validator, extract_parameters
from signal_generation import modulation, bit_generator
from channel import qudriga_importer, awgn
from analysis import plotter, metrics
from receiver import channel_estimator, irc_combiner, viterbi

import concurrent.futures

import numpy as np
from scipy.signal import convolve
from tqdm import tqdm
# import matplotlib.pyplot as plt

def get_receiver_constellation(mod_type: str) -> np.ndarray:
    """Returns the complex constellation for the receiver's Viterbi logic."""
    temp_modem = modulation.get_constellation(mod_type)
    return temp_modem

def run_single_iteration(config, snr_db, ci_db, show_plots=False):
    """
    Runs a single simulation for a given SNR and C/I, returning the resulting BER.
    """
    # --- Setup ---
    # Unpack config
    count_bit, samp_rate, _, mod_type, _, _, rx_config = extract_parameters.extract_config_parameters(config)
    L = rx_config.channel_memory_L
    N_ts = rx_config.training_sequence_len

    receiver_constellation = get_receiver_constellation(mod_type)
    bits_per_symbol = 1 if mod_type in ["BPSK", "GMSK"] else int(np.log2(len(receiver_constellation)))

    # --- Signal Generation ---
    # Нужны повторяемые данные
    # user1_data_bits = np.random.randint(0, 2, count_bit)
    user1_data_bits = bit_generator.generate_bit(count_bit, number_seed= 10)

    ts_symbols = receiver_constellation[np.random.randint(0, len(receiver_constellation), N_ts)]
    ts_symbols = np.array(rx_config.training_sequence, dtype=np.complex128)
    s1_data_modulated, s1_data_symbol_indices = modulation.modulate_bits(mod_type, user1_data_bits, samp_rate)
    s1_modulated = np.concatenate([ts_symbols, s1_data_modulated])
    
    # Interferer generation with C/I scaling
    interferer_total_bits = len(s1_modulated) * bits_per_symbol
     # Нужны повторяемые данные
    # interferer_bits = np.random.randint(0, 2, interferer_total_bits)
    interferer_bits = bit_generator.generate_bit(interferer_total_bits, number_seed = 125)

    s2_modulated, _ = modulation.modulate_bits(mod_type, interferer_bits, samp_rate)
    s2_modulated = s2_modulated[:len(s1_modulated)]
    interference_power = 1 / (10**(ci_db / 10))
    s2_modulated *= np.sqrt(interference_power)

    # --- Channel and Reception ---
    channel_path = "./channel/Ht2_0204_11.mat"
    h11, h12, h21, h22 = qudriga_importer.load_channel_matrix(channel_path)
    channel_idx = np.random.randint(0, h11.shape[1])
    h_true_user1 = np.array([h11[:L, channel_idx], h12[:L, channel_idx]])
    h_true_user2 = np.array([h21[:L, channel_idx], h22[:L, channel_idx]])

    s1_conv_ant1 = convolve(s1_modulated, h_true_user1[0,:], mode='valid')
    s1_conv_ant2 = convolve(s1_modulated, h_true_user1[1,:], mode='valid')
    s2_conv_ant1 = convolve(s2_modulated, h_true_user2[0,:], mode='valid')
    s2_conv_ant2 = convolve(s2_modulated, h_true_user2[1,:], mode='valid')
    rx_ant1_noisy = awgn.add_awgn(s1_conv_ant1 + s2_conv_ant1, snr_db, mod_type)
    rx_ant2_noisy = awgn.add_awgn(s1_conv_ant2 + s2_conv_ant2, snr_db, mod_type)
    rx_matrix = np.vstack([rx_ant1_noisy, rx_ant2_noisy])
    
    # --- Receiver Processing ---
    ts_rx_len = N_ts + L - 1
    rx_ts_part = rx_matrix[:, :ts_rx_len]
    # print(f"rx ts part:\n {rx_ts_part}")
    '''Подаём нашу импульсную характеристику напрямую на оценку канала, чтобы проверить её работу'''
    # h1_est = channel_estimator.estimate_channel_ls(convolve(h_true_user1[0, :], ts_symbols, mode='full'), ts_symbols, L)
    # h2_est = channel_estimator.estimate_channel_ls(convolve(h_true_user2[0, :], ts_symbols, mode='full'), ts_symbols, L)

    '''Вычисление оценки канала'''
    h1_est = channel_estimator.estimate_channel_ls(rx_ts_part[0, :], ts_symbols, L)
    h2_est = channel_estimator.estimate_channel_ls(rx_ts_part[1, :], ts_symbols, L)
    
    h_est_matrix = np.vstack([h1_est, h2_est])
    Q_est = np.eye(2)
    if rx_config.receiver_type == "IRC-MLSE":
        Q_est = irc_combiner.estimate_noise_covariance(rx_ts_part, h_est_matrix, ts_symbols)
    
    y_hat, S_n = irc_combiner.calculate_irc_parameters(rx_matrix, h_est_matrix, Q_est)
    data_start_idx = N_ts
    num_data_symbols_to_decode = len(s1_data_symbol_indices)
    y_hat_data = y_hat[data_start_idx : data_start_idx + num_data_symbols_to_decode]
    decoded_symbol_indices = viterbi.mlse_viterbi_decode(y_hat_data, S_n, L, receiver_constellation)
    
    # --- BER Calculation ---
    decoded_bits = modulation.demodulate_symbols(mod_type, decoded_symbol_indices, samp_rate)
    decoded_bits = decoded_bits[:len(user1_data_bits)]
    ber = metrics.calculate_ber(user1_data_bits, decoded_bits)
    
    # --- Optional Plotting for Single Run ---
    if show_plots:
        print("\n--- DISPLAYING DIAGNOSTIC PLOTS ---")
        plotter.visualize_mimo_reception(
            rx1=rx_matrix[0, :500], rx2=rx_matrix[1, :500],
            h11=h_true_user1[0, :], h12=h_true_user1[1, :],
            h21=h_true_user2[0, :], h22=h_true_user2[1, :]
        )
        plotter.plot_channel_estimation_results(h_true_user1[0, :], h_est_matrix[0, :], antenna_id=1)
        plotter.plot_channel_estimation_results(h_true_user1[1, :], h_est_matrix[1, :], antenna_id=2)
        plotter.plot_equalizer_constellations(y_hat_data, S_n, decoded_symbol_indices, receiver_constellation)
        
    return ber

def monte_carlo_ber_for_ci(ci_db, config, snr_db, num_iters):
    bers = [
        run_single_iteration(config, snr_db, ci_db, show_plots=False)
        for _ in range(num_iters)
    ]
    return np.mean(bers)

def main():
    """Main entry point that selects the simulation mode based on config."""
    config_path = "./config/settings.json"
    config = validator.validate_config(loader.ConfigLoader.load(config_path))
    
    mode_config = config.simulation_mode
    snr_db = config.noise_configuration.signal_to_noise_ratio_db

    if mode_config.mode == "SingleRun":
        print(f"--- Running in Single-Run Mode ---")
        print(f"C/I = {mode_config.single_run_ci_db} dB, SNR = {snr_db} dB")
        
        for receiver_type in ["MRC-MLSE", "IRC-MLSE"]:
            print(f"\n--- Testing Receiver: {receiver_type} ---")
            config.receiver_configuration.receiver_type = receiver_type
            # config.generation_signal_configuration.seed_configuration.number_seed = 42
            ber = run_single_iteration(config, snr_db, mode_config.single_run_ci_db, show_plots=True)
            print(f"\n--- Result for {receiver_type} ---")
            print(f"Bit Error Rate (BER): {ber:.6f}\n")
            
    if mode_config.mode == "MonteCarlo":
        print(f"--- Running in Monte Carlo Mode ---")
        mc = mode_config.monte_carlo
        CI_DB_RANGE = np.arange(mc.ci_db_min, mc.ci_db_max + 1, mc.ci_db_step)
        NUM_ITER = mc.num_iterations

        RECEIVERS = ["MRC-MLSE", "IRC-MLSE"]
        ber_results = {}

        for rx_type in RECEIVERS:
            print(f"\n--- Testing Receiver: {rx_type} ---")
            config.receiver_configuration.receiver_type = rx_type

            avg_ber = []
            for ci_db in tqdm(CI_DB_RANGE, desc=f"C/I Sweep for {rx_type}"):
                configs    = [config] * NUM_ITER
                snr_list   = [snr_db] * NUM_ITER
                ci_list    = [ci_db]  * NUM_ITER
                plots_flag = [False]  * NUM_ITER

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = list(executor.map(
                        run_single_iteration,
                        configs,
                        snr_list,
                        ci_list,
                        plots_flag
                    ))

                avg_ber.append(np.mean(results))

            ber_results[rx_type] = np.array(avg_ber)

        print("\n--- Plotting Final Results ---")
        plotter.plot_ber_vs_ci(CI_DB_RANGE, ber_results, RECEIVERS)

if __name__ == "__main__":
    main()