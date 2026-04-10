from core.irc_combining import irc_corrected_process
from core.st_irc_combining import st_irc_process
from core.ar_whitening import ar_prewhitening_process
from transceiver.burst import create_burst
from transceiver.generate_data import generate_data_bits
from transceiver.interference import interference_generation

from receiver.noise import add_thermal_noise
from receiver.scaling_and_combing import scaling_combining_and_noise, scaling_combining_and_noise_dbm
from receiver.channel_estimation import estimate_channel_ls, estimate_channel_lmmse, calculate_mse
from receiver.viterbi import mlse_viterbi_decode
from receiver.DeMUX import extract_data_segments

import csv
import multiprocessing

from scipy.signal import convolve
import numpy as np
from core.saic_whitening import single_antenna_processing
from core.temporal_whitening import single_antenna_temporal_whitening

import scipy.io as spio
import os

def estimate_antenna_correlation(h1, h2):
    h1_norm = h1 / (np.linalg.norm(h1) + 1e-10)
    h2_norm = h2 / (np.linalg.norm(h2) + 1e-10)
    rho = np.abs(np.vdot(h1_norm, h2_norm))
    return rho

def single_burst_iteration(args):
    (
        seed, snr, ci, target_ratio_db, num_interferers, h11, h12, h21, h22, L, modem,
        training_sequence, bs_nf_db, temp_k, fs_hz, calculation_mode,
        channel_estimation_method, training_sequence_len, traceback_depth,
        num_data_bits_per_burst, tail_bits, guard_period, config, channel_model,
        combining_mode, irc_regularization,
        apply_saic_preprocessing, saic_method, saic_regularization, saic_thermal_noise_variance,
        apply_temporal_whitening, temporal_method, temporal_regularization, temporal_thermal_noise_variance, temporal_full_burst,
        bs_tx_power_dbm, bs_antenna_gain_dbi, ms_antenna_gain_dbi, path_loss_db, channel_bandwidth_hz
    ) = args

    # np.random.seed(seed)

    # 1. TRANSMITTER SIDE
    data_bits = generate_data_bits(num_data_bits_per_burst, seed)
    tx_burst, original_data_symbols, tail_symbols = create_burst(data_bits, modem, training_sequence, tail_bits, guard_period)

     # 2. CHANNEL PROPAGATION
    channel_idx = np.random.randint(0, h11.shape[1])
    h_true_ant1 = h11[:L, channel_idx]
    h_true_ant2 = h12[:L, channel_idx]
    s1_rx_ant1 = convolve(tx_burst, h_true_ant1, 'full')
    s1_rx_ant2 = convolve(tx_burst, h_true_ant2, 'full')

    # 3. INTERFERENCE GENERATION
    total_interf_rx_ant1, total_interf_rx_ant2 = interference_generation(
        s1_rx_ant1, num_interferers, h21, h22,
        L, channel_idx, modem, len(tx_burst), seed
    )

    if getattr(config, "DEBUG_INTERFERENCE", True):
        debug_mat_file = "debug_interference_check.mat"
        if not os.path.exists(debug_mat_file):
            try:
                mat_data = {
                    'signal_ant1': s1_rx_ant1,
                    'signal_ant2': s1_rx_ant2,
                    'interf_ant1': total_interf_rx_ant1,
                    'interf_ant2': total_interf_rx_ant2,
                    'tx_burst': tx_burst,
                    'fs': fs_hz,
                    'mod_type': modem.type
                }
                spio.savemat(debug_mat_file, mat_data)
                print(f"\n[DEBUG] Signals exported to {debug_mat_file}\n")
            except Exception as e:
                print(f"Failed to export mat file: {e}")

    # 4. SCALING AND COMBINING
    # Check if dBm mode is enabled (use physical power calculations)
    use_dbm_mode = getattr(config, "use_dbm_mode", False)
    
    if use_dbm_mode:
        # Use dBm-based power calculations
        rx_ant1, rx_ant2, power_info = scaling_combining_and_noise_dbm(
            s1_rx_ant1, s1_rx_ant2,
            total_interf_rx_ant1, total_interf_rx_ant2,
            target_ratio_db,
            calculation_mode,
            tx_power_dbm=bs_tx_power_dbm,
            antenna_gain_dbi=bs_antenna_gain_dbi,
            path_loss_db=path_loss_db,
            noise_figure_db=bs_nf_db,
            bandwidth_hz=channel_bandwidth_hz,
            temperature_k=temp_k,
            constant_snr_db=snr,
            constant_ci_db=ci
        )
        rx_ant1_noisy = rx_ant1
        rx_ant2_noisy = rx_ant2
    else:
        # Use original relative dB-based calculations
        rx_ant1, rx_ant2 = scaling_combining_and_noise(
            s1_rx_ant1, s1_rx_ant2, total_interf_rx_ant1, total_interf_rx_ant2,
            target_ratio_db, calculation_mode, constant_snr_db=snr, constant_ci_db=ci
        )
        # 5. RECEIVER: Add noise
        rx_ant1_noisy = add_thermal_noise(rx_ant1, bs_nf_db, fs_hz, temp_k, seed)
        rx_ant2_noisy = add_thermal_noise(rx_ant2, bs_nf_db, fs_hz, temp_k, seed)


    # rho_antennas = estimate_antenna_correlation(h_true_ant1, h_true_ant2)
    # print(f"[Debug] Antenna correlation: {rho_antennas:.3f}")

    # 6. RECEIVER: Channel Estimation
    ts_start_idx = len(original_data_symbols) // 2 + len(tail_symbols)
    ts_end_idx = ts_start_idx + len(training_sequence)
    
    r_ant1_ts = rx_ant1_noisy[ts_start_idx: ts_end_idx + L - 1]
    
    mse_ls = 0.0
    mse_lmmse = 0.0
    
    save_mse = config.results_output.save_mse_debug    
    
    if save_mse:
        h_lmmse_debug = estimate_channel_lmmse(
            r_ant1_ts, 
            training_sequence, 
            L, 
            snr_db=target_ratio_db 
        )

        h_ls_debug, _ = estimate_channel_ls(r_ant1_ts, training_sequence, L)
        mse_ls = calculate_mse(h_true_ant1, h_ls_debug)
        mse_lmmse = calculate_mse(h_true_ant1, h_lmmse_debug)

    if channel_estimation_method == 'lmmse':
        h_est_ant1 = estimate_channel_lmmse(r_ant1_ts, training_sequence, L, snr_db=15.0)
        # Ant 2
        r_ant2_ts = rx_ant2_noisy[ts_start_idx: ts_end_idx + L - 1]
        h_est_ant2 = estimate_channel_lmmse(r_ant2_ts, training_sequence, L, snr_db=15.0)
    elif channel_estimation_method == 'ls':
        h_est_ant1, _ = estimate_channel_ls(r_ant1_ts, training_sequence, L)
        # Ant 2
        r_ant2_ts = rx_ant2_noisy[ts_start_idx: ts_end_idx + L - 1]
        h_est_ant2, _ = estimate_channel_ls(r_ant2_ts, training_sequence, L)
    else: # true
        h_est_ant1 = h_true_ant1
        h_est_ant2 = h_true_ant2

    # 7. RECEIVER: Equalization and Decoding
    rx_ant1_proc = rx_ant1_noisy
    h_est_ant1_proc = h_est_ant1

    if apply_saic_preprocessing:
        rx_ant1_proc, h_est_ant1_proc = single_antenna_processing(
            rx_ant1_proc,
            h_est_ant1_proc,
            training_sequence,
            enable_saic=True,
            method=saic_method,
            regularization=saic_regularization,
            thermal_noise_variance=saic_thermal_noise_variance
        )

    # Temporal whitening can be applied after SAIC (if both enabled) or standalone
    if apply_temporal_whitening:
        rx_ant1_proc, h_est_ant1_proc = single_antenna_temporal_whitening(
            rx_ant1_proc,
            h_est_ant1_proc,
            training_sequence,
            enable_temporal_whitening=True,
            method=temporal_method,
            regularization=temporal_regularization,
            thermal_noise_variance=temporal_thermal_noise_variance,
            full_burst=temporal_full_burst
        )

    # Antenna 2 processing
    rx_ant2_proc = rx_ant2_noisy
    h_est_ant2_proc = h_est_ant2

    if apply_saic_preprocessing:
        rx_ant2_proc, h_est_ant2_proc = single_antenna_processing(
            rx_ant2_proc,
            h_est_ant2_proc,
            training_sequence,
            enable_saic=True,
            method=saic_method,
            regularization=saic_regularization,
            thermal_noise_variance=saic_thermal_noise_variance
        )

    if apply_temporal_whitening:
        rx_ant2_proc, h_est_ant2_proc = single_antenna_temporal_whitening(
            rx_ant2_proc,
            h_est_ant2_proc,
            training_sequence,
            enable_temporal_whitening=True,
            method=temporal_method,
            regularization=temporal_regularization,
            thermal_noise_variance=temporal_thermal_noise_variance,
            full_burst=temporal_full_burst
        )

    # replace noisy signals and estimated channels with processed versions for combining
    rx_ant1_for_comb = rx_ant1_proc
    rx_ant2_for_comb = rx_ant2_proc
    h_est_ant1_for_comb = h_est_ant1_proc
    h_est_ant2_for_comb = h_est_ant2_proc
    if combining_mode == "IRC":
        rx_combined, h_est_avg = irc_corrected_process(
        [rx_ant1_for_comb, rx_ant2_for_comb],
        [h_est_ant1_for_comb, h_est_ant2_for_comb],
        training_sequence,
        shrinkageMethod='oas',
        shrinkage=0.1,
        loading_factor=0.28,
        )
    elif combining_mode == "ST-IRC":
        # Используем 1 временной тап (M=1), итого 4 виртуальные антенны
        method = getattr(config.mode_selection, 'st_irc_method', 'ar-prewhitening')
        if method == 'ar-prewhitening':
            rx_combined, h_est_avg = ar_prewhitening_process(
                [rx_ant1_for_comb, rx_ant2_for_comb],
                [h_est_ant1_for_comb, h_est_ant2_for_comb],
                training_sequence,
                M_taps=1,
                loading_factor=0.1 
            )
        elif method == 'direct':
            rx_combined, h_est_avg = st_irc_process(
                [rx_ant1_for_comb, rx_ant2_for_comb],
                [h_est_ant1_for_comb, h_est_ant2_for_comb],
                training_sequence,
                M_taps=1,
                shrinkageMethod='oas',
                loading_factor=0.28  # Регуляризация важна, т.к. матрица 4x4
            )
    elif combining_mode == "MRC":
        rx_combined, h_est_avg = irc_corrected_process(
        [rx_ant1_for_comb, rx_ant2_for_comb],
        [h_est_ant1_for_comb, h_est_ant2_for_comb],
        training_sequence,
        shrinkage=0.1,
        loading_factor=1.0
    )
        # g1 = h_est_ant1_for_comb[::-1].conj()
        # g2 = h_est_ant2_for_comb[::-1].conj()
        # mf1 = np.convolve(rx_ant1_for_comb, g1, mode='same')
        # mf2 = np.convolve(rx_ant2_for_comb, g2, mode='same')
        # rx_combined = mf1 + mf2
        # h_est_avg = h_est_ant1_for_comb + h_est_ant2_for_comb
    elif combining_mode == "EGC":
        # ERC MODE
        rx_combined = (rx_ant1_for_comb + rx_ant2_for_comb) / 2
        h_est_avg = (h_est_ant1_for_comb + h_est_ant2_for_comb) / 2
    elif combining_mode == "SINGLE":
        rx_combined = rx_ant2_for_comb
        h_est_avg = h_est_ant2_for_comb
    elif combining_mode == "SAIC":

        proc_ant2, eff_ch2 = single_antenna_processing(
            rx_ant2_noisy ,
            h_est_ant2,
            training_sequence,
            enable_saic=True,
            method='bias_removal',
            regularization=irc_regularization
        )

        rx_combined = proc_ant2
        h_est_avg = eff_ch2
    else:
        raise ValueError(f"Unknown combining mode: {combining_mode}")

    mlse_input = rx_combined[:len(tx_burst) + L - 1]
    decoded_indices = mlse_viterbi_decode(mlse_input, h_est_avg, modem.constellation, traceback_depth)

    decoded_data_indices = extract_data_segments(original_data_symbols, decoded_indices, tail_symbols, training_sequence_len, modem, data_bits)
    decoded_bits = modem.demodulate_indices(decoded_data_indices)

    # 8. BER CALCULATION
    errors = np.sum(data_bits != decoded_bits[:len(data_bits)])
    bits = len(data_bits)
    return errors, bits, mse_ls, mse_lmmse