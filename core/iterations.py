from core.irc_combining import irc_diversity_combining
from transceiver.burst import create_burst
from transceiver.generate_data import generate_data_bits
from transceiver.interference import interference_generation

from receiver.noise import add_thermal_noise
from receiver.scaling_and_combing import scaling_and_combing
from receiver.channel_estimation import estimate_channel_ls
from receiver.viterbi import mlse_viterbi_decode
from receiver.DeMUX import extract_data_segments

from scipy.signal import convolve
import numpy as np

def single_burst_iteration(args):
    (
        target_ratio_db, num_interferers, h11, h12, h21, h22, L, modem,
        training_sequence, bs_nf_db, temp_k, fs_hz, calculation_mode,
        channel_estimation_method, training_sequence_len, traceback_depth,
        num_data_bits_per_burst, tail_bits, guard_period, config, channel_model,
        enable_irc, irc_regularization
    ) = args
    
    # TODO: uncomment second string for repeatability
    # np.random.seed(111)
    # 1. TRANSMITTER SIDE
    data_bits = generate_data_bits(num_data_bits_per_burst)
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
        L, channel_idx, modem, len(tx_burst)
    )

    # 4. SCALING AND COMBINING
    rx_ant1, rx_ant2 = scaling_and_combing(
        s1_rx_ant1, s1_rx_ant2, target_ratio_db,calculation_mode, bs_nf_db,
        fs_hz, temp_k,total_interf_rx_ant1, total_interf_rx_ant2
    )

    # 5. RECEIVER: Add noise
    rx_ant1_noisy = add_thermal_noise(rx_ant1, bs_nf_db, fs_hz, temp_k)
    rx_ant2_noisy = add_thermal_noise(rx_ant2, bs_nf_db, fs_hz, temp_k)

    # 6. RECEIVER: Channel Estimation
    if channel_estimation_method == 'true':
        h_est_ant1, h_est_ant2 = h_true_ant1, h_true_ant2
    else:
        ts_start_idx = len(original_data_symbols) // 2 + len(tail_symbols)
        ts_end_idx = ts_start_idx + len(training_sequence)
        h_est_ant1 = estimate_channel_ls(
            rx_ant1_noisy[ts_start_idx: ts_end_idx + L - 1], training_sequence, L
        )
        h_est_ant2 = estimate_channel_ls(
            rx_ant2_noisy[ts_start_idx: ts_end_idx + L - 1], training_sequence, L
        )

    # 7. RECEIVER: Equalization and Decoding
    if enable_irc:
        # IRC MODE
        rx_combined, h_est_avg = irc_diversity_combining(
            rx_ant1_noisy, rx_ant2_noisy,
            h_est_ant1, h_est_ant2,
            training_sequence,
            enable_irc=True,
            regularization=irc_regularization
        )
    else:
        # MRC MODE
        rx_combined = (rx_ant1_noisy + rx_ant2_noisy) / 2
        h_est_avg = (h_est_ant1 + h_est_ant2) / 2

    mlse_input = rx_combined[:len(tx_burst) + L - 1]
    decoded_indices = mlse_viterbi_decode(mlse_input, h_est_avg, modem.constellation, traceback_depth)

    decoded_data_indices = extract_data_segments(original_data_symbols, decoded_indices, tail_symbols, training_sequence_len, modem, data_bits)
    decoded_bits = modem.demodulate_indices(decoded_data_indices)

    # 8. BER CALCULATION
    errors = np.sum(data_bits != decoded_bits[:len(data_bits)])
    bits = len(data_bits)
    return errors, bits