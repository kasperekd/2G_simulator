from config import validator, loader, extract_parameters
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import scipy.io as spio
from typing import Dict
import threading, time
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 1. MODULATION CLASS
# ==============================================================================

class Modulator:
    def __init__(self, modulation_type='QPSK'):
        self.type = modulation_type
        if self.type == 'BPSK':
            self.bits_per_symbol = 1
            self.constellation = np.array([-1, 1])
        elif self.type == 'QPSK':
            self.bits_per_symbol = 2
            self.constellation = np.array([
                1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j
            ]) / np.sqrt(2)
        elif self.type == '8PSK':
            self.bits_per_symbol = 3
            self.constellation = np.exp(1j * np.pi / 4 * np.arange(8))
        elif self.type == '16QAM':
            self.bits_per_symbol = 4
            self.constellation = np.array([
                -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
                -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
                 1 - 3j,  1 - 1j,  1 + 1j,  1 + 3j,
                 3 - 3j,  3 - 1j,  3 + 1j,  3 + 3j
            ]) / np.sqrt(10)
        # TODO: Add 32QAM and a proper GMSK implementation.
        else:
            raise ValueError(f"Unsupported modulation type: {self.type}")
        self.num_symbols = len(self.constellation)
        self._map_bits_to_idx = {
            tuple(map(int, np.binary_repr(i, width=self.bits_per_symbol))): i
            for i in range(self.num_symbols)
        }
        self._map_idx_to_bits = {v: k for k, v in self._map_bits_to_idx.items()}

    def modulate(self, bits):
        if len(bits) % self.bits_per_symbol != 0:
            bits = np.append(
                bits,
                np.zeros(
                    self.bits_per_symbol - (len(bits) % self.bits_per_symbol),
                    dtype=int
                )
            )
        symbols = [
            self.constellation[
                self._map_bits_to_idx[tuple(bits[i:i + self.bits_per_symbol])]
            ]
            for i in range(0, len(bits), self.bits_per_symbol)
        ]
        return np.array(symbols)

    def demodulate_indices(self, indices):
        bit_list = [
            bit for idx in indices for bit in self._map_idx_to_bits[idx]
        ]
        return np.array(bit_list)

# ==============================================================================
# 2. HELPER AND SIMULATION FUNCTIONS
# ==============================================================================


def load_quadriga_channel(file_path: str):
    # FIXME: This function and its usage should be replaced by a proper 3GPP channel model generator.
    mat_data = spio.loadmat(file_path)
    return mat_data['Ht11'], mat_data['Ht12'], mat_data['Ht21'], mat_data['Ht22']


def estimate_channel_ls(received_ts, known_ts, L):
    len_ts = len(known_ts)
    if len(received_ts) < len_ts:
        received_ts = np.pad(received_ts, (0, len_ts - len(received_ts)))
    S = np.zeros((len_ts, L), dtype=complex)
    for i in range(len_ts):
        for j in range(L):
            if i - j >= 0:
                S[i, j] = known_ts[i - j]
    r = received_ts[:len_ts]
    try:
        h_est = np.linalg.inv(S.conj().T @ S) @ S.conj().T @ r
    except np.linalg.LinAlgError:
        h_est = np.linalg.pinv(S) @ r
    return h_est


def mlse_viterbi_decode(
    received_symbols, channel_taps, constellation, traceback_depth=15
):
    channel_memory = len(channel_taps) - 1
    num_symbols_in_constellation = len(constellation)

    if channel_memory < 0:
        channel_memory = 0

    if channel_memory > 0:
        num_states = num_symbols_in_constellation ** channel_memory
        states = [
            tuple(
                reversed(
                    [
                        (i // (num_symbols_in_constellation ** j))
                        % num_symbols_in_constellation
                        for j in range(channel_memory)
                    ]
                )
            )
            for i in range(num_states)
        ]
    else:
        num_states = 1
        states = [()]

    path_metrics = np.full(num_states, np.inf)
    path_metrics[0] = 0
    path_history = np.zeros((len(received_symbols), num_states, 2), dtype=int)

    for t in range(len(received_symbols)):
        r = received_symbols[t]
        new_metrics = np.full(num_states, np.inf)

        for curr_state_idx, curr_state in enumerate(states):
            if path_metrics[curr_state_idx] == np.inf:
                continue

            for input_idx in range(num_symbols_in_constellation):
                # Calculate expected symbol based on current input and state (past symbols)
                expected = channel_taps[0] * constellation[input_idx]
                for i, sym_idx in enumerate(curr_state):
                    if i + 1 < len(channel_taps):
                        expected += channel_taps[i + 1] * constellation[sym_idx]

                branch_metric = np.abs(r - expected)**2

                # Determine next state
                next_state = (input_idx,) + curr_state[:-1] if channel_memory > 0 else ()
                next_state_idx = states.index(next_state)

                new_metric = path_metrics[curr_state_idx] + branch_metric
                if new_metric < new_metrics[next_state_idx]:
                    new_metrics[next_state_idx] = new_metric
                    path_history[t, next_state_idx] = [curr_state_idx, input_idx]

        path_metrics = new_metrics

    # Traceback
    decoded_indices = []
    # Start from the state with the minimum path metric at the end
    current_state_idx = np.argmin(path_metrics)

    for t in range(len(received_symbols) - 1, -1, -1):
        prev_state_idx, input_idx = path_history[t, current_state_idx]
        decoded_indices.append(input_idx)
        current_state_idx = prev_state_idx
        if t > 0 and len(decoded_indices) >= traceback_depth:
            # A simplified traceback approach for streaming data simulation
            # For a block-based simulation, full traceback is better
            pass

    return np.array(list(reversed(decoded_indices)))


def generate_data_bits(num_bits):
    # TODO: Replace with a proper data source block, possibly including channel coding.
    return np.random.randint(0, 2, num_bits)


def add_thermal_noise(signal, config):
    # Physical noise calculation
    k_boltzmann = 1.380649e-23
    nf_linear = 10**(config.physical_layer_parameters.bs_nf_db / 10)
    # TODO: Noise bandwidth should be channel bandwidth (e.g. 200e3 for GSM) not Fs
    noise_power = k_boltzmann * config.physical_layer_parameters.temp_k * config.physical_layer_parameters.fs_hz * nf_linear
    noise_std_dev = np.sqrt(noise_power / 2)
    noise = noise_std_dev * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise

# TODO: changing the parameters for the types of modulation
def create_burst(data_bits, modem, training_sequence, channel_memory):
    data_symbols = modem.modulate(data_bits)
    part1_len = len(data_symbols) // 2
    part1_syms = data_symbols[:part1_len]
    part2_syms = data_symbols[part1_len:]
    tail_symbols = np.zeros(channel_memory)
    burst = np.concatenate([
        tail_symbols, part1_syms, training_sequence, part2_syms, tail_symbols
    ])
    return burst, data_symbols

def single_iteration(param):
    (target_ratio_db, num_interferers, h11, h12, h21, h22, L, modem,
     training_sequence, num_bursts, bs_nf_db, temp_k, fs_hz, calculation_mode,
     channel_estimation_method, training_sequence_len, traceback_depth,
     config, num_data_symbols_per_burst) = param
    
    total_errors, total_bits = 0, 0
    for _ in range(num_bursts):
        # 1. TRANSMITTER SIDE
        data_bits = generate_data_bits(num_data_symbols_per_burst * modem.bits_per_symbol)
        tx_burst, original_data_symbols = create_burst(data_bits, modem, training_sequence, L)

        # 2. CHANNEL PROPAGATION
        channel_idx = np.random.randint(0, h11.shape[1])
        h_true_ant1 = h11[:L, channel_idx]
        h_true_ant2 = h12[:L, channel_idx]
        s1_rx_ant1 = convolve(tx_burst, h_true_ant1, 'full')
        s1_rx_ant2 = convolve(tx_burst, h_true_ant2, 'full')

        # 3. INTERFERENCE GENERATION
        max_len = len(s1_rx_ant1)
        total_interf_rx_ant1 = np.zeros(max_len, dtype=complex)
        total_interf_rx_ant2 = np.zeros(max_len, dtype=complex)
        for _ in range(num_interferers):
            interf_bits = generate_data_bits(len(tx_burst) * modem.bits_per_symbol)
            interf_syms = modem.modulate(interf_bits)
            h_interf_ant1 = h21[:L, channel_idx]
            h_interf_ant2 = h22[:L, channel_idx]
            interf_conv1 = convolve(interf_syms, h_interf_ant1, 'full')
            interf_conv2 = convolve(interf_syms, h_interf_ant2, 'full')
            total_interf_rx_ant1[:len(interf_conv1)] += interf_conv1
            total_interf_rx_ant2[:len(interf_conv2)] += interf_conv2

        # 4. SCALING AND COMBINING
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

        # 5. RECEIVER: Add noise
        rx_ant1_noisy = add_thermal_noise(rx_ant1, config)
        rx_ant2_noisy = add_thermal_noise(rx_ant2, config)

        # 6. RECEIVER: Channel Estimation
        if channel_estimation_method == 'true':
            h_est_ant1, h_est_ant2 = h_true_ant1, h_true_ant2
        else:  # 'ls'
            ts_start_idx = len(original_data_symbols) // 2 + L
            ts_end_idx = ts_start_idx + training_sequence_len
            h_est_ant1 = estimate_channel_ls(rx_ant1_noisy[ts_start_idx: ts_end_idx + L - 1], training_sequence, L)
            h_est_ant2 = estimate_channel_ls(rx_ant2_noisy[ts_start_idx: ts_end_idx + L - 1], training_sequence, L)

        # 7. RECEIVER: Equalization and Decoding
        rx_combined = (rx_ant1_noisy + rx_ant2_noisy) / 2
        h_est_avg = (h_est_ant1 + h_est_ant2) / 2

        mlse_input = rx_combined[:len(tx_burst) + L - 1]
        decoded_indices = mlse_viterbi_decode(mlse_input, h_est_avg, modem.constellation, traceback_depth)

        # 8. BER CALCULATION
        d1_len = len(original_data_symbols) // 2
        d2_len = len(original_data_symbols) - d1_len
        decoded_d1_indices = decoded_indices[L: L + d1_len]
        d2_start_idx = L + d1_len + training_sequence_len
        d2_end_idx = d2_start_idx + d2_len
        decoded_d2_indices = decoded_indices[d2_start_idx: d2_end_idx]
        decoded_data_indices = np.concatenate([decoded_d1_indices, decoded_d2_indices])
        decoded_bits = modem.demodulate_indices(decoded_data_indices)

        total_errors += np.sum(data_bits != decoded_bits[:len(data_bits)])
        total_bits += len(data_bits)

    ber = total_errors / total_bits if total_bits > 0 else 0.5
    return target_ratio_db, ber

def simulate(config):
    np.random.seed(111)
    start_time = time.perf_counter()
    num_interferers, channel_mat_file, channel_memory, range_db, modulation_type, calculation_mode, channel_estimation_method, num_bursts, channel_model, num_data_symbols_per_burst, training_sequence_len, traceback_depth, training_sequence, bs_nf_db, temp_k, fs_hz = extract_parameters.extract_config_parameters(config)
    h11, h12, h21, h22 = load_quadriga_channel(channel_mat_file)
    L = channel_memory
    modem = Modulator(modulation_type)
    # FIXME: random TS for BPSK. Now we have GMSK TS for all modulation
    ts_indices = np.random.randint(
            0, len(modem.constellation), training_sequence_len
        )
    training_sequence = modem.constellation[ts_indices]
    target_ratio_range_db = np.arange(range_db[0], range_db[1], range_db[2])

    ratio_values, ber_values = np.zeros(len(target_ratio_range_db)), np.zeros(len(target_ratio_range_db))

    print("Starting simulation:")
    print(
        f"Modulation: {modulation_type}, "
        f"Mode: {calculation_mode}, "
        f"Estimation: {channel_estimation_method}"
    )    
    params_list = []
    for target_ratio_db in target_ratio_range_db:
        params_list.append((
            target_ratio_db, num_interferers, h11, h12, h21, h22, L, modem,
            training_sequence, num_bursts, bs_nf_db, temp_k, fs_hz, calculation_mode,
            channel_estimation_method, training_sequence_len, traceback_depth,
            config, num_data_symbols_per_burst
        ))

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(single_iteration, params_list)

    for i, (ratio, ber) in enumerate(sorted(results, key=lambda x: x[0])):
        ratio_values[i] = ratio
        ber_values[i] = ber
        print(f"  {calculation_mode} = {ratio:5.1f} dB, BER = {ber:.6f}")

    elapsed = time.perf_counter() - start_time
    print(f"Simulation completed with multiprocessing in {elapsed:.2f} seconds")

    return np.array(ratio_values), np.array(ber_values)

def plot_results(ratio_values, ber_values, config):
    plt.figure(figsize=(10, 6))
    ber_plot = np.where(ber_values == 0, 1e-6, ber_values)
    plt.semilogy(ratio_values, ber_plot, 'bo-', linewidth=2, markersize=6)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    title = f"BER vs {config.mode_selection.calculation_mode} for {config.core_simulation_parameters.modulation_type}\n"
    title += (
        f"(Estimation: {config.mode_selection.channel_estimation_method}, "
        f"Interferers: {config.num_interferers})"
    )
    plt.title(title)
    plt.xlabel(f'{config.mode_selection.calculation_mode} (dB)')
    plt.ylabel('BER')
    plt.ylim([1e-5, 1])
    plt.show()

# TODO LIST:
# integrate config (+)
# adding parallel processing(+)
# allocation of functions to modules (-)
# adding burst types for others modulation(-)
def main():
    config_path = "./config/settings.json"
    config = validator.validate_config(loader.ConfigLoader.load(config_path))
    # ----------------------------------------------------------------- ---

    ratio_values, ber_values = simulate(config)

    print('\nFinal BER Results:')
    for r, ber in zip(ratio_values, ber_values):
        print(f'{config.mode_selection.calculation_mode}={r:4.1f} dB => BER={ber:.6f}')
    plot_results(ratio_values, ber_values, config)


if __name__ == '__main__':
    main()