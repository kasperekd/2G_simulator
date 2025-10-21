import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import scipy.io as spio
from typing import Dict

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
# 2. CONFIGURATION
# ==============================================================================

class MLSEConfig:
    def __init__(self):
        # TODO: Implement a function to select these based on Test Case
        self.num_interferers = 1
        self.use_pim = False

        # Core Simulation Parameters
        self.modulation_type = 'QPSK'
        self.channel_model = 'TU50'  # 'AWGN' or 3GPP models
        self.channel_memory = 3
        self.num_data_symbols_per_burst = 116
        self.target_ratio_range_db = np.arange(0, 25, 2)  # This is C/I or SINR range
        self.num_bursts = 100

        # Mode Selection
        self.calculation_mode = 'SINR'  # 'CI' or 'SINR'
        self.channel_estimation_method = 'ls'  # 'true' (perfect) or 'ls' (least squares)

        # Physical Layer Parameters
        self.bs_nf_db = 3.0      # Noise Figure in dB
        self.fs_hz = 1083333.33  # Sampling frequency
        self.temp_k = 300        # System temperature in Kelvin

        # Burst Structure Parameters
        self.training_sequence_len = 26
        self.traceback_depth = 15

        # File Paths
        self.channel_mat_file = 'Ht2_0204_11.mat'

        # Derived parameters
        self.modem = Modulator(self.modulation_type)
        self.bits_per_symbol = self.modem.bits_per_symbol
        self.constellation = self.modem.constellation
        ts_indices = np.random.randint(
            0, len(self.constellation), self.training_sequence_len
        )
        self.training_sequence = self.constellation[ts_indices]

# ==============================================================================
# 3. HELPER AND SIMULATION FUNCTIONS
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


def add_thermal_noise(signal, config: MLSEConfig):
    # Physical noise calculation
    k_boltzmann = 1.380649e-23
    nf_linear = 10**(config.bs_nf_db / 10)
    # TODO: Noise bandwidth should be channel bandwidth (e.g. 200e3 for GSM) not Fs
    noise_power = k_boltzmann * config.temp_k * config.fs_hz * nf_linear
    noise_std_dev = np.sqrt(noise_power / 2)
    noise = noise_std_dev * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise


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


def simulate(config: MLSEConfig):
    h11, h12, h21, h22 = load_quadriga_channel(config.channel_mat_file)
    L = config.channel_memory
    modem = config.modem

    ratio_values, ber_values = [], []

    print("Starting simulation:")
    print(
        f"Modulation: {config.modulation_type}, "
        f"Mode: {config.calculation_mode}, "
        f"Estimation: {config.channel_estimation_method}"
    )

    for target_ratio_db in config.target_ratio_range_db:
        total_errors, total_bits = 0, 0
        for _ in range(config.num_bursts):
            # 1. TRANSMITTER SIDE
            data_bits = generate_data_bits(
                config.num_data_symbols_per_burst * config.bits_per_symbol
            )
            tx_burst, original_data_symbols = create_burst(
                data_bits, modem, config.training_sequence, L
            )

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
            for _ in range(config.num_interferers):
                interf_bits = generate_data_bits(
                    len(tx_burst) * config.bits_per_symbol
                )
                interf_syms = modem.modulate(interf_bits)
                h_interf_ant1 = h21[:L, channel_idx]
                h_interf_ant2 = h22[:L, channel_idx]
                interf_conv1 = convolve(interf_syms, h_interf_ant1, 'full')
                interf_conv2 = convolve(interf_syms, h_interf_ant2, 'full')
                total_interf_rx_ant1[:len(interf_conv1)] += interf_conv1
                total_interf_rx_ant2[:len(interf_conv2)] += interf_conv2
            # TODO: Add PIM interference generation if config.use_pim is True

            # 4. SCALING AND COMBINING
            signal_power = np.mean(np.abs(s1_rx_ant1)**2 + np.abs(s1_rx_ant2)**2)
            current_interf_power = np.mean(
                np.abs(total_interf_rx_ant1)**2 + np.abs(total_interf_rx_ant2)**2
            )
            if current_interf_power < 1e-20:
                current_interf_power = 1e-20

            target_ratio_linear = 10**(target_ratio_db / 10)

            if config.calculation_mode == 'CI':
                required_interf_power = signal_power / target_ratio_linear
            else:  # SINR
                k_b = 1.380649e-23
                nf_lin = 10**(config.bs_nf_db / 10)
                noise_power = k_b * config.temp_k * config.fs_hz * nf_lin * 2
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
            if config.channel_estimation_method == 'true':
                h_est_ant1, h_est_ant2 = h_true_ant1, h_true_ant2
            else:  # 'ls'
                ts_start_idx = len(original_data_symbols) // 2 + L
                ts_end_idx = ts_start_idx + config.training_sequence_len
                h_est_ant1 = estimate_channel_ls(
                    rx_ant1_noisy[ts_start_idx:ts_end_idx + L - 1],
                    config.training_sequence,
                    L
                )
                h_est_ant2 = estimate_channel_ls(
                    rx_ant2_noisy[ts_start_idx:ts_end_idx + L - 1],
                    config.training_sequence,
                    L
                )

            # 7. RECEIVER: Equalization and Decoding
            # FIXME: Simple averaging is suboptimal. Implement proper MRC.
            rx_combined = (rx_ant1_noisy + rx_ant2_noisy) / 2
            h_est_avg = (h_est_ant1 + h_est_ant2) / 2

            mlse_input = rx_combined[:len(tx_burst) + L - 1]
            decoded_indices = mlse_viterbi_decode(
                mlse_input, h_est_avg, config.constellation, config.traceback_depth
            )

            # 8. BER CALCULATION
            d1_len = len(original_data_symbols) // 2
            d2_len = len(original_data_symbols) - d1_len
            decoded_d1_indices = decoded_indices[L: L + d1_len]
            d2_start_idx = L + d1_len + config.training_sequence_len
            d2_end_idx = d2_start_idx + d2_len
            decoded_d2_indices = decoded_indices[d2_start_idx: d2_end_idx]
            decoded_data_indices = np.concatenate([
                decoded_d1_indices, decoded_d2_indices
            ])
            decoded_bits = modem.demodulate_indices(decoded_data_indices)

            total_errors += np.sum(data_bits != decoded_bits[:len(data_bits)])
            total_bits += len(data_bits)

        ber = total_errors / total_bits if total_bits > 0 else 0.5
        ratio_values.append(target_ratio_db)
        ber_values.append(ber)
        print(f"  {config.calculation_mode} = {target_ratio_db:5.1f} dB, BER = {ber:.6f}")

    return np.array(ratio_values), np.array(ber_values)


def plot_results(ratio_values, ber_values, config):
    plt.figure(figsize=(10, 6))
    ber_plot = np.where(ber_values == 0, 1e-6, ber_values)
    plt.semilogy(ratio_values, ber_plot, 'bo-', linewidth=2, markersize=6)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    title = f"BER vs {config.calculation_mode} for {config.modulation_type}\n"
    title += (
        f"(Estimation: {config.channel_estimation_method}, "
        f"Interferers: {config.num_interferers})"
    )
    plt.title(title)
    plt.xlabel(f'{config.calculation_mode} (dB)')
    plt.ylabel('BER')
    plt.ylim([1e-5, 1])
    plt.show()


def main():
    np.random.seed(111)
    config = MLSEConfig()

    config.modulation_type = 'QPSK'
    config.calculation_mode = 'CI'
    config.channel_estimation_method = 'ls'
    config.target_ratio_range_db = np.arange(-5, 21, 0.5)
    config.modem = Modulator(config.modulation_type)
    config.constellation = config.modem.constellation
    # ----------------------------------------------------------------- ---

    ratio_values, ber_values = simulate(config)

    print('\nFinal BER Results:')
    for r, ber in zip(ratio_values, ber_values):
        print(f'{config.calculation_mode}={r:4.1f} dB => BER={ber:.6f}')
    plot_results(ratio_values, ber_values, config)


if __name__ == '__main__':
    main()