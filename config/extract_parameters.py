from .validator import SystemConfig
import numpy as np

def extract_config_parameters(config: SystemConfig):
    num_interferers = config.num_interferers

    # core parameters
    core_parameters = config.core_simulation_parameters
    mod_type = core_parameters.modulation_type
    channel_model = core_parameters.channel_model
    channel_memory = core_parameters.channel_memory
    range_db = core_parameters.target_ratio_range_db
    target_ratio_range_db = np.arange(range_db.start, range_db.stop, range_db.step)
    traceback_depth = core_parameters.traceback_depth

    # mode selection
    mode_selection = config.mode_selection
    calculation_mode = mode_selection.calculation_mode
    channel_estimation_method = mode_selection.channel_estimation_method

    # physical layer parameters
    physical_layer_parameters = config.physical_layer_parameters
    bs_nf_db = physical_layer_parameters.bs_nf_db
    fs_hz = physical_layer_parameters.fs_hz
    temp_k = physical_layer_parameters.temp_k

    # burst structure parameters
    # TODO: added QAM32 in core_simulation, settings, validator
    burst_structure_parameters = config.burst_structure_parameters
    burst_symbol_rate = burst_structure_parameters.burst_symbol_rate
    if burst_symbol_rate == 'normal':
        burst_dict = {
            'GMSK': [[0,0,0], 58 * 2, 26, [0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1], 8],
            '8PSK': [[1,1,1,1,1,1,1,1,1], 174 * 2, 78, [1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,0,0,1], 24],
            'QAM16': [[0,0,0,1,0,1,1,0,0,1,1,0], 232 * 2, 104, [1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1], 33],
        }
    elif burst_symbol_rate == 'higher':
        burst_dict = {
            'QPSK': [[0,0,0,1,1,1,1,0], 138 * 2, 62, [0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0], 21],
            'QAM16': [[0,0,0,1,0,1,1,0,0,1,1,0,1,1,0,1], 276 * 2, 124, [0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1], 42],
        }
    num_bursts = burst_structure_parameters.num_bursts
    if mod_type not in burst_dict:
        raise KeyError(f"mod type '{mod_type}' missing from the dictionary burst_dict")
    tail_bits, num_data_bits_per_burst, training_sequence_len, training_sequence, guard_period = burst_dict[mod_type]

    # file paths
    channel_mat_file = config.file_paths.channel_mat_file

    return (num_interferers, channel_mat_file, channel_memory, 
            target_ratio_range_db, mod_type, calculation_mode, 
            channel_estimation_method, num_bursts, channel_model, 
            num_data_bits_per_burst, training_sequence_len, traceback_depth,
            training_sequence, bs_nf_db, temp_k, fs_hz, tail_bits, guard_period)