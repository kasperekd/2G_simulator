from .validator import SystemConfig
from typing import Tuple, Optional, List, Any

def extract_config_parameters(config: SystemConfig):
    num_interferers = config.num_interferers

    # core parameters
    core_parameters = config.core_simulation_parameters
    mod_type = core_parameters.modulation_type
    channel_model = core_parameters.channel_model
    channel_memory = core_parameters.channel_memory
    num_data_symbols_per_burst = core_parameters.num_data_symbols_per_burst
    range_db = core_parameters.target_ratio_range_db
    target_ratio_range_db = [range_db.start, range_db.stop, range_db.step]
    num_burst = core_parameters.num_bursts

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
    burst_structure_parameters = config.burst_structure_parameters
    training_sequence_len = burst_structure_parameters.training_sequence_len
    traceback_depth = burst_structure_parameters.traceback_depth

    # file paths
    channel_mat_file = config.file_paths.channel_mat_file

    # table training sequence
    table_training_sequence = config.table_training_sequence
    if mod_type == 'BPSK':
        training_sequence = table_training_sequence.ts_bpsk
    elif mod_type == 'GMSK':
        training_sequence = table_training_sequence.ts_gmsk
    elif mod_type == '8PSK':
        training_sequence = table_training_sequence.ts_8psk
    elif mod_type == 'QPSK':
        training_sequence = table_training_sequence.ts_qpsk
    elif mod_type == 'QAM16':
        training_sequence = table_training_sequence.ts_qam16
    # TODO: added QAM32 in core_simulation, settings, validator
    # elif mod_type == 'QAM32':
    #     training_sequence = table_training_sequence.ts_qam32
    return num_interferers, channel_mat_file, channel_memory, target_ratio_range_db, mod_type, calculation_mode, channel_estimation_method, num_burst, channel_model, num_data_symbols_per_burst, training_sequence_len, traceback_depth, training_sequence, bs_nf_db, temp_k, fs_hz