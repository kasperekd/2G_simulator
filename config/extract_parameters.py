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
    combining_mode = mode_selection.combining_mode
    irc_regularization = mode_selection.irc_regularization

    # physical layer parameters
    physical_layer_parameters = config.physical_layer_parameters
    bs_nf_db = physical_layer_parameters.bs_nf_db
    fs_hz = physical_layer_parameters.fs_hz
    temp_k = physical_layer_parameters.temp_k

    # burst structure parameters
    burst_structure_parameters = config.burst_structure_parameters
    burst_symbol_rate = burst_structure_parameters.burst_symbol_rate
    num_bursts = burst_structure_parameters.num_bursts
    
    # file paths
    channel_mat_file = config.file_paths.channel_mat_file

    # saic parameters
    saic_cfg = config.saic
    apply_saic_preprocessing = saic_cfg.apply_saic_preprocessing
    saic_method = saic_cfg.method
    saic_regularization = saic_cfg.regularization
    saic_thermal_noise_variance = saic_cfg.thermal_noise_variance

    # temporal whitening parameters
    temp_cfg = config.temporal_whitening
    apply_temporal_whitening = temp_cfg.apply_temporal_whitening
    temporal_method = temp_cfg.method
    temporal_regularization = temp_cfg.regularization
    temporal_thermal_noise_variance = temp_cfg.thermal_noise_variance
    temporal_full_burst = temp_cfg.full_burst

    # power parameters in dBm
    power_cfg = config.power_parameters
    bs_tx_power_dbm = power_cfg.bs_tx_power_dbm
    bs_antenna_gain_dbi = power_cfg.bs_antenna_gain_dbi
    ms_antenna_gain_dbi = power_cfg.ms_antenna_gain_dbi
    path_loss_db = power_cfg.path_loss_db
    channel_bandwidth_hz = power_cfg.channel_bandwidth_hz

    return (
        num_interferers, channel_mat_file, channel_memory, 
        target_ratio_range_db, mod_type, calculation_mode, 
        channel_estimation_method, num_bursts, channel_model, 
        traceback_depth, bs_nf_db, temp_k, fs_hz, burst_symbol_rate,
        combining_mode, irc_regularization,
        apply_saic_preprocessing, saic_method, saic_regularization, saic_thermal_noise_variance,
        apply_temporal_whitening, temporal_method, temporal_regularization, temporal_thermal_noise_variance, temporal_full_burst,
        bs_tx_power_dbm, bs_antenna_gain_dbi, ms_antenna_gain_dbi, path_loss_db, channel_bandwidth_hz
    )