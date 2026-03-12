"""
Module for validating configuration data using Pydantic.

This module defines schema structures for validating configuration parameters 
of a communication system, including modulation type, transmission channel, noise, 
and receiver settings.

Classes:
    SeedConfig: Pydantic model for PRNG seed configuration.
    GenerationSignalConfig: Pydantic model for signal generation settings.
    ChannelConfig: Pydantic model for channel configuration.
    NoiseConfig: Pydantic model for noise parameters.
    ReceiverConfig: Pydantic model for receiver parameters.
    SystemConfig: Main Pydantic model for full system configuration validation.
"""

from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Literal, Dict, Any, List, Optional

class TargetRationRangeConfig(BaseModel):
    start: int
    stop: int
    step: float

class CoreParametrsConfig(BaseModel):
    '''Validation for core simulation'''
    modulation_type: Literal['BPSK',"GMSK", "8PSK", "QPSK", "QAM16", "QAM32"]
    channel_model: Literal["AWGN", "TU50", "EQ50", "HT100", "RA130"]
    channel_memory: int = Field(
        ge=2,
        description="channel memory (must be >= 2)."
    )
    target_ratio_range_db: TargetRationRangeConfig
    traceback_depth: int

class ModeSelectionConfig(BaseModel):
    calculation_mode: Literal["SINR","CI", "SNR"]
    channel_estimation_method: Literal["ls","corr","true","lmmse"] 
    combining_mode: Literal["IRC", "ST-IRC", "MRC", "EGC", "SAIC", "SINGLE"]
    st_irc_method: Literal["direct", "ar-prewhitening"] = Field(
        default="direct",
        description="Method for Space-Time IRC: 'direct' (MMSE weights) or 'ar-prewhitening' (Cholesky noise whitening)."
    )
    irc_regularization: float = Field(
        ge=0,
        description="IRC regularization parameter (must be >= 0)."
    )
    
class PhyLayerParametersConfig(BaseModel):
    bs_nf_db: float
    fs_hz: float
    temp_k: int

class BurstStructParametersConfig(BaseModel):
    burst_symbol_rate: Literal["normal","higher"]    
    num_bursts: int = Field(
        ge=1,
        description="num_bursts (must be >= 1)."
    )

class FilePathsConfig(BaseModel):
    channel_mat_file: str

class ResultsOutputConfig(BaseModel):
    save_results: bool = Field(
        default=False,
        description="Whether to save simulation results to CSV file."
    )
    save_mse_debug: bool = Field(
        default=False,
        description="Calculate and save Channel Estimation."
    )
    output_directory: str = Field(
        default="./results",
        description="Directory where CSV results will be saved."
    )

class SAICConfig(BaseModel):
    apply_saic_preprocessing: bool = Field(
        default=False,
        description="Apply single-antenna SAIC whitening preprocessing before combining."
    )
    method: Literal['basic', 'bias_removal'] = Field(
        default='basic',
        description="SAIC preprocessing method."
    )
    regularization: float = Field(
        default=1e-6,
        ge=0,
        description="Regularization parameter for SAIC whitening."
    )
    thermal_noise_variance: float = Field(
        default=1e-6,
        ge=0,
        description="Estimated thermal noise variance used in bias removal."
    )


class TemporalWhiteningConfig(BaseModel):
    apply_temporal_whitening: bool = Field(
        default=False,
        description="Apply temporal whitening preprocessing before combining."
    )
    method: Literal['mahalanobis', 'bias_removal'] = Field(
        default='bias_removal',
        description="Temporal whitening method to use."
    )
    regularization: float = Field(
        default=1e-6,
        ge=0,
        description="Regularization parameter for temporal whitening."
    )
    thermal_noise_variance: Optional[float] = Field(
        default=None,
        description="Estimated thermal noise variance (optional) used in bias removal."
    )
    full_burst: bool = Field(
        default=True,
        description="Apply whitening to full burst (True) or only TS (False)."
    )


class ChannelSweepConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Enable automatic sweep across multiple channel models."
    )
    models: List[Literal['AWGN', 'TU50', 'EQ50', 'HT100', 'RA130']] = Field(
        default_factory=lambda: ['AWGN', 'TU50', 'EQ50', 'HT100', 'RA130'],
        description="List of channel model names to sweep over when enabled."
    )
    model_to_file: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Optional mapping of channel model name to channel mat file path. If absent, the sweeper will try ./channel/{model}.mat."
    )

class SystemConfig(BaseModel):
    num_interferers: int
    use_pim: bool
    core_simulation_parameters: CoreParametrsConfig
    mode_selection: ModeSelectionConfig
    physical_layer_parameters: PhyLayerParametersConfig
    burst_structure_parameters: BurstStructParametersConfig
    file_paths: FilePathsConfig
    results_output: ResultsOutputConfig = Field(default_factory=ResultsOutputConfig)
    saic: SAICConfig = Field(default_factory=SAICConfig)
    temporal_whitening: TemporalWhiteningConfig = Field(default_factory=TemporalWhiteningConfig)
    channel_sweep: ChannelSweepConfig = Field(default_factory=ChannelSweepConfig)

def validate_config(config_data: Dict[str, Any]) -> SystemConfig:
    """Validates a configuration dictionary against the schema.

    Args:
        config_data (Dict[str, Any]): Dictionary containing config values.

    Returns:
        SystemConfig: A validated configuration object.

    Raises:
        ValueError: If validation fails, with a detailed error message.
    """
    try:
        return SystemConfig(**config_data)
    except ValidationError as e:
        raise ValueError(f"Config validation failed:\n{e.json(indent=2)}")