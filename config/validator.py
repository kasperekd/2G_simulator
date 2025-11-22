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
    calculation_mode: Literal["SINR","CI"]
    channel_estimation_method: Literal["ls","corr","true"] 
    combining_mode: Literal["IRC", "MRC", "SAIC"]
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

class SystemConfig(BaseModel):
    num_interferers: int
    use_pim: bool
    core_simulation_parameters: CoreParametrsConfig
    mode_selection: ModeSelectionConfig
    physical_layer_parameters: PhyLayerParametersConfig
    burst_structure_parameters: BurstStructParametersConfig
    file_paths: FilePathsConfig

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