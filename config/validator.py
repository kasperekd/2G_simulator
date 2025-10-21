"""
Module for validating configuration data using Pydantic.

Defines schema structures for verifying configuration parameters
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
from typing import Literal, Dict, Any, List

class SeedConfig(BaseModel):
    """Validation for signal and burst generation seed."""
    seed_enabled: bool = Field(
        default=True,
        description="Whether the seed for generation is enabled."
    )
    number_seed: int = Field(
        ge=0,
        description="Seed number (must be >= 0)."
    )

class GenerationSignalConfig(BaseModel):
    count_bit: int = Field(
        ge=1,
        description="Number of data bits to generate (must be >= 1)."
    )
    seed_configuration: SeedConfig
    sampling_rate: int = Field(
        ge=1,
        description="Sampling rate per symbol (must be >= 1)."
    )

class ChannelConfig(BaseModel):
    """Validation schema for channel parameters."""
    channel_type: Literal["AWGN", "Rayleigh", "Rician"] = Field(
        default="AWGN",
        description="Type of communication channel."
    )

class NoiseConfig(BaseModel):
    """Validation schema for noise parameters."""
    noise_enabled: bool = Field(
        ...,
        description="Whether noise addition is enabled."
    )
    signal_to_noise_ratio_db: float = Field(
        ge=0,
        description="Signal-to-noise ratio in decibels (must be >= 0)."
    )

class ReceiverConfig(BaseModel):
    """Validation schema for receiver parameters."""
    receiver_type: Literal["IRC-MLSE", "MRC-MLSE"] = Field(
        default="IRC-MLSE",
        description="The type of MLSE receiver to use."
    )
    channel_memory_L: int = Field(
        ge=1,
        description="The assumed memory (length of CIR) for the Viterbi decoder."
    )
    training_sequence_len: int = Field(
        ge=1,
        description="The length of the training sequence."
    )
    training_sequence: List[int] = Field(
        ...,
        description="The training sequence itself (e.g., BPSK symbols -1, 1)."
    )

    @field_validator('training_sequence')
    def check_ts_length(cls, v, values):
        # The 'values' object is a Pydantic internal data structure
        if 'training_sequence_len' in values.data and len(v) != values.data['training_sequence_len']:
            raise ValueError('Length of training_sequence must match training_sequence_len')
        return v
    
class MonteCarloConfig(BaseModel):
    """Validation schema for Monte Carlo simulation parameters."""
    ci_db_min: int = Field(...)
    ci_db_max: int = Field(...)
    ci_db_step: int = Field(gt=0)
    num_iterations: int = Field(gt=0, description="Number of runs to average per data point.")

class SimulationModeConfig(BaseModel):
    """Validation schema for the simulation mode."""
    mode: Literal["SingleRun", "MonteCarlo"] = Field(
        default="SingleRun",
        description="Selects the simulation operational mode."
    )
    single_run_ci_db: float = Field(
        default=0,
        description="C/I value in dB for the SingleRun mode."
    )
    monte_carlo: MonteCarloConfig
    
class SystemConfig(BaseModel):
    """Main validation schema for the entire system configuration."""
    simulation_mode: SimulationModeConfig
    generation_signal_configuration: GenerationSignalConfig
    modulation_type: Literal["BPSK", "GMSK", "8PSK", "QPSK", "QAM16"]
    channel_configuration: ChannelConfig
    noise_configuration: NoiseConfig
    receiver_configuration: ReceiverConfig

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