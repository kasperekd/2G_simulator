"""
Module for validating configuration data using Pydantic.

Defines schema structures for verifying configuration parameters
of a communication system, including modulation type, transmission channel, and noise settings.

Classes:
    ChannelConfig: Pydantic model for channel configuration.
    NoiseConfig: Pydantic model for noise parameters.
    SystemConfig: Main Pydantic model for full system configuration validation.

Functions:
    validate_config(config_data): Validates a configuration dictionary
        and returns a SystemConfig object or raises a detailed exception.
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Dict, Any
from pathlib import Path

class SeedConfig(BaseModel):
    """Validation Seed for geneartion signal and burst"""
    seed_enabled: bool = Field(
        default=True,
        deprecation="Whether seed for generation is enabled"
    )
    number_seed: int = Field(
        ge=0,
        description="Seed number in decibels (must be >= 0)."
    )

class GenerationSignalConfig(BaseModel):
    count_bit: int = Field(
        ge=1,
        description="count of bits in decibels (must be >= 1)."
    )
    seed_configuration: SeedConfig
    sampling_rate: int = Field(
        ge=1,
        description="Sampling rate in decibels (must be >= 1)."
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

class SystemConfig(BaseModel):
    """Main validation schema for system configuration."""
    generation_signal_configuration: GenerationSignalConfig
    modulation_scheme: Literal["GMSK", "QPSK", "QAM16"] = Field(
        ...,
        description="Modulation type (required field)."
    )
    channel_configuration: ChannelConfig
    noise_configuration: NoiseConfig

def validate_config(config_data: Dict[str, Any]) -> SystemConfig:
    """Validates a configuration dictionary against the schema.

    Args:
        config_data (Dict[str, Any]): Dictionary containing config values.

    Returns:
        SystemConfig: Validated configuration object.

    Raises:
        ValueError: If validation fails, with detailed error message.
    """
    try:
        return SystemConfig(**config_data)
    except ValidationError as e:
        raise ValueError(f"Config validation failed:\n{e.json(indent=2)}")