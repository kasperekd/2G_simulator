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

class ChannelConfig(BaseModel):
    """Validation schema for channel parameters."""
    type: Literal["AWGN", "Rayleigh", "Rician"] = Field(
        default="AWGN",
        description="Type of communication channel."
    )

class NoiseConfig(BaseModel):
    """Validation schema for noise parameters."""
    enabled: bool = Field(
        default=True,
        description="Whether noise addition is enabled."
    )
    snr_db: float = Field(
        ge=0,
        description="Signal-to-noise ratio in decibels (must be >= 0)."
    )

class SystemConfig(BaseModel):
    """Main validation schema for system configuration."""
    modulation: Literal["BPSK", "QPSK", "16-QAM", "64-QAM"] = Field(
        ...,
        description="Modulation type (required field)."
    )
    channel: ChannelConfig
    noise: NoiseConfig

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
