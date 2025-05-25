from typing import Tuple, Optional

def extract_config_parameters(config: object) -> Tuple[
    int,          # count_bit
    int,          # sampling_rate
    Optional[int], # number_seed
    str,          # modulation_scheme
    str,          # channel_type
    Optional[float] # SNR
]:
    """Extracts and validates simulation parameters from configuration object.

    Args:
        config: Configuration object with nested parameter structures.
            Expected attributes:
            - generation_signal_configuration
            - modulation_scheme
            - channel_configuration
            - noise_configuration

    Returns:
        Tuple containing:
        - count_bit: Number of bits to generate
        - sampling_rate: Signal sampling rate in Hz
        - number_seed: Random seed value (None if disabled)
        - modulation_scheme: Modulation type string
        - channel_type: Channel model type
        - SNR: Signal-to-noise ratio in dB (None if disabled)

    Raises:
        AttributeError: If required configuration fields are missing
        ValueError: If any parameter has invalid value
    """
    # Signal generation
    gen_config = config.generation_signal_configuration
    count_bit = gen_config.count_bit
    sampling_rate = gen_config.sampling_rate
    
    # Optional seed
    number_seed = (gen_config.seed_configuration.number_seed 
                 if gen_config.seed_configuration.seed_enabled 
                 else None)
    
    # Modulation and channel
    modulation_scheme = config.modulation_scheme
    channel_type = config.channel_configuration.channel_type
    
    # Optional noise
    SNR = (config.noise_configuration.signal_to_noise_ratio_db 
          if config.noise_configuration.noise_enabled 
          else None)

    # Validation
    if count_bit <= 0:
        raise ValueError("Count bit must be positive")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive")

    return count_bit, sampling_rate, number_seed, modulation_scheme, channel_type, SNR