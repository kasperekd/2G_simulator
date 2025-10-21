from .validator import SystemConfig
from typing import Tuple, Optional, List, Any

def extract_config_parameters(config: SystemConfig) -> Tuple[
    int,          # count_bit
    int,          # sampling_rate
    Optional[int],# number_seed
    str,          # modulation_type
    str,          # channel_type
    Optional[float], # SNR
    Any           # receiver_config
]:
    """Extracts and validates simulation parameters from a configuration object.

    Args:
        config: A validated SystemConfig object.

    Returns:
        Tuple containing all major simulation parameters.

    Raises:
        ValueError: If any parameter has an invalid value (already handled by Pydantic).
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
    modulation_type = config.modulation_type
    channel_type = config.channel_configuration.channel_type
    
    # Optional noise
    SNR = (config.noise_configuration.signal_to_noise_ratio_db 
          if config.noise_configuration.noise_enabled 
          else None)

    # Receiver config
    receiver_config = config.receiver_configuration
    
    return count_bit, sampling_rate, number_seed, modulation_type, channel_type, SNR, receiver_config