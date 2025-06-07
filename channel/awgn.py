import numpy as np

def add_awgn(signal: np.ndarray, 
             snr_db: float,
             modulation_type: str) -> np.ndarray:
    """
    Add AWGN with proper energy calculations for different modulation types.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (complex-valued)
    snr_db : float
        Signal-to-noise ratio in dB
    modulation_type : str
        Type of modulation ('BPSK', 'QPSK', '8PSK', 'QAM16')
    
    Returns:
    --------
    tuple:
        - Noisy signal
        - Dictionary with energy parameters (Eb/N0, Es/N0, etc.)
    """
    bits_per_symbol = {
        'GMSK': 1,
        'QPSK': 2,
        '8PSK': 3,
        'QAM16': 4
    }
    
    if modulation_type not in bits_per_symbol:
        raise ValueError(f"Unsupported modulation type: {modulation_type}")
    
    k = bits_per_symbol[modulation_type]
    
    snr_linear = 10 ** (snr_db / 10)
    
    Es = np.mean(np.abs(signal) ** 2)

    Eb = Es / k
    
    N0 = Es / (k * snr_linear)

    noise_power = N0 * k

    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 
            1j*np.random.randn(*signal.shape))
    
    # if needs :)
    # energy_params = {
    #     'Es': Es,
    #     'Eb': Eb,
    #     'N0': N0,
    #     'Eb/N0': Eb/N0,
    #     'Es/N0': Es/N0,
    #     'Eb/N0_dB': 10*np.log10(Eb/N0),
    #     'Es/N0_dB': 10*np.log10(Es/N0)
    # }
    
    return signal + noise