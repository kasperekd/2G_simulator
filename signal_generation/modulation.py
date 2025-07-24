import numpy as np
from commpy.modulation import PSKModem, QAMModem

def _modulate_gmsk(bit_sequence, samp_rate):
    """
    Manual implementation of a simplified GMSK modulator.
    This linear approximation is suitable for use with an MLSE equalizer.
    """
    if len(bit_sequence) == 0:
        return np.array([]), np.array([])

    # GMSK is a binary CPM, often approximated as filtered BPSK for MLSE.
    # Convert bits {0, 1} to symbols {-1, 1}
    symbols = 2 * np.array(bit_sequence) - 1.0
    
    # Symbol indices are simply the bits themselves for a binary modulation
    symbol_indices = bit_sequence

    # GMSK parameters (typical for GSM)
    bt_product = 0.3
    filter_span_in_symbols = 4

    # Upsample the symbols
    if samp_rate > 1:
        upsampled = np.zeros(len(symbols) * samp_rate)
        upsampled[::samp_rate] = symbols
    else:
        upsampled = symbols

    # Create the Gaussian filter
    t = np.arange(-filter_span_in_symbols/2, filter_span_in_symbols/2, 1.0/samp_rate)
    gaussian_filter = np.exp(-2 * (np.pi**2) * (bt_product**2) / np.log(2) * (t**2))
    gaussian_filter /= np.sum(gaussian_filter)

    # Convolve with the Gaussian filter to get the frequency pulse
    frequency_pulse = np.convolve(upsampled, gaussian_filter, mode='same')
    
    # Integrate frequency to get phase
    phase = np.cumsum(frequency_pulse) * (np.pi / (2 * samp_rate))
    
    # Modulate phase onto a complex carrier
    signal = np.exp(1j * phase)

    return signal, symbol_indices


def modulate_bits(modem, bit_sequence: np.ndarray, samp_rate: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Modulates a bit sequence using a provided commpy modem or manual GMSK.

    Args:
        modem: An instantiated commpy modem object or the string "GMSK".
        bit_sequence (np.ndarray): The input binary sequence.
        samp_rate (int): The oversampling factor.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - modulated_signal (complex): The complex baseband signal.
            - symbol_indices (int): The sequence of indices corresponding to the symbols.
    """
    if modem == "GMSK":
        return _modulate_gmsk(bit_sequence, samp_rate)

    # This part is for commpy modems (PSK, QAM)
    complex_symbols = modem.modulate(bit_sequence)
    symbol_indices = np.array([np.argmin(np.abs(s - modem.constellation)) for s in complex_symbols])

    if samp_rate > 1:
        upsampled = np.zeros(len(complex_symbols) * samp_rate, dtype=np.complex128)
        upsampled[::samp_rate] = complex_symbols
        pulse_shape = np.ones(samp_rate) / np.sqrt(samp_rate)
        signal = np.convolve(upsampled, pulse_shape, mode='same')
    else:
        signal = complex_symbols

    return signal, symbol_indices


def create_modem(mod_type: str):
    """
    Factory function to create and return a commpy modem object or a GMSK identifier.
    
    Args:
        mod_type (str): The modulation type ("QPSK", "8PSK", "16QAM", "GMSK").
        
    Returns:
        A commpy modem object or the string "GMSK".
    """
    if mod_type == "QPSK":
        return PSKModem(m=4)
    elif mod_type == "8PSK":
        return PSKModem(m=8)
    elif mod_type == "QAM16":
        return QAMModem(m=16)
    elif mod_type == "GMSK":
        # Since we are implementing GMSK manually
        return "GMSK"
    else:
        raise ValueError(f"Unsupported modulation type for modem factory: {mod_type}")


def demodulate_symbols(modem, symbol_indices: np.ndarray) -> np.ndarray:
    """
    Demodulates a sequence of symbol indices back to bits.

    Args:
        modem: The commpy modem object or the string "GMSK".
        symbol_indices (np.ndarray): The sequence of decoded symbol indices.

    Returns:
        np.ndarray: The resulting bit sequence.
    """
    if modem == "GMSK":
        # For our BPSK-like GMSK, the symbol indices are the bits themselves.
        return symbol_indices

    # For commpy modems
    complex_symbols = modem.constellation[symbol_indices]
    return modem.demodulate(complex_symbols, demod_type='hard')