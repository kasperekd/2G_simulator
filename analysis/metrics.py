import numpy as np

def calculate_ber(original_bits, received_bits) -> float:
    """Calculates the Bit Error Rate (BER) between transmitted and received bit sequences.

    Args:
        original_bits: Array of transmitted bits (0s and 1s).
        received_bits: Array of received bits (0s and 1s).

    Returns:
        BER value between 0 and 1 (ratio of erroneous bits to total bits).

    Raises:
        ValueError: If input arrays have different lengths.
    """
    if len(original_bits) != len(received_bits):
        raise ValueError("Input arrays must have equal length!")
    
    error_count = np.sum(original_bits != received_bits)
    return error_count / len(original_bits)

def calculate_ser(original_symbols, received_symbols) -> float:
    """Calculates the Symbol Error Rate (SER) between transmitted and received symbols.

    Args:
        original_symbols: Array of transmitted symbols (any comparable dtype).
        received_symbols: Array of received symbols.

    Returns:
        SER value between 0 and 1 (ratio of erroneous symbols to total symbols).

    Raises:
        ValueError: If input arrays have different lengths.
    """
    if len(original_symbols) != len(received_symbols):
        raise ValueError("Input arrays must have equal length!")
    
    error_count = np.sum(original_symbols != received_symbols)
    return error_count / len(original_symbols)