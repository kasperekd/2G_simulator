import numpy as np

def generate_burst(tx_data, training_seq):
    """
    Generate a GSM normal burst (148 bits) according to GSM 05.05 specification.

    A normal GSM burst consists of:
        [TAIL | DATA1 | CTRL | TRAINING | CTRL | DATA2 | TAIL]
        [  3  |  57   |  1   |   26     |  1   |  57   |  3   ] bits

    If tx_data is shorter than 114 bits, the remaining bits will be padded with zeros.

    Parameters
    ----------
    tx_data : np.ndarray
        Input data bits (should be 114 bits, otherwise will be padded or truncated).
    training_seq : np.ndarray
        Training sequence (must be exactly 26 bits long) used for synchronization
        and channel estimation.

    Returns
    -------
    burst : np.ndarray
        A 1D array of 148 bits representing a complete GSM normal burst.
    """
    # Padding tx_data if it's shorter than 114 bits
    if len(tx_data) < 114:
        tx_data = np.concatenate([tx_data, np.zeros(114 - len(tx_data), dtype=int)])

    TAIL = np.array([0, 0, 0])
    CTRL = np.array([1])

    return np.concatenate([
        TAIL,
        tx_data[:57],
        CTRL,
        training_seq,
        CTRL,
        tx_data[57:],
        TAIL
    ])
