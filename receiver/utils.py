import numpy as np

def dec2base(d: np.ndarray, base: int, ndigits: int) -> np.ndarray:
    """
    Converts a decimal number or array of numbers to a specified base.

    Args:
        d (np.ndarray): The decimal number(s) to convert.
        base (int): The target base.
        ndigits (int): The number of digits in the output representation.

    Returns:
        np.ndarray: The base-B representation of the number(s).
    """
    d = np.asarray(d)
    powers = base**np.arange(ndigits - 1, -1, -1)
    
    if d.ndim == 0:
        return (d // powers) % base
    else:
        return (d[:, np.newaxis] // powers) % base

def base2dec(b: np.ndarray, base: int) -> np.ndarray:
    """
    Converts a number or array of numbers from a specified base to decimal.

    Args:
        b (np.ndarray): The base-B representation(s).
        base (int): The base of the input number(s).

    Returns:
        np.ndarray: The decimal representation.
    """
    b = np.asarray(b)
    ndigits = b.shape[-1]
    powers = base**np.arange(ndigits - 1, -1, -1)
    
    return np.sum(b * powers, axis=-1)