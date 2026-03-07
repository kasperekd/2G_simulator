import numpy as np
def generate_data_bits(num_bits, rng):
    # TODO: Replace with a proper data source block, possibly including channel coding.
    return rng.integers(0, 2, num_bits)