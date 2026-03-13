import numpy as np
def generate_data_bits(num_bits, seed):
    # TODO: Replace with a proper data source block, possibly including channel coding.
    # np.random.seed(seed)
    return np.random.randint(0, 2, num_bits)