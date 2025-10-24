import numpy as np
# TODO: changing the parameters for the types of modulation
def create_burst(data_bits, modem, training_sequence, channel_memory):
    data_symbols = modem.modulate(data_bits)
    part1_len = len(data_symbols) // 2
    part1_syms = data_symbols[:part1_len]
    part2_syms = data_symbols[part1_len:]
    # tail_symbols = modem.modulate(tail_bits)
    tail_symbols = np.zeros(3)
    # training_sequence = modem.modulate(training_sequence)
    burst = np.concatenate([
        tail_symbols, part1_syms, training_sequence, part2_syms, tail_symbols
    ])
    return burst, data_symbols