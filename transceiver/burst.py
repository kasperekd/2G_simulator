import numpy as np
# TODO: changing the parameters for the types of modulation
def create_burst(data_bits, modem, training_sequence, tail_bits, guard_period):
    data_symbols = modem.modulate(data_bits)
    part1_len = len(data_symbols) // 2
    part1_syms = data_symbols[:part1_len]
    part2_syms = data_symbols[part1_len:]
    guard_symbols = modem.modulate(np.zeros(guard_period))
    tail_symbols = modem.modulate(tail_bits)
    burst = np.concatenate([
        tail_symbols, part1_syms, training_sequence, part2_syms, tail_symbols, guard_symbols
    ])
    return burst, data_symbols, tail_symbols