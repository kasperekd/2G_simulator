import numpy as np
def extract_data_segments(original_data_symbols, decoded_indices, tail_symbols, training_sequence_len, modem, data_bits):
    d1_len = len(original_data_symbols) // 2
    d2_len = len(original_data_symbols) - d1_len
    decoded_d1_indices = decoded_indices[len(tail_symbols): len(tail_symbols) + d1_len]

    d2_start_idx = len(tail_symbols) + d1_len + training_sequence_len // (modem.bits_per_symbol)
    d2_end_idx = d2_start_idx + d2_len
    decoded_d2_indices = decoded_indices[d2_start_idx: d2_end_idx]

    decoded_data_indices = np.concatenate([decoded_d1_indices, decoded_d2_indices])
    return decoded_data_indices