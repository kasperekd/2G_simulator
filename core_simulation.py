from config import loader, validator, extract_parameters
from signal_generation import bit_generator, modulation
from channel import qudriga_importer, awgn, simulate_reception
from analysis import plotter

import matplotlib.pyplot as plt

import numpy as np

if __name__ == "__main__":
    # 1. Получение данных из конфига
    config_path = "D:\\Github\\2G_simulator\\config\settings.json"
    raw_config = loader.ConfigLoader.load(config_path)
    config = validator.SystemConfig(**raw_config)

    count_bit, sampling_rate, number_seed,\
          modulation_type, channel_type, SNR = extract_parameters.extract_config_parameters(config)

    # 2. Генерация битовой последовательности
    sequence_bit1 = bit_generator.generate_bit(count_bit, number_seed)
    sequence_bit2 = bit_generator.generate_bit(count_bit, 1233)
    # plotter.plot_line(sequence_bit1)
    # plotter.plot_line(sequence_bit2)

    # 3. Модуляция
    modulated_signal_s1 = modulation.modulation(modulation_type, sequence_bit1, sampling_rate)
    modulated_signal_s2 = modulation.modulation(modulation_type, sequence_bit2, sampling_rate)
    plotter.plot_scatter(modulated_signal_s1)
    plotter.plot_scatter(modulated_signal_s2)
    # plotter.plot_line(sequence_bit1)

    # 4. канал
    channel_path = "D:\\Github\\2G_simulator\\channel\\Ht2_0204_11.mat"
    h11, h12, h21, h22 = qudriga_importer.load_channel_matrix(channel_path)

    # 5. приём сигнала
    rx_ant1, rx_ant2 = simulate_reception.simulate_reception(
        modulation_type,
        modulated_signal_s1,
        modulated_signal_s2,
        h11, h12, h21, h22,
        SNR,
        num_tests=1000
    )

    plotter.visualize_mimo_reception(rx_ant1, rx_ant2, h11, h12, h21, h22)

    # plotter.plot_scatter(rx_ant1, color="orange")
    # plotter.plot_scatter(rx_ant2)