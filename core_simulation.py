from config import loader, validator, extract_parameters
from signal_generation import bit_generator, modulation
from channel import qudriga_importer, awgn, simulate_reception
from analysis import plotter

import numpy as np

if __name__ == "__main__":
    # 1. Получение данных из конфига
    config_path = "D:\\Github\\2G_simulator\\config\settings.json"
    raw_config = loader.ConfigLoader.load(config_path)
    config = validator.SystemConfig(**raw_config)

    count_bit, sampling_rate, number_seed,\
          modulation_scheme, channel_type, SNR = extract_parameters.extract_config_parameters(config)

    # 2. Генерация битовой последовательности
    sequence_bit1 = bit_generator.generate_bit(count_bit)
    sequence_bit2 = bit_generator.generate_bit(count_bit)
    # plotter.plot_line(sequence_bit)

    # 3. Модуляция
    modulated_signal_s1 = modulation.modulation(modulation_scheme, sequence_bit1, sampling_rate)
    modulated_signal_s2 = modulation.modulation(modulation_scheme, sequence_bit2, sampling_rate)
    plotter.plot_scatter(modulated_signal_s1)
    
    # 4. канал
    channel_path = "D:\\Github\\2G_simulator\\channel\\Ht2_0204_11.mat"
    h11, h12, h21, h22 = qudriga_importer.load_channel_matrix(channel_path)

    plotter.plot_line(sequence_bit1)

    # 5. приём сигнала
    rx_ant1, rx_ant2 = simulate_reception.simulate_reception(
        modulated_signal_s1,
        modulated_signal_s2,
        h11, h12, h21, h22,
        SNR,
        num_tests=1000
    )

    plotter.plot_scatter(rx_ant1, color="orange")
    plotter.plot_scatter(rx_ant2)