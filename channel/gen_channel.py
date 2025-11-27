import numpy as np
import matplotlib.pyplot as plt
import math

def generate_cir(
    channel_model='TU50',
    carrier_frequency=900e6,
    sampling_rate=1e6,
    num_time_steps=100,
    num_tx_ant=1,
    num_rx_ant=1,
    random_seed=42
):
    """
    Генерирует импульсную характеристику канала (CIR) для заданной модели.
    """
    
    # Определение параметров модели канала
    channel_params = _get_channel_parameters(channel_model)
    delays = channel_params['delays']
    powers_db = channel_params['powers_db']
    velocity_kmh = channel_params['velocity_kmh']
    
    powers_linear = 10**(powers_db / 10.0)
    powers_linear = powers_linear / np.sum(powers_linear)
    
    velocity_ms = velocity_kmh / 3.6
    c = 3e8
    max_doppler_hz = velocity_ms * carrier_frequency / c
    
    mean_delay = np.sum(powers_linear * delays)
    rms_delay_spread = np.sqrt(
        np.sum(powers_linear * (delays**2)) - mean_delay**2
    )
    coherence_bw = 1 / (5 * rms_delay_spread)
    coherence_time = 9 / (16 * np.pi * max_doppler_hz) if max_doppler_hz > 0 else np.inf
    
    # Генерация CIR
    a, tau = _generate_cir(
        delays=delays,
        powers=powers_linear,
        carrier_freq=carrier_frequency,
        max_doppler=max_doppler_hz,
        sampling_rate=sampling_rate,
        num_time_steps=num_time_steps,
        num_tx_ant=num_tx_ant,
        num_rx_ant=num_rx_ant,
        random_seed=random_seed
    )
    
    # bonus info
    channel_info = {
        'channel_model': channel_model,
        'carrier_frequency': carrier_frequency,
        'velocity_kmh': velocity_kmh,
        'velocity_ms': velocity_ms,
        'max_doppler_hz': max_doppler_hz,
        'rms_delay_spread': rms_delay_spread,
        'coherence_bandwidth': coherence_bw,
        'coherence_time': coherence_time,
        'num_paths': len(delays),
        'delays_us': delays * 1e6,
        'powers_db': powers_db
    }
    
    return a, tau, channel_info


def _get_channel_parameters(channel_model):
    """
    Возвращает параметры задержек и мощностей для заданной модели канала.
    """
    models = {
        'TU50': {
            # Typical Urban, 50 km/h
            'delays': np.array([0.0, 0.2, 0.5, 1.6, 2.3, 5.0]) * 1e-6,
            'powers_db': np.array([-3.0, 0.0, -2.0, -6.0, -8.0, -10.0]),
            'velocity_kmh': 50
        },
        'TU3': {
            # Typical Urban, 3 km/h
            'delays': np.array([0.0, 0.2, 0.5, 1.6, 2.3, 5.0]) * 1e-6,
            'powers_db': np.array([-3.0, 0.0, -2.0, -6.0, -8.0, -10.0]),
            'velocity_kmh': 3
        },
        'RA130': {
            # Rural Area, 130 km/h
            'delays': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) * 1e-6,
            'powers_db': np.array([0.0, -4.0, -8.0, -12.0, -16.0, -20.0]),
            'velocity_kmh': 130
        },
        'HT100': {
            # Hilly Terrain, 100 km/h
            'delays': np.array([0.0, 0.1, 0.3, 0.5, 15.0, 17.2]) * 1e-6,
            'powers_db': np.array([0.0, -1.5, -4.5, -7.5, -8.0, -17.7]),
            'velocity_kmh': 100
        },
        'EQ50': {
            # Equalization, 50 km/h
            'delays': np.array([0.0, 3.2, 6.4, 9.6, 12.8, 16.0]) * 1e-6,
            'powers_db': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'velocity_kmh': 50
        },
    }

    
    if channel_model not in models:
        raise ValueError(f"Неизвестная модель канала: {channel_model}. Доступны: {list(models.keys())}")
    
    return models[channel_model]


def _generate_cir(delays, powers, carrier_freq, max_doppler, sampling_rate,
                  num_time_steps, num_tx_ant, num_rx_ant, random_seed):
    """
    Внутренняя функция для генерации импульсной характеристики.
    """
    batch_size = 1
    num_rx_groups = 1 
    num_tx_groups = 1
    num_paths = len(delays)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Временная ось
    time_samples = np.arange(num_time_steps) / sampling_rate
    
    # Инициализация массива коэффициентов
    # Shape: [batch, num_rx_groups, num_rx_ant, num_tx_groups, num_tx_ant, num_paths, num_time_steps]
    a = np.zeros([batch_size, num_rx_groups, num_rx_ant, num_tx_groups, num_tx_ant,
                  num_paths, num_time_steps], dtype=np.complex64)
    
    # Генерация уникальных федингов для каждой пары антенн (Tx -> Rx)
    for r in range(num_rx_ant):
        for t in range(num_tx_ant):
            for path_idx in range(num_paths):
                # Генерация Rayleigh fading с Doppler-эффектом
                doppler_shift = max_doppler * (2 * np.random.rand() - 1)
                phase = 2 * np.pi * doppler_shift * time_samples
                
                # I и Q компоненты (Rayleigh fading)
                i_comp = np.random.randn(num_time_steps)
                q_comp = np.random.randn(num_time_steps)
                
                # Комплексная огибающая с нормализацией по мощности
                tap_response = (i_comp + 1j * q_comp) * np.exp(1j * phase)
                tap_response *= np.sqrt(powers[path_idx] / 2)
                
                # Записываем в массив (индексы групп жестко заданы 0, т.к. num_rx_groups=1)
                a[0, 0, r, 0, t, path_idx, :] = tap_response
    
    # Задержки (одинаковые для всех пар антенн в этой модели)
    tau = np.zeros([batch_size, num_rx_groups, num_rx_ant, num_tx_groups, num_tx_ant, num_paths], dtype=np.float32)
    for r in range(num_rx_ant):
        for t in range(num_tx_ant):
            tau[0, 0, r, 0, t, :] = delays
    
    return a, tau