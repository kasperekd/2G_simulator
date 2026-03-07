import numpy as np
import matplotlib.pyplot as plt
import math

RICE = "RICE"
JAKES = "JAKES"
GAUSS1 = "GAUSS1"
GAUSS2 = "GAUSS2"

def generate_cir(
    channel_model='TU50',
    channel_taps=6,
    carrier_frequency=900e6,
    sampling_rate=1083333.33,
    num_time_steps=100,
    num_tx_ant=1,
    num_rx_ant=1,
    random_seed=42
):
    """
    Генерирует импульсную характеристику канала (CIR) для заданной модели.
    """
    
    channel_name = channel_model[:2]
    velocity_kmh = float(channel_model[2:])
    # Определение параметров модели канала
    channel_params = _get_channel_parameters(channel_name, channel_taps)
    delays = channel_params['delays']
    powers_db = channel_params['powers_db']
    doppler_category = channel_params['doppler_category']
    
    powers_linear = 10**(powers_db / 10.0)
    print(f'powers_liner = {powers_linear}\n')
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
        doppler_category=doppler_category,
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


def _get_channel_parameters(channel_name, channel_taps):
    """
    Возвращает параметры задержек и мощностей для заданной модели канала.
    """
    model_by_taps = {
        6 :{
            'RA': {
                # Rural Area, 130 km/h
                'delays': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) * 1e-6,
                'powers_db': np.array([0.0, -4.0, -8.0, -12.0, -16.0, -20.0]),
                'doppler_category': np.array([RICE, JAKES, JAKES, JAKES, JAKES, JAKES])
            },
            'TU': {
                # Typical Urban, 50 km/h
                'delays': np.array([0.0, 0.2, 0.4, 1.6, 2.4, 5.0]) * 1e-6,
                'powers_db': np.array([-3.0, 0.0, -2.0, -6.0, -8.0, -10.0]),
                'doppler_category': np.array([JAKES, JAKES, GAUSS1, GAUSS1, GAUSS2, GAUSS2])
            },
            'HT': {
                # Hilly Terrain, 100 km/h
                'delays': np.array([0.0, 0.2, 0.4, 0.6, 15.0, 17.2]) * 1e-6,
                'powers_db': np.array([0.0, -2.0, -4.0, -7.0, -6.0, -12.0]),
                'doppler_category': np.array([JAKES, JAKES, JAKES, JAKES, GAUSS2, GAUSS2])
            },
            'EQ': {
                # Equalization, 50 km/h
                'delays': np.array([0.0, 3.2, 6.4, 9.6, 12.8, 16.0]) * 1e-6,
                'powers_db': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                'doppler_category': np.array([JAKES, JAKES, JAKES, JAKES, JAKES, JAKES])
            },
        },

        12 :{
            'TU': {
                # Typical Urban, 50 km/h
                'delays': np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.8, 2.4, 3.0, 3.2, 5.0]) * 1e-6,
                'powers_db': np.array([-4.0, -3.0, 0.0, -2.0, -3.0, -5.0, -7.0, -5.0, -6.0, -9.0, -11.0, -10.0]),
                'doppler_category': np.array([JAKES, JAKES, JAKES, GAUSS1, GAUSS1, GAUSS1, GAUSS1, GAUSS1, GAUSS2, GAUSS2, GAUSS2, GAUSS2])
            },
            'HT': {
                # Hilly Terrain, 100 km/h
                'delays': np.array([0.0, 0.2, 0.4, 0.6, 0.8, 2.0, 2.4, 15.0, 15.2, 15.8, 17.2, 20.0]) * 1e-6,
                'powers_db': np.array([-10.0, -8.0, -6.0, -4.0, 0.0, 0.0, -4.0, -8.0, -9.0, -10.0, -12.0, -14.0]),
                'doppler_category': np.array([JAKES, JAKES, JAKES, GAUSS1, GAUSS1, GAUSS1, GAUSS2, GAUSS2, GAUSS2, GAUSS2, GAUSS2, GAUSS2])
            },
            'EQ': {
                # Equalization, 50 km/h
                'delays': np.array([0.0, 3.2, 6.4, 9.6, 12.8, 16.0]) * 1e-6,
                'powers_db': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                'doppler_category': np.array([JAKES, JAKES, JAKES, JAKES, JAKES, JAKES, JAKES, JAKES, JAKES, JAKES, JAKES, JAKES])
            },
        }
    }

    if channel_name not in model_by_taps[channel_taps]:
        raise ValueError(f"Неизвестная модель канала: {channel_name}. Доступны: {list(model_by_taps[channel_taps].keys())}")
    
    return model_by_taps[channel_taps][channel_name]

def jakes_fading(num_time_steps, sampling_rate, f_max, num_sinusoids):
    t = np.arange(num_time_steps) / sampling_rate
    n = np.arange(1, num_sinusoids + 1)

    alpha_n = np.pi * n / (num_sinusoids + 1)
    phi_n = np.random.uniform(0, 2*np.pi, num_sinusoids)

    fading = np.zeros(num_time_steps, dtype=np.complex64)

    for k in range(num_sinusoids):
        fading += np.exp(
            1j * (2*np.pi*f_max*np.cos(alpha_n[k]) * t + phi_n[k])
        )

    fading /= np.sqrt(num_sinusoids)
    return fading


def gaussian_fading(num_time_steps, sampling_rate, sigma_f, num_sinusoids):
    t = np.arange(num_time_steps) / sampling_rate

    f_n = np.random.normal(0.0, sigma_f, num_sinusoids)
    phi_n = np.random.uniform(0, 2*np.pi, num_sinusoids)

    fading = np.sum(
        np.exp(1j * (2*np.pi*f_n[:, None] * t[None, :] + phi_n[:, None])),
        axis=0
    )

    fading /= np.sqrt(num_sinusoids)
    return fading

def _generate_cir(
    delays,
    powers,
    doppler_category,
    carrier_freq,
    max_doppler,
    sampling_rate,
    num_time_steps,
    num_tx_ant,
    num_rx_ant,
    random_seed=None
):
    """
    COST 207 channel impulse response generator (WSSUS).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    num_paths = len(delays)
    num_sinusoids = 32

    a = np.zeros(
        [1, 1, num_rx_ant, 1, num_tx_ant, num_paths, num_time_steps],
        dtype=np.complex64
    )

    tau = np.zeros(
        [1, 1, num_rx_ant, 1, num_tx_ant, num_paths],
        dtype=np.float32
    )

    t = np.arange(num_time_steps) / sampling_rate

    for r in range(num_rx_ant):
        for tx in range(num_tx_ant):
            for l in range(num_paths):

                if doppler_category[l] == JAKES:
                    fading = jakes_fading(
                        num_time_steps,
                        sampling_rate,
                        max_doppler,
                        num_sinusoids
                    )

                elif doppler_category[l] in (GAUSS1, GAUSS2):
                    # COST 207: Gaussian PSD — разная ширина
                    sigma_f = (
                        0.1 * max_doppler if doppler_category[l] == GAUSS1
                        else 0.3 * max_doppler
                    )
                    fading = gaussian_fading(
                        num_time_steps,
                        sampling_rate,
                        sigma_f,
                        num_sinusoids
                    )

                elif doppler_category[l] == RICE:
                    # Типовой K-фактор (можно параметризовать)
                    K = 6.0
                    los = np.exp(1j * 2*np.pi*max_doppler*t)
                    rayleigh = jakes_fading(
                        num_time_steps,
                        sampling_rate,
                        max_doppler,
                        num_sinusoids
                    )
                    fading = (
                        np.sqrt(K/(K+1)) * los +
                        np.sqrt(1/(K+1)) * rayleigh
                    )

                else:
                    raise ValueError("Unknown Doppler category")

                # Масштабирование по мощности луча
                a[0, 0, r, 0, tx, l, :] = fading * np.sqrt(powers[l])
                tau[0, 0, r, 0, tx, l] = delays[l]

    return a, tau
