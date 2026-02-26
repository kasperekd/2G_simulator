import numpy as np
import matplotlib.pyplot as plt
import math

RICE = "RICE"
CLASS = "CLASS"

def generate_cir(
    channel_model='TU50',
    channel_taps=6,
    frequency_c=1800e6,
    frequency_s=1083.3333e3,
    num_time_steps=50000,
    random_seed=42,
    num_sinusoids=32,
    rice_k_factor=10.0
):
    """
    COST207 channel с поддержкой RICE для RAxxx моделей (первый луч).
    Возвращает:
        H  : (Nt, L) комплексная TDL-матрица
        tau: (L,) задержки в секундах
    """
    print(f"carrier frequency = {frequency_c}",)
    # 1. Разбор модели
    channel_name = channel_model[:2]
    velocity_kmh = float(channel_model[2:])
    channel_params = get_channel_parameters(channel_name, channel_taps)
    powers_db = channel_params['powers_db']
    delays = channel_params['delays']

    # 2. Нормализация мощности
    powers_linear = 10**(powers_db / 10.0)
    total_power = np.sum(powers_linear)
    
    # Проверяем RICE для RAxxx моделей
    rice_k_factor_liner = 10**(rice_k_factor/10)
    is_rice_first_path = (channel_name == 'RA' and rice_k_factor_liner > 0)
    
    num_paths = len(delays)

    # 3. Вычисление Допплера
    c = 3e8
    frequency_doppler = (velocity_kmh / 3.6) * (frequency_c / c)
    print(f"frequency doppler = {frequency_doppler}")

    # 4. Временная ось
    Ts = 1.0 / frequency_s
    t = np.arange(num_time_steps) * Ts

    # 5. Jakes углы
    n = np.arange(1, num_sinusoids + 1)
    theta = np.pi * (n - 0.5) / num_sinusoids

    H = np.zeros((num_time_steps, num_paths), dtype=complex)

    for path_idx in range(num_paths):
        rng = np.random.default_rng(random_seed + path_idx)
        # Угловой сдвиг для каждого луча (WSSUS)
        theta_offset = rng.uniform(0, 2*np.pi)
        theta_l = theta + theta_offset
        
        # Доплер частоты для каждого луча
        f_ln = frequency_doppler * np.cos(theta_l)
        
        # Случайные фазы
        phi = rng.uniform(0, 2*np.pi, size=num_sinusoids)
        
        # Формирование фазовой матрицы
        phase = 2*np.pi * t[:, None] * f_ln[None, :] + phi[None, :]
        
        # Сумма синусоид (Rayleigh компонент)
        c_sin = np.sqrt(2.0 / num_sinusoids)
        u_rayleigh = c_sin * np.exp(1j * phase)  # (Nt, Nsin)
        scattered_component = np.sum(u_rayleigh, axis=1)  # (Nt,)
        
        # Нормализация разбросанной компоненты (сигма^2 = 1)
        sigma_sq = np.mean(np.abs(scattered_component)**2)
        scattered_component /= np.sqrt(sigma_sq)
        
        # RICE для первого луча (если включено)
        if is_rice_first_path and path_idx == 0:
            # LOS компонента: мощность LOS / (LOS + scattered) = K/(K+1)
            los_power = rice_k_factor_liner / (rice_k_factor_liner + 1)
            scattered_power = 1 / (rice_k_factor_liner + 1)
            
            # Нормализуем разбросанную компоненту под её долю мощности
            scattered_component *= np.sqrt(scattered_power)
            
            # LOS компонента (постоянная амплитуда + случайная фаза)
            los_amplitude = np.sqrt(los_power)
            los_phase = rng.uniform(0, 2*np.pi)
            los_component = los_amplitude * np.exp(1j * los_phase)
            
            # Итоговая Rician компонента
            H[:, path_idx] = los_component + scattered_component
        else:
            # Обычный Rayleigh
            H[:, path_idx] = scattered_component
        
        # Масштабирование по мощности пути
        path_power = powers_linear[path_idx] / total_power
        H[:, path_idx] *= np.sqrt(path_power)

    return H, np.array(delays)

def get_channel_parameters(channel_name, channel_taps):
    """
    Возвращает параметры задержек и мощностей для заданной модели канала.
    """
    model_by_taps = {
        4 :{
            'RA': {
                # Rural Area, 130 km/h
                'delays': np.array([0.0, 0.2, 0.4, 0.6]) * 1e-6,
                'powers_db': np.array([0.0, -2.0, -10.0, -20.0]),
            },
        },


        6 :{
            'RA': {
                # Rural Area, 130 km/h
                'delays': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) * 1e-6,
                'powers_db': np.array([0.0, -4.0, -8.0, -12.0, -16.0, -20.0]),
            },
            'TU': {
                # Typical Urban, 50 km/h
                'delays': np.array([0.0, 0.2, 0.4, 1.6, 2.4, 5.0]) * 1e-6,
                'powers_db': np.array([-3.0, 0.0, -2.0, -6.0, -8.0, -10.0]),
            },
            'HT': {
                # Hilly Terrain, 100 km/h
                'delays': np.array([0.0, 0.2, 0.4, 0.6, 15.0, 17.2]) * 1e-6,
                'powers_db': np.array([0.0, -2.0, -4.0, -7.0, -6.0, -12.0]),
            },
            'EQ': {
                # Equalization, 50 km/h
                'delays': np.array([0.0, 3.2, 6.4, 9.6, 12.8, 16.0]) * 1e-6,
                'powers_db': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
        },

        12 :{
            'TU': {
                # Typical Urban, 50 km/h
                'delays': np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.8, 2.4, 3.0, 3.2, 5.0]) * 1e-6,
                'powers_db': np.array([-4.0, -3.0, 0.0, -2.0, -3.0, -5.0, -7.0, -5.0, -6.0, -9.0, -11.0, -10.0]),
            },
            'HT': {
                # Hilly Terrain, 100 km/h
                'delays': np.array([0.0, 0.2, 0.4, 0.6, 0.8, 2.0, 2.4, 15.0, 15.2, 15.8, 17.2, 20.0]) * 1e-6,
                'powers_db': np.array([-10.0, -8.0, -6.0, -4.0, 0.0, 0.0, -4.0, -8.0, -9.0, -10.0, -12.0, -14.0]),
            },
            'EQ': {
                # Equalization, 50 km/h
                'delays': np.array([0.0, 3.2, 6.4, 9.6, 12.8, 16.0]) * 1e-6,
                'powers_db': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
        }
    }

    if channel_name not in model_by_taps[channel_taps]:
        raise ValueError(f"Неизвестная модель канала: {channel_name}. Доступны: {list(model_by_taps[channel_taps].keys())}")
    
    return model_by_taps[channel_taps][channel_name]

def compute_acf_fft(x, unbiased=True):
    x = np.asarray(x)
    N = len(x)
    X = np.fft.fft(x, n=2*N)
    acf = np.fft.ifft(X * np.conj(X))[:N]  # комплексная АКФ

    if unbiased:
        acf /= (N - np.arange(N))          # деление на число пар (N-k)

    acf /= acf[0]                          # нормировка к 1 на нуле
    return acf


if __name__ == "__main__":
    channel_model = "TU50"
    frequency = 1800e6
    velocity_kmh = float(channel_model[2:])
    frequency_s = 1083.333e3
    H, tau = generate_cir(
        channel_model=channel_model,
        channel_taps=6,
        frequency_c=frequency,
        frequency_s=frequency_s,
        random_seed=42,
        num_sinusoids=2048
    )
    print(H.shape)
    # ГРАФИКИ
    Nt, L = H.shape
    Ts = 1 / frequency_s
    time = np.arange(Nt) * Ts
    delay = tau * 1e6

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    for l in range(L):
        ax.plot(
            time,
            np.ones_like(time) * delay[l],
            np.abs(H[:, l]),
            linewidth=1.0
        )

    ax.set_xlabel("t / s")
    ax.set_ylabel("τ / µs")
    ax.set_zlabel("|h(τ,t)|")
    ax.set_title("Unit impulse response of channel")

    # ===== FFT по времени =====
    fs = frequency_s
    Nfft = 65536

    S = np.fft.fftshift(np.fft.fft(H, n=Nfft, axis=0), axes=0)
    S_power = np.abs(S)**2
    S_power_dB = 10 * np.log10(S_power + 1e-12)  # dB, защита от нулей

    freq = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/fs))
    c = 3e8
    velocity_kmh = float(channel_model[2:])
    f_D = (velocity_kmh / 3.6) * (frequency / c)

    # ===== 3D график =====
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')


    for l in range(L):
        ax.plot(
            freq,
            np.ones_like(freq) * delay[l],
            S_power_dB[:, l],
            linewidth=1.0
        )

    ax.set_xlabel("f / Hz")
    ax.set_ylabel("τ / µs")
    ax.set_zlabel("S(τ,f) [dB]")
    ax.set_title("Scattering function of channel")


    Nt, L = H.shape
    time = np.arange(Nt) / 1e6        # секунды
    delay = tau * 1e6                 # в микросекунды

    P = np.abs(H)**2
    P_dB = 10 * np.log10(P + 1e-12)
    print("Средняя мощность лучей:")
    print(np.mean(P, axis=0))

    T, D = np.meshgrid(time, delay, indexing='ij')

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        T,
        D,
        P_dB,
        cmap='viridis'
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Delay (µs)")
    ax.set_zlabel("Power (dB)")
    ax.set_title(f"3D Power Delay Profile {channel_model}")

    fig.colorbar(surf, ax=ax, label="Power (dB)")

    # ACF
    from scipy.special import j0
    # Берём один луч
    h = H[:, 0]
    h = h - np.mean(h)

    acf = compute_acf_fft(h)

    lags = np.arange(len(acf)) / frequency_s


    # ----- Теоретический J0 -----
    c = 3e8
    f_D = (int(channel_model[2:])/3.6) * frequency / c

    acf_theory = j0(2*np.pi*f_D*lags)

    mse = np.mean((np.real(acf[:2000]) - acf_theory[:2000])**2)
    print("MSE vs J0:", mse)
    # print(np.var(h))

    # ----- График -----
    plt.figure(figsize=(8,5))
    plt.plot(lags, np.real(acf), label="Empirical ACF")
    plt.plot(lags, acf_theory, '--', label="J0 theory")
    plt.xlim(0, 40)
    plt.xlabel("Lag (s)")
    plt.ylabel("Normalized ACF")
    plt.legend()
    plt.grid()
    plt.show()