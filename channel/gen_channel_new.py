import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.special import j0

def generate_cir(
    channel_model='TU50',
    channel_taps=6,
    frequency_c=1800e6,
    frequency_s=1083.3333e3,
    num_time_steps=50000,
    seed=111,
    num_sinusoids=256,
    rice_k_factor=10.0
):
    """
    COST207 channel с поддержкой RICE для RAxxx моделей (первый луч).
    Возвращает:
        H  : (Nt, L) комплексная TDL-матрица
        tau: (L,) задержки в секундах
    """
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

    # 4. Временная ось
    Ts = 1.0 / frequency_s
    t = np.arange(num_time_steps) * Ts

    # 5. Jakes углы
    n = np.arange(1, num_sinusoids + 1)
    theta = np.pi * (n - 0.5) / num_sinusoids

    H = np.zeros((num_time_steps, num_paths), dtype=complex)

    for path_idx in range(num_paths):
        np.random.seed(seed + path_idx)
        # Угловой сдвиг для каждого луча (WSSUS)
        theta_offset = np.random.uniform(0, 2*np.pi)
        theta_l = theta + theta_offset
        
        # Доплер частоты для каждого луча
        f_ln = frequency_doppler * np.cos(theta_l)
        
        # Случайные фазы
        phi = np.random.uniform(0, 2*np.pi, size=num_sinusoids)
        
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

            los_power = rice_k_factor_liner / (rice_k_factor_liner + 1)
            scattered_power = 1 / (rice_k_factor_liner + 1)

            scattered_component *= np.sqrt(scattered_power)

            los_amplitude = np.sqrt(los_power)
            los_phase = np.random.uniform(0, 2*np.pi)

            # LOS угол
            los_angle = np.random.uniform(0, 2*np.pi)
            f_los = frequency_doppler * np.cos(los_angle)

            los_component = los_amplitude * np.exp(
                1j * (2*np.pi * f_los * t + los_phase)
            )

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

def _generate_single_cir(args):
    idx, current_seed, gen_kwargs = args
    
    kwargs = gen_kwargs.copy()
    kwargs["seed"] = current_seed 
    
    return generate_cir(**kwargs)


def generate_multiple_cir(
    num_realizations,
    num_processes=4,
    base_seed=111,
    **generate_cir_kwargs
):
    print("=" * 80)
    print("GENERATE CIR in progress")
    print("=" * 80)

    # Подготавливаем задачи: для каждой итерации считаем seed со сдвигом 20
    tasks = [
        (idx, base_seed + idx * 40, generate_cir_kwargs)
        for idx in range(num_realizations)
    ]

    with Pool(num_processes) as pool:
        results = pool.map(_generate_single_cir, tasks)

    H_list = [r[0] for r in results]
    tau = results[0][1]

    return H_list, tau

def compute_acf_stable(x):
    x = x - np.mean(x)
    res = np.correlate(x, x, mode='full')
    res = res[res.size // 2:]
    return res / res[0]

# ================================================================================
# Построение графиков, проверка работоспособности
# ================================================================================
# if __name__ == "__main__":
#     channel_model = "TU50"
#     frequency = 1800e6
#     velocity_kmh = float(channel_model[2:]) 
#     v_ms = velocity_kmh / 3.6
#     c = 3e8
#     fd = (v_ms * frequency) / c  
    
#     frequency_s = 1083.333e3
#     dt = 1 / frequency_s  
    
#     sin_values = np.arange(8, 1024, 8) 
#     mse_results = []

#     for sin in sin_values:
#         print(f"sin = {sin}")
#         H, tau = generate_cir(
#             channel_model=channel_model,
#             channel_taps=6,
#             frequency_c=frequency,
#             frequency_s=frequency_s,
#             random_seed=42,
#             num_sinusoids=sin
#         )
        
#         h_first_tap = H[:, 0]
#         n = len(h_first_tap)
#         # Ограничиваем лаги, чтобы оценка ACF была стабильной (например, до n/10)
#         max_lag = int(n / 10)
#         lags = np.arange(0, max_lag)
        
#         # Вычисляем выборочную ACF
#         acf = []
#         for lag in lags:
#             if lag == 0:
#                 corr = np.mean(np.abs(h_first_tap)**2)
#             else:
#                 corr = np.mean(h_first_tap[lag:] * np.conj(h_first_tap[:-lag]))
#             acf.append(corr)
        
#         # Нормируем и берем вещественную часть
#         acf_sim = np.real(np.array(acf) / acf[0])
        
#         # Вычисляем теоретическую ACF для тех же моментов времени
#         t_lags = lags * dt
#         acf_theory = j0(2 * np.pi * fd * t_lags)
        
#         # Расчет MSE (Mean Squared Error)
#         mse = np.mean((acf_sim - acf_theory)**2)
#         mse_results.append(mse)

#     # Построение графика MSE
#     plt.figure(figsize=(10, 6))
#     plt.plot(sin_values, mse_results, 'o-', linewidth=2, markersize=8)
    
#     plt.yscale('log') # Логарифмическая шкала часто нагляднее для MSE
#     plt.title(f'MSE of ACF vs Number of Sinusoids ({channel_model})')
#     plt.xlabel('Number of Sinusoids')
#     plt.ylabel('Mean Squared Error (MSE)')
#     plt.grid(True, which='both', linestyle='--')
#     plt.show()
 

# def main():

#     # ---- параметры канала ----
#     channel_model = "TU50"
#     channel_taps = 6

#     # ---- параметры сигнала ----
#     frequency_c = 1800e6
#     frequency_s = 1083.3333e3
#     num_time_steps = 50000

#     # ---- параметры генерации ----
#     num_sinusoids = 1024
#     rice_k_factor = 10

#     # ---- параметры параллелизма ----
#     num_realizations = 4      # для MIMO 2x2
#     num_processes = 4

#     print("Generating CIR realizations...\n")

#     # ---------------- генерация CIR ----------------
#     H_list, tau = generate_multiple_cir(
#         num_realizations=num_realizations,
#         num_processes=num_processes,
#         channel_model=channel_model,
#         channel_taps=channel_taps,
#         frequency_c=frequency_c,
#         frequency_s=frequency_s,
#         num_time_steps=num_time_steps,
#         num_sinusoids=num_sinusoids,
#         rice_k_factor=rice_k_factor
#     )

#     print(f"\nGenerated {len(H_list)} CIR realizations")

#     # ---------------- извлечение каналов ----------------
#     channels = extract_cir(H_list)

#     print("\nChannels extracted:\n")

#     for key, value in channels.items():
#         print(f"{key} -> shape {value.shape}")

#     # ---------------- пример доступа ----------------
#     h11 = channels["h11"]

#     print("\nExample channel:")
#     print("h11 shape:", h11.shape)

#     # paths x time
#     num_paths, num_time = h11.shape

#     print(f"paths = {num_paths}")
#     print(f"time samples = {num_time}")


# if __name__ == "__main__":
#     main()



# if __name__ == "__main__":
#     channel_model = "TU50"
#     frequency = 1800e6
#     velocity_kmh = float(channel_model[2:])
#     frequency_s = 1083.333e3
#     rng = np.random.default_rng(42)
#     H, tau = generate_cir(
#         channel_model=channel_model,
#         channel_taps=12,
#         frequency_c=frequency,
#         frequency_s=frequency_s,
#         rng=rng,
#         num_sinusoids=2048*2
#     )
#     print(H.shape)
#     # ГРАФИКИ
#     Nt, L = H.shape
#     Ts = 1 / frequency_s
#     time = np.arange(Nt) * Ts
#     delay = tau * 1e6

#     fig = plt.figure(figsize=(10,7))
#     ax = fig.add_subplot(111, projection='3d')

#     for l in range(L):
#         ax.plot(
#             time,
#             np.ones_like(time) * delay[l],
#             np.abs(H[:, l]),
#             linewidth=1.0
#         )

#     ax.set_xlabel("t / s")
#     ax.set_ylabel("τ / µs")
#     ax.set_zlabel("|h(τ,t)|")
#     ax.set_title("Unit impulse response of channel")

#     # ===== FFT по времени =====
#     fs = frequency_s
#     Nfft = 65536

#     S = np.fft.fftshift(np.fft.fft(H, n=Nfft, axis=0), axes=0)
#     S_power = np.abs(S)**2
#     S_power_dB = 10 * np.log10(S_power + 1e-12)  # dB, защита от нулей

#     freq = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/fs))
#     c = 3e8
#     velocity_kmh = float(channel_model[2:])
#     f_D = (velocity_kmh / 3.6) * (frequency / c)

#     # ===== 3D график =====
#     fig = plt.figure(figsize=(10,7))
#     ax = fig.add_subplot(111, projection='3d')


#     for l in range(L):
#         ax.plot(
#             freq,
#             np.ones_like(freq) * delay[l],
#             S_power_dB[:, l],
#             linewidth=1.0
#         )

#     ax.set_xlabel("f / Hz")
#     ax.set_ylabel("τ / µs")
#     ax.set_zlabel("S(τ,f) [dB]")
#     ax.set_title("Scattering function of channel")


#     Nt, L = H.shape
#     time = np.arange(Nt) / 1e6        # секунды
#     delay = tau * 1e6                 # в микросекунды

#     P = np.abs(H)**2
#     P_dB = 10 * np.log10(P + 1e-12)
#     print("Средняя мощность лучей:")
#     print(np.mean(P, axis=0))

#     T, D = np.meshgrid(time, delay, indexing='ij')

#     fig = plt.figure(figsize=(10,7))
#     ax = fig.add_subplot(111, projection='3d')

#     surf = ax.plot_surface(
#         T,
#         D,
#         P_dB,
#         cmap='viridis'
#     )

#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Delay (µs)")
#     ax.set_zlabel("Power (dB)")
#     ax.set_title(f"3D Power Delay Profile {channel_model}")

#     fig.colorbar(surf, ax=ax, label="Power (dB)")
#     # elev — угол над горизонтом (высота), azim — поворот вокруг оси Z
#     ax.view_init(elev=30, azim=-60) 

#     # ACF
#     from scipy.special import j0
#     # Берём один луч
#     h = H[:, 0]
#     h = h - np.mean(h)

#     acf = compute_acf_stable(h)

#     lags = np.arange(len(acf)) / frequency_s


#     # ----- Теоретический J0 -----
#     c = 3e8
#     f_D = (int(channel_model[2:])/3.6) * frequency / c

#     acf_theory = j0(2*np.pi*f_D*lags)

#     mse = np.mean((np.real(acf[:2000]) - acf_theory[:2000])**2)
#     print("MSE vs J0:", mse)
#     # print(np.var(h))

#     # ----- График -----
#     plt.figure(figsize=(8,5))
#     plt.plot(lags, np.real(acf), label="Empirical ACF")
#     plt.plot(lags, acf_theory, '--', label="J0 theory")
#     plt.xlim(0, 40)
#     plt.xlabel("Lag (s)")
#     plt.ylabel("Normalized ACF")
#     plt.legend()
#     plt.grid()
#     plt.show()