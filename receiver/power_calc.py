"""
Модуль для расчёта мощностей в dBm и линейном масштабе.

Поддерживает:
- Преобразование между dBm и линейной шкалой (ватты)
- Расчёт теплового шума в dBm
- Расчёт мощности приёма с учётом потерь трассы
"""
import numpy as np

# constants
BOLTZMANN_K = 1.380649e-23  # J/K
T0 = 290  # Reference temperature in Kelvin


def dbm_to_watts(dbm: float | np.ndarray) -> float | np.ndarray:
    """Convert dBm to watts."""
    return 10 ** ((dbm - 30) / 10)


def watts_to_dbm(watts: float | np.ndarray) -> float | np.ndarray:
    """Convert watts to dBm."""
    return 10 * np.log10(watts + 1e-20) + 30


def db_to_linear(db: float | np.ndarray) -> float | np.ndarray:
    """Convert dB to linear scale (ratio)."""
    return 10 ** (db / 10)


def calculate_thermal_noise_power_dbm(
    bandwidth_hz: float,
    temperature_k: float = T0,
    noise_figure_db: float = 0.0
) -> float:
    """
    Рассчитать мощность теплового шума в dBm.
    
    Noise Power (dBm) = kTB (dBm) + Noise Figure (dB)
    
    kTB (dBm) = -174 + 10*log10(bandwidth in Hz)
    
    Args:
        bandwidth_hz: Полоса частот в Гц
        temperature_k: Температура в Кельвинах (обычно 290K)
        noise_figure_db: Шум-фактор в дБ
    
    Returns:
        Мощность шума в dBm
    """
    # kTB в линейном масштабе
    noise_power_watts = BOLTZMANN_K * temperature_k * bandwidth_hz
    
    # Перевод в dBm
    noise_power_dbm = watts_to_dbm(noise_power_watts)
    
    # Добавляем шум-фактор
    total_noise_dbm = noise_power_dbm + noise_figure_db
    
    return total_noise_dbm


def calculate_received_power_dbm(
    tx_power_dbm: float,
    tx_antenna_gain_dbi: float = 0.0,
    rx_antenna_gain_dbi: float = 0.0,
    path_loss_db: float = 0.0,
    cable_loss_db: float = 0.0,
    other_losses_db: float = 0.0
) -> float:
    """
    Рассчитать мощность на входе приёмника в dBm.
    
    Rx Power (dBm) = Tx Power (dBm) + Tx Antenna Gain (dBi)
                    + Rx Antenna Gain (dBi) - Path Loss (dB)
                    - Cable Loss (dB) - Other Losses (dB)
    
    Args:
        tx_power_dbm: Мощность передатчика в dBm
        tx_antenna_gain_dbi: Усиление передающей антенны в dBi
        rx_antenna_gain_dbi: Усиление приёмной антенны в dBi
        path_loss_db: Потери трассы в дБ
        cable_loss_db: Потери в фидере в дБ
        other_losses_db: Прочие потери в дБ
    
    Returns:
        Принимаемая мощность в dBm
    """
    rx_power_dbm = (
        tx_power_dbm 
        + tx_antenna_gain_dbi 
        + rx_antenna_gain_dbi 
        - path_loss_db 
        - cable_loss_db 
        - other_losses_db
    )
    return rx_power_dbm


def calculate_snr_from_powers_dbm(
    signal_power_dbm: float,
    noise_power_dbm: float
) -> float:
    """
    Рассчитать SNR в дБ из мощностей в dBm.
    
    SNR (dB) = Signal Power (dBm) - Noise Power (dBm)
    
    Args:
        signal_power_dbm: Мощность сигнала в dBm
        noise_power_dbm: Мощность шума в dBm
    
    Returns:
        SNR в дБ
    """
    return signal_power_dbm - noise_power_dbm


def calculate_snr_from_powers_watts(
    signal_power_watts: float,
    noise_power_watts: float
) -> float:
    """
    Рассчитать SNR в дБ из мощностей в ваттах.
    
    SNR (dB) = 10 * log10(Signal Power / Noise Power)
    
    Args:
        signal_power_watts: Мощность сигнала в ваттах
        noise_power_watts: Мощность шума в ваттах
    
    Returns:
        SNR в дБ
    """
    if noise_power_watts < 1e-20:
        noise_power_watts = 1e-20
    return 10 * np.log10(signal_power_watts / noise_power_watts)


def calculate_ci_from_interference_dbm(
    signal_power_dbm: float,
    interference_power_dbm: float
) -> float:
    """
    Рассчитать CI (Carrier-to-Interference) в дБ из мощностей в dBm.
    
    CI (dB) = Signal Power (dBm) - Interference Power (dBm)
    
    Args:
        signal_power_dbm: Мощность сигнала в dBm
        interference_power_dbm: Мощность интерфера в dBm
    
    Returns:
        CI в дБ
    """
    return signal_power_dbm - interference_power_dbm


def calculate_sinr_from_powers_dbm(
    signal_power_dbm: float,
    interference_power_dbm: float,
    noise_power_dbm: float
) -> float:
    """
    Рассчитать SINR в дБ из мощностей в dBm.
    
    Сначала переводим в ватты, суммируем помеху и шум,
    затем считаем отношение.
    
    SINR (dB) = 10 * log10(Ps / (Pi + Pn))
    
    Args:
        signal_power_dbm: Мощность сигнала в dBm
        interference_power_dbm: Мощность интерфера в dBm
        noise_power_dbm: Мощность шума в dBm
    
    Returns:
        SINR в дБ
    """
    signal_watts = dbm_to_watts(signal_power_dbm)
    interference_watts = dbm_to_watts(interference_power_dbm)
    noise_watts = dbm_to_watts(noise_power_dbm)
    
    total_interference_noise = interference_watts + noise_watts
    
    if total_interference_noise < 1e-20:
        total_interference_noise = 1e-20
    
    sinr_db = 10 * np.log10(signal_watts / total_interference_noise)
    
    return sinr_db


def ratio_to_interference_power_dbm(
    ratio_db: float,
    signal_power_dbm: float,
    noise_power_dbm: float,
    calculation_mode: str
) -> float:
    """
    Конвертировать отношение (CI/SINR/SNR) в абсолютную мощность интерфера/шума в dBm.

    При CI режиме: CI = Ps - Pi → Pi = Ps - CI
    При SINR режиме: SINR = Ps / (Pi + Pn) → решаем относительно Pi
    При SNR режиме: SNR = Ps - Pn → Pn = Ps - SNR (возвращаем мощность шума)

    Args:
        ratio_db: Отношение в dB
        signal_power_dbm: Мощность сигнала в dBm
        noise_power_dbm: Мощность теплового шума в dBm
        calculation_mode: 'CI', 'SINR', или 'SNR'

    Returns:
        Мощность в dBm (интерфера для CI/SINR, шума для SNR)
    """
    if calculation_mode == 'CI':
        # CI = Ps - Pi → Pi = Ps - CI
        interference_power_dbm = signal_power_dbm - ratio_db
        return interference_power_dbm

    elif calculation_mode == 'SINR':
        # SINR = Ps / (Pi + Pn)
        # 10^(SINR/10) = Ps / (Pi + Pn)
        # Pi + Pn = Ps / 10^(SINR/10)
        # Pi = Ps / 10^(SINR/10) - Pn

        sinr_linear = db_to_linear(ratio_db)
        signal_watts = dbm_to_watts(signal_power_dbm)
        noise_watts = dbm_to_watts(noise_power_dbm)

        total_interf_noise_watts = signal_watts / sinr_linear
        interference_watts = total_interf_noise_watts - noise_watts

        if interference_watts < 1e-20:
            interference_watts = 1e-20

        return watts_to_dbm(interference_watts)

    elif calculation_mode == 'SNR':
        # SNR = Ps - Pn → Pn = Ps - SNR
        noise_power_dbm = signal_power_dbm - ratio_db
        return noise_power_dbm

    else:
        raise ValueError(f"Unknown calculation_mode: {calculation_mode}")

