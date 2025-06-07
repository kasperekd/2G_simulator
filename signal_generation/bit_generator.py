import numpy as np
import random as rd

def generate_bit(count_of_bit, number_seed):
    """Генерирует случайную последовательность бит (0 и 1).
    Args:
        count_of_bit (int): Количество бит в генерируемой последовательности.
    Returns:
        list[int]: Список, содержащий случайные биты (0 или 1).
    Raises:
        ValueError: Если count_of_bit меньше или равен нулю.
        ValueError: Если seed_enabled=True, но number_seed не задан.
        TypeError: Если параметры имеют неверный тип.
    """
    if count_of_bit <= 0:
        raise ValueError("Signal_generation - Count of bit for generating<= 0!")

    if number_seed is None:
        raise ValueError("При seed_enabled=True необходимо указать number_seed")
    if not isinstance(number_seed, int):
        raise TypeError("number_seed должен быть целым числом")
        
    # Устанавливаем seed только для numpy
    np.random.seed(number_seed)

    # Генерация битов
    sequence_of_bit = np.random.randint(0, 2, count_of_bit)

    # Если seed был установлен, сбрасываем его
    if number_seed is not None:
        np.random.seed(None)

    return sequence_of_bit