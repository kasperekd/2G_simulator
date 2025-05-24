import math
import random as rd

def generate_bit(count_of_bit):
    """Генерирует случайную последовательность бит (0 и 1).
    Args:
        count_of_bit (int): Количество бит в генерируемой последовательности.
    Returns:
        list[int]: Список, содержащий случайные биты (0 или 1).
    Raises:
        Exception: Если count_of_bit меньше или равен нулю.
    """
    if count_of_bit <= 0:
        raise Exception("Signal_generation - Count of bit for generating<= 0!")

    sequence_of_bit = []
    for _ in range(count_of_bit):
        bit = rd.randint(0,1)
        sequence_of_bit.append(bit)
    return sequence_of_bit