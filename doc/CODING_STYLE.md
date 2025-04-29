# Стиль кодирования

## Именование
- Переменные: `snake_case` (`bitstream`, `sampling_rate`).
- Классы: `CamelCase` (`QudrigaImporter`, `MLSEDemodulator`).
- Константы: `UPPER_SNAKE_CASE` (`DEFAULT_SNR`, `MAX_ITERATIONS`).

## Форматирование
- Отступы: 4 пробела.
- Длина строки: <= 80 - 130 символов.

## Комментарии
- Docstrings по стандарту Google:
  ```python
  def calculate_ber(self):
      """Вычисляет Bit Error Rate (BER).
      
      Returns:
          float: Доля ошибочных битов.
      """
  ```
- Только для сложных участков: пояснение логики (например, детали свертки с CIR).