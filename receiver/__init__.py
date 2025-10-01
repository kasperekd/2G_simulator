import numpy as np

def make_symbols(Lh: int) -> np.ndarray:
    """
    Generate all possible symbol sequences of length Lh + 1 for BPSK modulation.

    This function constructs all sequences of +1 and -1 of length Lh+1
    using a recursive backtracking approach. It is useful for building 
    the state symbol table for channels with memory of length Lh.

    Args:
        Lh (int): Length of the channel memory (impulse response length).

    Returns:
        np.ndarray: A 2D array of shape (2^(Lh+1), Lh+1) containing all possible
                    sequences of +1 and -1 representing channel states.
    
    Example:
        >>> make_symbols(2)
        array([[ 1,  1,  1],
               [ 1,  1, -1],
               [ 1, -1,  1],
               [ 1, -1, -1],
               [-1,  1,  1],
               [-1,  1, -1],
               [-1, -1,  1],
               [-1, -1, -1]])
    """
    result = []
    def backtrack(seq):
        if len(seq) == Lh + 1:
            result.append(seq.copy())
            return
        for bit in [1, -1]:
            seq.append(bit)
            backtrack(seq)
            seq.pop()

    backtrack([])
    return np.array(result)

def make_start(Lh: int, symbols: np.array) -> int:
    """
    Find the start state number corresponding to the given channel memory length Lh.

    This function uses a predefined start symbol sequence for the given Lh,
    then searches the SYMBOLS table for the corresponding state number that
    matches this sequence.

    Args:
        Lh (int): Length of the estimated channel impulse response (memory).
        symbols (np.array): A 2D array where each row represents a state as a symbol sequence.

    Returns:
        int: The index (1-based) of the start state in SYMBOLS that matches the start symbol sequence.

    Raises:
        ValueError: If the given Lh does not have a predefined start symbol sequence.

    Example:
        >>> symbols = np.array([[1, -1, 1], [1, -1, -1], [-1, 1, 1]])
        >>> make_start(3, symbols)
        1
    """
    start_not_found = 1
    if Lh == 1:
        start_symbols = [1]
    elif Lh == 2:
        start_symbols = [1, -1]
    elif Lh == 3:
        start_symbols = [1, -1, 1]
    elif Lh == 4:
        start_symbols = [1, -1, 1, -1]
    else:
        return ValueError(f"there are no states for a given Lh={Lh}")

    start = -1
    while start_not_found:
        start += 1
        if sum(symbols[start, :Lh] == start_symbols) == Lh:
            start_not_found = 0

    return start

def make_stop(Lh: int, symbols: np.array) -> list[int]:
    """
    Identify the stop state indices corresponding to the given channel memory length Lh.

    This function uses predefined stop symbol sequences for the given Lh,
    then searches the SYMBOLS table for states matching these sequences.
    It returns the indices of all matching stop states.

    Args:
        Lh (int): Length of the estimated channel impulse response (memory).
        symbols (np.array): A 2D array where each row represents a state as a symbol sequence.

    Returns:
        list[int]: A list of indices (0-based) of stop states matching the predefined stop symbols.

    Raises:
        ValueError: If the given Lh does not have predefined stop symbols,
                    or if not all stop states are found in SYMBOLS.

    Example:
        >>> symbols = np.array([[ -1, 1, -1, 1], [-1, 1, 1, 1], [1, -1, 1, -1]])
        >>> make_stop(4, symbols)
        [0, 1]
    """
    if Lh == 1:
        stop_symbols = [-1]
        count = 1
    elif Lh == 2:
        stop_symbols = [-1, 1]
        count = 1
    elif Lh == 3:
        stop_symbols = [-1, 1, -1]
        count = 1
    elif Lh == 4:
        stop_symbols = [[-1, 1, -1, 1],[-1, 1, 1, 1]]
        count = 2
    else:
        return ValueError(f" Illegal value of Lh = {Lh}, terminating...")
    stop = []
    index = 0
    stops_found = 0
    while stops_found < count and index < len(symbols):
        target = stop_symbols[stops_found] if count > 1 else stop_symbols
        if np.array_equal(symbols[index], target):
            stop.append(index)
            stops_found += 1
        index += 1
    return stop

def make_next(symbols: np.array) -> list[list[int]]:
    """
    Create a lookup table mapping each present state to its legal next states.

    This function builds a table where each row corresponds to a current state,
    and the two legal next states for that state are stored in columns 0 and 1.
    States are represented by their indices in the symbols array.

    Args:
        symbols (np.array): 2D array of symbol sequences representing states,
                            shape (num_states, sequence_length).

    Returns:
        list[list[int]]: A list where each element is a list of two indices representing
                         the next legal states for the corresponding current state.

    Raises:
        ValueError: If more than two next states are identified for any given state.

    Example:
        >>> symbols = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]])
        >>> make_next(symbols)
        [[1, 2], [3, 0], [1, 2], [3, 0]]
    """
    states, maxsum = symbols.shape # states - строки, maxsum - столбец
    # print(f'states and maxsum = {states} and {maxsum}')
    next_s = np.zeros((states, 2), dtype=int)
    search_matrix = symbols[:, 1:maxsum]
    # print(f'serch_matrix:\n {search_matrix}')
    maxsum -= 1
    for this_state in range(states):
        search_vector = symbols[this_state, :maxsum]
        print(f'search_vector:\n {search_vector}')
        k = -1
        for search in range(states):
            if (sum(search_matrix[search,:]==search_vector)==maxsum):
                k+=1
                if k > 2:
                    return ValueError("identified too many next states")
                next_s[this_state,k] = search
    return next_s

def make_previous(symbols: np.array) -> list[list[int]]:
    """
    Create a lookup table mapping each present state to its legal previous states.

    This function builds a table where each row corresponds to a current state,
    and the two legal previous states for that state are stored in columns 0 and 1.
    States are represented by their indices in the symbols array.

    Args:
        symbols (np.array): 2D array of symbol sequences representing states,
                            shape (num_states, sequence_length).

    Returns:
        list[list[int]]: A list where each element is a list of two indices representing
                         the previous legal states for the corresponding current state.

    Raises:
        ValueError: If more than two previous states are identified for any given state.

    Example:
        >>> symbols = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]])
        >>> make_previous(symbols)
        [[2, 3], [0, 1], [2, 3], [0, 1]]
    """
    states, maxsum = symbols.shape
    previous_s = np.zeros((states, 2), dtype=int)
    maxsum -= 1
    search_matrix = symbols[:, :maxsum]
    for this_state in range(states):
        search_vector = symbols[this_state, 1:maxsum + 1]
        k = 0
        for search in range(states):
            if (sum(search_matrix[search,:]==search_vector)==maxsum):
                previous_s[this_state,k] = search
                k += 1
                if k > 2:
                    return ValueError("identified too many next states")
    return previous_s

def make_increment(symbols: np.array, next: np.array, Rhh: np.array):
    """
    Calculate increments linked to state transitions given symbols, next states and channel vector.

    Args:
        symbols (np.array): 2D array (M x Lh+1) symbol sequences for M states.
        next_states (np.array): 2D array (M x 2) next states indices for each state.
        Rhh (np.array): 1D array with channel impulse response coefficients, length Lh.

    Returns:
        np.array: 2D array (M x M) increment values for state transitions.
    """
    M, Lh = symbols.shape
    increment = np.zeros((M, M))
    Rhh_col = Rhh[1:Lh+1].reshape(-1, 1)
    for n in range(M):
        m = next[n, 0]
        increment[n, m] = np.real(np.conj(symbols[m, 0]) * (symbols[n, :] @ Rhh_col))
        m = next[n, 1]
        increment[n, m] = np.real(np.conj(symbols[m, 0]) * (symbols[n, :] @ Rhh_col))
    return increment

'''
    Мини пример использования перехода по ветвям с вычислением ошибки
    пока что без использование матрицы INCREMENT и SURVIVOR путей
'''
if __name__ == '__main__':

    Lh = 3  # Память канала, даёт 8 состояний (2^(Lh+1))
    Rhh = np.array([1.0, 0.9, 0.8, 0.7, 0.6])#([1.0, 0.5, 0.25, 0.125, 0.0625])

    # 1. Создаем все символы (состояния)
    symbols = make_symbols(Lh)
    print("All states (symbols):")
    print(symbols)

    # 2. Определяем стартовое состояние (обратите внимание: в make_start возвращается индекс +1)
    start = make_start(Lh, symbols)
    print("\nStart state index (0-based):", start)

    # 3. Строим таблицу переходов next_states
    next_states = make_next(symbols)
    print("\nNext states lookup table:")
    for i, nxt in enumerate(next_states):
        print(f"State {i}: can go to states {nxt}")

    # 4. Строим таблицу предыдущих состояний previous_states
    previous_states = make_previous(symbols)
    print("\nPrevious states lookup table:")
    for i, prev in enumerate(previous_states):
        print(f"State {i}: can come from states {prev}")

    # --- Добавляем декодирование входного сигнала y с подробным выводом ---
    y = np.array([0.7, -0.2, -0.65, 0.44, 0.63])  # Входной сигнал для декодирования

    current_state = start  # Текущее состояние начинается со стартового
    decoded_sequence = []  # Список для хранения декодированных символов

    print("\nDecoding steps:")
    for i in range(len(y)):
        print(f"\nStep {i+1}:")
        print(f"Current input y[{i}] = {y[i]}")

        print(f"Possible previous states for current state {current_state}: {previous_states[current_state]}")
        print(f"Possible next states from current state {current_state}: {next_states[current_state]}")

        # Получаем два возможных следующих состояния
        next_0, next_1 = next_states[current_state]
        increment = make_increment(symbols, next_states, Rhh)
        print(f'increment:\n {increment}')
        # Вычисляем ошибку для каждого возможного перехода
        error_0 = (y[i] - symbols[next_0][0]) ** 2
        error_1 = (y[i] - symbols[next_1][0]) ** 2

        print(f"Error if go to state {next_0} (symbol {symbols[next_0][0]}): {error_0:.4f}")
        print(f"Error if go to state {next_1} (symbol {symbols[next_1][0]}): {error_1:.4f}")

        # Выбираем следующий переход с минимальной ошибкой
        if error_0 <= error_1:
            chosen_state = next_0
        else:
            chosen_state = next_1

        print(f"Chosen next state: {chosen_state} with symbol {symbols[chosen_state][0]}")

        # Сохраняем символ (первый элемент символов состояния) как выходной
        decoded_sequence.append(symbols[chosen_state][0])

        # Переходим в выбранное состояние
        current_state = chosen_state

    print("\nDecoded symbol sequence from input y:", decoded_sequence)
