import numpy as np
# from receiver.utils import dec2base, base2dec

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
        if len(seq) == Lh - 1:
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
        # print(f'search_vector:\n {search_vector}')
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

    # np.set_printoptions(threshold=np.inf)

    A = np.array([1, -1])
    Lh = 3  # Память канала, даёт 8 состояний (2^(Lh+1))
    M = 2**Lh
    Rhh = np.array([1.0, 0.9, 0.8, 0.7, 0.6])#([1.0, 0.5, 0.25, 0.125, 0.0625])

    # 1. Создаем все символы (состояния)
    symbols = make_symbols(Lh)
    # symbols = np.array([1,-1]).reshape(-1,1)
    print("All states (symbols):")
    print(symbols)

    # 2. Определяем стартовое состояние (обратите внимание: в make_start возвращается индекс +1)
    # start = make_start(Lh, symbols)
    # print("\nStart state index (0-based):", start)
    start = 0

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

    current_state = 0  # Текущее состояние начинается со стартового
    decoded_sequence = []  # Список для хранения декодированных символов
    
    increment = make_increment(symbols, next_states, Rhh) # Матрица метрик переходов по состояниям
    print(f'increment:\n {increment}')

    '''Заготовка входных состояний'''

    surv = np.full((M**Lh, len(y)), -np.inf)
    metric = np.zeros((M**Lh, len(y)))

    # Задали переход из t = 0 в t = 1
    ss = next_states[current_state, 0]
    metric[ss, 0] =  np.real(np.conj(symbols[ss, 0]) * y[0] - increment[current_state, ss])
    surv[ss, 0] = current_state

    ss = next_states[current_state, 1]
    metric[ss, 0] =  np.real(np.conj(symbols[ss, 0]) * y[0] - increment[current_state, ss])
    surv[ss, 0] = current_state
    T = len(y)  # длина входного сигнала
    num_states = symbols.shape[0]

    for t in range(1, T):
        for curr_state in range(num_states):
            prev_states = previous_states[curr_state]
            metrics_candidates = []
            
            for prev_state in prev_states:
                # Вычисляем суммарную метрику:
                # предыдущее накопленное значение + метрика перехода от prev_state к curr_state + влияние сигнала y[t]
                increment_metric = increment[prev_state, curr_state]
                metric_value = metric[prev_state, t-1] + np.real(np.conj(symbols[curr_state, 0]) * y[t] - increment_metric)
                metrics_candidates.append((metric_value, prev_state))
            
            # Выбираем максимальную метрику и запоминаем состояние-предшественник
            best_metric, best_prev_state = max(metrics_candidates, key=lambda x: x[0])
            metric[curr_state, t] = best_metric
            surv[curr_state, t] = best_prev_state

    # Восстановление лучшего пути (декодированной последовательности)
    best_final_state = np.argmax(metric[:, T-1])
    decoded_states = [best_final_state]
    for t in range(T-1, 0, -1):
        best_final_state = int(surv[best_final_state, t])
        decoded_states.append(best_final_state)
    decoded_states.reverse()

    decoded_symbols = [symbols[state, 0] for state in decoded_states]
    print("Decoded symbol sequence:", decoded_symbols)