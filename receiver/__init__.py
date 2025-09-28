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

    start = 0
    while start_not_found:
        if sum(symbols[start, :] == start_symbols) == Lh:
            start_not_found = 0
        start += 1

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
    next_s = []
    states, maxsum = symbols.shape # states - строки, maxsum - столбец
    search_matrix = symbols[:, 1:maxsum]
    maxsum -= 1
    for this_state in range(states):
        search_vector = symbols[this_state, :maxsum]
        k = 0
        for search in range(states):
            if (sum(search_matrix[search,:]==search_vector)==maxsum):
                next_s[this_state,k] = search
                k+=1
                if k > 2:
                    return ValueError("identified too many next states")
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
    previous_s = []
    states, maxsum = symbols.shape
    maxsum -= 1
    search_matrix = symbols[:, :maxsum]
    for this_state in range(states):
        search_vector = symbols[this_state, 1:maxsum + 1]
        k = 0
        for search in range(states):
            if (sum(search_matrix[search,:]==search_vector)==maxsum):
                k += 1
                previous_s[this_state,k] = search
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
    M, Lh = np.size[symbols]
    increment = np.zeros(M)
    Rhh_col = Rhh[1:Lh].reshape(-1, 1)
    for n in range(M):
        m = next[n, 1]
        increment[n, m] = np.real(np.conj(symbols[m, 1]) @ symbols[n, :] @ Rhh_col)
        m = next[n, 2]
        increment[n, m] = np.real(np.conj(symbols[m, 1]) @ symbols[n, :] @ Rhh_col)
    return increment
