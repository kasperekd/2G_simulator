import numpy as np
from . import utils
from .__init__ import make_symbols, make_increment, make_next, make_previous
'''Версия Саши'''
def mlse_viterbi_decode(y, Rhh, Lh, receiver_constellation):
    # print(y)
    # 1. Создаем все символы (состояния)
    symbols = make_symbols(Lh)
    # symbols = np.array([1,-1]).reshape(-1,1)
    # print("All states (symbols):")
    # print(symbols)
    M = receiver_constellation.shape[0]
    # print(M)
    next_states = make_next(symbols)
    # print("\nNext states lookup table:")
    # for i, nxt in enumerate(next_states):
    #     print(f"State {i}: can go to states {nxt}")

    # 4. Строим таблицу предыдущих состояний previous_states
    previous_states = make_previous(symbols)
    # print("\nPrevious states lookup table:")
    # for i, prev in enumerate(previous_states):
    #     print(f"State {i}: can come from states {prev}")

    # --- Добавляем декодирование входного сигнала y с подробным выводом ---

    current_state = 0  # Текущее состояние начинается со стартового
    
    increment = make_increment(symbols, next_states, Rhh) # Матрица метрик переходов по состояниям
    # print(f'increment:\n {increment}')
    
    '''Заготовка входных состояний'''

    # print(M)
    # print(Lh)
    surv = np.full((M**(Lh-1), len(y)), -np.inf)
    metric = np.zeros((M**(Lh-1), len(y)))
    # print(metric.shape)

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
    # print(f'decoded symbols:\n{np.array(decoded_symbols)}')
    return np.array(decoded_symbols)

'''Версия Кости'''
# def mlse_viterbi_decode(y_hat: np.ndarray, S: np.ndarray, L: int, constellation: np.ndarray) -> np.ndarray:
#     """
#     Generic Viterbi decoder for MLSE, implemented manually.
#     This approach is necessary because MLSE branch metrics are state-dependent,
#     which doesn't fit the standard `commpy` Viterbi decoder for convolutional codes.

#     Args:
#         y_hat (np.ndarray): Input samples from the IRC combiner.
#         S (np.ndarray): Channel autocorrelation (S_0, S_1, ..., S_L-1).
#         L (int): Channel memory length (length of CIR).
#         constellation (np.ndarray): The array of complex symbols for the modulation.

#     Returns:
#         np.ndarray: The estimated sequence of symbol indices (0, 1, ..., M-1).
#     """
#     M = len(constellation)
#     # print(f'M = {M}')
#     num_data_symbols = len(y_hat)
#     num_states = M**(L - 1)
#     # print(f'num states = {num_states}')
#     if num_states > 2**14:
#         raise ValueError(f"Too many states for Viterbi: {num_states}. Reduce M or L.")

#     # --- Trellis Definition ---
#     all_states = utils.dec2base(np.arange(num_states), M, L - 1)
    
#     next_state_table = np.zeros((num_states, M), dtype=int)
#     for state_idx in range(num_states):
#         for symbol_idx in range(M):
#             next_state_symbols = np.concatenate(([symbol_idx], all_states[state_idx, :-1]))
#             next_state_table[state_idx, symbol_idx] = utils.base2dec(next_state_symbols, M)
#     # print(f'next state symbols = {next_state_symbols}')
#     # print(f'next state table =\n{next_state_table}')
#     # --- Viterbi Algorithm ---
#     path_metrics = np.full(num_states, -np.inf)
#     path_metrics[0] = 0.0
    
#     path_memory = np.zeros((num_states, num_data_symbols), dtype=int)

#     # Pre-calculate ISI for all states to speed up the main loop
#     isi_per_state = np.zeros(num_states, dtype=np.complex128)
#     for state_idx in range(num_states):
#         past_symbol_indices = all_states[state_idx]
#         past_complex_symbols = constellation[past_symbol_indices]
#         isi_per_state[state_idx] = np.sum(past_complex_symbols * S[1:L])

#     # Forward pass
#     for t in range(num_data_symbols):
#         new_path_metrics = np.full(num_states, -np.inf)
#         new_path_memory = np.zeros(num_states, dtype=int)
        
#         for current_state_idx in range(num_states):
#             if np.isneginf(path_metrics[current_state_idx]):
#                 continue

#             for input_symbol_idx in range(M):
#                 y_expected = constellation[input_symbol_idx] * S[0] + isi_per_state[current_state_idx]
#                 branch_metric = -np.abs(y_hat[t] - y_expected)**2
#                 path_metric = path_metrics[current_state_idx] + branch_metric
                
#                 next_state_idx = next_state_table[current_state_idx, input_symbol_idx]
                
#                 if path_metric > new_path_metrics[next_state_idx]:
#                     new_path_metrics[next_state_idx] = path_metric
#                     new_path_memory[next_state_idx] = current_state_idx
        
#         path_metrics = new_path_metrics
#         path_memory[:, t] = new_path_memory

#     # --- Traceback ---
#     decoded_indices = np.zeros(num_data_symbols, dtype=int)
#     current_state = np.argmax(path_metrics)
    
#     for t in range(num_data_symbols - 1, -1, -1):
#         previous_state = path_memory[current_state, t]
        
#         # Find the input symbol that caused the transition from previous_state to current_state
#         for input_symbol_idx in range(M):
#             if next_state_table[previous_state, input_symbol_idx] == current_state:
#                 decoded_indices[t] = input_symbol_idx
#                 break
#         current_state = previous_state
        
#     return decoded_indices