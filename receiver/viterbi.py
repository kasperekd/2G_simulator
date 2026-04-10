import numpy as np
def mlse_viterbi_decode(
    received_symbols, channel_taps, constellation, traceback_depth=15
):
    
    channel_memory = len(channel_taps) - 1
    num_symbols_in_constellation = len(constellation)

    if channel_memory < 0:
        channel_memory = 0

    if channel_memory > 0:
        num_states = num_symbols_in_constellation ** channel_memory
        states = [
            tuple(
                reversed(
                    [
                        (i // (num_symbols_in_constellation ** j))
                        % num_symbols_in_constellation
                        for j in range(channel_memory)
                    ]
                )
            )
            for i in range(num_states)
        ]
    else:
        num_states = 1
        states = [()]

    path_metrics = np.full(num_states, np.inf)
    path_metrics[0] = 0
    path_history = np.zeros((len(received_symbols), num_states, 2), dtype=int)

    for t in range(len(received_symbols)):
        r = received_symbols[t]
        new_metrics = np.full(num_states, np.inf)

        for curr_state_idx, curr_state in enumerate(states):
            if path_metrics[curr_state_idx] == np.inf:
                continue

            for input_idx in range(num_symbols_in_constellation):
                # Calculate expected symbol based on current input and state (past symbols)
                expected = channel_taps[0] * constellation[input_idx]
                for i, sym_idx in enumerate(curr_state):
                    if i + 1 < len(channel_taps):
                        expected += channel_taps[i + 1] * constellation[sym_idx]

                branch_metric = np.abs(r - expected)**2

                # Determine next state
                next_state = (input_idx,) + curr_state[:-1] if channel_memory > 0 else ()
                next_state_idx = states.index(next_state)

                new_metric = path_metrics[curr_state_idx] + branch_metric
                if new_metric < new_metrics[next_state_idx]:
                    new_metrics[next_state_idx] = new_metric
                    path_history[t, next_state_idx] = [curr_state_idx, input_idx]

        path_metrics = new_metrics

    # Traceback
    decoded_indices = []
    # Start from the state with the minimum path metric at the end
    current_state_idx = np.argmin(path_metrics)

    for t in range(len(received_symbols) - 1, -1, -1):
        prev_state_idx, input_idx = path_history[t, current_state_idx]
        decoded_indices.append(input_idx)
        current_state_idx = prev_state_idx
        if t > 0 and len(decoded_indices) >= traceback_depth:
            # A simplified traceback approach for streaming data simulation
            # For a block-based simulation, full traceback is better
            pass

    return np.array(list(reversed(decoded_indices)))
