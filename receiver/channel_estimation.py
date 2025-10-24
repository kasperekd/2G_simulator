import numpy as np
def estimate_channel_ls(received_ts, known_ts, L):
    len_ts = len(known_ts)
    if len(received_ts) < len_ts:
        received_ts = np.pad(received_ts, (0, len_ts - len(received_ts)))
    S = np.zeros((len_ts, L), dtype=complex)
    for i in range(len_ts):
        for j in range(L):
            if i - j >= 0:
                S[i, j] = known_ts[i - j]
    r = received_ts[:len_ts]
    try:
        h_est = np.linalg.inv(S.conj().T @ S) @ S.conj().T @ r
    except np.linalg.LinAlgError:
        h_est = np.linalg.pinv(S) @ r
    return h_est