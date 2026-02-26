import numpy as np
from core.irc_combining import estimate_noise_covariance_improved, adaptive_diagonal_loading

def form_spacetime_vector(signals, M=1):
    """
    Forms a Space-Time vector by stacking delayed versions of signals.
    
    Args:
        signals: List of arrays [ant1, ant2]
        M: Number of temporal delays (M=1 means current + 1 delay)
        
    Returns:
        st_signals: List of arrays [ant1_d0, ant2_d0, ant1_d1, ant2_d1, ...]
    """
    st_signals = []
    
    # Original signals (Delay 0)
    st_signals.extend(signals)
    
    # Delayed signals
    for m in range(1, M + 1):
        for sig in signals:
            # Shift signal right by m samples (pad with 0 at start)
            delayed_sig = np.concatenate([np.zeros(m, dtype=complex), sig[:-m]])
            st_signals.append(delayed_sig)
            
    return st_signals

def form_spacetime_channels(channels, M=1):
    """
    Forms effective Space-Time channels corresponding to the ST signals.
    
    If signal x(t) has channel h, then x(t-1) has effective channel z^-1 * h
    (which is h shifted by 1 sample: [0, h0, h1...]).
    """
    st_channels = []
    L = len(channels[0])
    
    # Max length needed for the most delayed channel
    # Original: L. Delayed by M: L + M
    max_len = L + M
    
    # 1. Add original channels
    for h in channels:
        h_padded = np.pad(h, (0, M), mode='constant')
        st_channels.append(h_padded)
        
    # 2. Add delayed channels
    for m in range(1, M + 1):
        for h in channels:
            # Shift h right by m samples
            # Pad start with m zeros, end with M-m zeros
            h_delayed = np.pad(h, (m, M - m), mode='constant')
            st_channels.append(h_delayed)
            
    return st_channels

def st_irc_process(
    received_signals_antennas,
    channel_estimates_antennas,
    known_training_sequence,
    M_taps=1,
    shrinkageMethod='oas',
    loading_factor=0.2
):
    """
    Space-Time Interference Rejection Combining (ST-IRC).
    
    Implements the architecture where the receiver processes a space-time vector
    r_st = [r(t), r(t-1)...]^T to exploit temporal correlation of interference.
    
    Args:
        M_taps: Number of delay elements (1 means we use t and t-1)
    """
    # 1. Expand observations to Space-Time domain
    st_signals = form_spacetime_vector(received_signals_antennas, M_taps)
    st_channels = form_spacetime_channels(channel_estimates_antennas, M_taps)
    
    N_virtual = len(st_signals) # 2 * (M+1)
    L_extended = len(st_channels[0])
    
    # 2. Estimate Space-Time Covariance Matrix (Q_st)
    Q_st = estimate_noise_covariance_improved(
        st_signals,
        known_training_sequence,
        st_channels,
        shrinkageMethod=shrinkageMethod,
        shrinkageFactor=0.1
    )
    
    # 3. Regularization
    # 4x4 matrix with 26 samples is prone to instability
    avg_noise_power = np.trace(Q_st).real / N_virtual
    Q_reg = (1 - loading_factor) * Q_st + loading_factor * avg_noise_power * np.eye(N_virtual, dtype=complex)
    
    try:
        Q_inv = np.linalg.inv(Q_reg)
    except np.linalg.LinAlgError:
        Q_inv = np.linalg.pinv(Q_reg)
        
    # 4. Calculate ST-IRC Filters (Whitening Filters)
    # w = Q_inv * h_matched
    # This creates a vector of weights for all virtual antennas
    st_filters = []
    for i in range(N_virtual):
        # Filter for i-th virtual antenna
        g_i = np.zeros(L_extended, dtype=complex)
        for j in range(N_virtual):
            v_ij = Q_inv[i, j]
            # Matched filter for j-th virtual channel
            # Note: st_channels[j] is already time-aligned correctly
            h_j_matched = st_channels[j][::-1].conj()
            g_i += v_ij * h_j_matched
        st_filters.append(g_i)
        
    # 5. Spatial-Temporal Filtering
    # Output z = sum( filter_i * signal_i )
    signal_len = len(received_signals_antennas[0])
    combined_signal = np.zeros(signal_len, dtype=complex)
    
    for i in range(N_virtual):
        filtered = np.convolve(st_signals[i], st_filters[i], mode='same')
        combined_signal += filtered
        
    # 6. Effective Channel
    # H_eff = sum( filter_i * H_i )
    
    # Convolution length will increase: L_extended + L_filter - 1
    # st_filters length is L_extended
    h_eff_len = 2 * L_extended - 1
    effective_channel_full = np.zeros(h_eff_len, dtype=complex)
    
    for i in range(N_virtual):
        # We convolve the channel of the i-th virtual antenna with its filter
        res = np.convolve(st_channels[i], st_filters[i], mode='full')
        effective_channel_full += res
        
    # 7. Trim Effective Channel for Viterbi
    # The energy is concentrated in the middle
    center = h_eff_len // 2
    start_idx = center - (L_extended // 2)
    effective_channel = effective_channel_full[start_idx : start_idx + L_extended]
    
    return combined_signal, effective_channel