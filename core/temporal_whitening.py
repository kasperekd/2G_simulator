import numpy as np
from scipy.linalg import sqrtm

def estimate_temporal_covariance(
    received_signal,
    channel_estimate,
    known_training_sequence
):
    """Estimate temporal autocorrelation of the residual over the training symbols.

    This computes the sample covariance R_e = E[e * e^H] using the
    residual after removing the expected useful signal on the training
    sequence.

    Args:
        received_signal: received samples (1D array)
        channel_estimate: estimated channel impulse response
        known_training_sequence: training symbols (length M)

    Returns:
        R_e: MxM temporal covariance matrix
        residual: residual vector used to form R_e
    """
    M = len(known_training_sequence)

    expected_signal = np.convolve(
        channel_estimate, 
        known_training_sequence, 
        mode='same'
    )[:M]

    # Received signal on training sequence
    r_ts = received_signal[:M]

    # Residual = interference + noise
    residual = r_ts - expected_signal

    # Temporal autocorrelation matrix: R_e = E[e * e^H]
    # Simple estimate: e * e^H / M
    R_e = (residual[:, None] @ residual[None, :].conj()) / M

    return R_e, residual


def estimate_noise_subspace(eigenvalues, method='mdl'):
    """Estimate the number of signal components from eigenvalues.

    Supports MDL (default) and a simplified AIC fallback.

    Args:
        eigenvalues: sorted eigenvalues (descending)
        method: 'mdl' or 'aic'

    Returns:
        Estimated number of signal eigencomponents (int)
    """
    M = len(eigenvalues)

    if method == 'mdl':
        # MDL criterion
        mdl_values = []
        for k in range(1, M):
            arithmetic_mean = np.mean(eigenvalues[k:])
            geometric_mean = np.exp(np.mean(np.log(eigenvalues[k:] + 1e-10)))

            mdl = (M - k) * M * np.log(arithmetic_mean / geometric_mean) + 0.5 * k * (2*M - k) * np.log(M)
            mdl_values.append(mdl)

        num_signal_components = np.argmin(mdl_values) + 1

    else:  # 'aic'
        # AIC criterion (simplified)
        # Assume last 25% eigenvalues are noise
        num_signal_components = int(0.75 * M)

    return num_signal_components


# ====================================================================
# Whitening Methods
# ====================================================================

def mahalanobis_whitening_matrix(R_e, regularization=1e-6):
    """Compute Mahalanobis whitening matrix W = R_e^{-1/2}.

    Regularization stabilizes small/negative eigenvalues.

    Returns the whitening matrix and the eigenvalues (descending).
    """
    M = R_e.shape[0]

    # Regularization
    R_e_reg = R_e + regularization * np.eye(M)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R_e_reg)

    # Protect against tiny/negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, regularization)

    # Sort in descending order
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Whitening matrix: W = V * Λ^{-1/2} * V^H
    Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    W = eigenvectors @ Lambda_inv_sqrt @ eigenvectors.conj().T

    return W, eigenvalues


def bias_removal_whitening_matrix(R_e, thermal_noise_variance, regularization=1e-10):
    """Bias-removal Mahalanobis whitening (Cichocki-style).

    This method estimates the noise floor and subtracts it from
    eigenvalues before inverting, reducing bias in the whitening
    transform.
    """
    M = R_e.shape[0]

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R_e)

    # Sort eigenvalues descending
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Estimate noise subspace
    num_signal = estimate_noise_subspace(eigenvalues, method='aic')

    # Estimate noise variance if not provided: use mean of presumed noise eigenvalues
    if thermal_noise_variance is None:
        # use mean of the smallest eigenvalues as noise variance estimate
        noise_eigenvalues = eigenvalues[num_signal:]
        thermal_noise_variance = np.mean(noise_eigenvalues)

    # Subtract estimated noise variance from eigenvalues (bias removal)
    eigenvalues_corrected = eigenvalues - thermal_noise_variance

    # Clamp corrected eigenvalues to a safe minimum
    eigenvalues_corrected = np.maximum(
        eigenvalues_corrected, 
        thermal_noise_variance / 10
    )

    # Small regularization after bias removal
    eigenvalues_corrected = eigenvalues_corrected + regularization

    # Build whitening matrix
    Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues_corrected))
    W = eigenvectors @ Lambda_inv_sqrt @ eigenvectors.conj().T

    return W, eigenvalues_corrected


# ====================================================================
# Signal Whitening
# ====================================================================

def apply_temporal_whitening_to_ts(
    training_sequence_signal,
    whitening_matrix
):
    """Apply whitening matrix to the training-sequence portion.

    Args:
        training_sequence_signal: vector of training samples (M)
        whitening_matrix: MxM whitening matrix

    Returns:
        whitened_ts: vector of whitened training samples
    """
    whitened_ts = whitening_matrix @ training_sequence_signal
    return whitened_ts


def extend_whitening_to_full_burst(received_signal, whitening_matrix, ts_start_index=61):
    """Extend whitening to an entire burst using a sliding window.

    The whitening matrix (size M) is applied to every M-length window
    and the central sample of each whitened window is placed into the
    corresponding position in the output burst.
    """
    M = whitening_matrix.shape[0]
    N = len(received_signal)

    whitened_signal = np.zeros(N, dtype=complex)

    # Apply whitening using sliding windows and keep center samples
    for i in range(N - M + 1):
        window = received_signal[i:i+M]
        whitened_window = whitening_matrix @ window

            # keep the central sample of the whitened window
        whitened_signal[i + M//2] = whitened_window[M//2]

    # Fill edges by copying nearest computed values
    for i in range(M//2):
        whitened_signal[i] = whitened_signal[M//2]
        whitened_signal[-(i+1)] = whitened_signal[-(M//2+1)]

    return whitened_signal


# ====================================================================
# Effective Channel
# ====================================================================

def compute_effective_channel_temporal(
    channel_estimate,
    whitening_matrix
):
    """Compute effective channel after temporal whitening.

    Because W is MxM while channel_estimate may be length L, this
    function extracts a central LxL submatrix of W (if M >= L) and
    multiplies it with the channel vector. When L > M, a simplified
    scaling based on the diagonal is applied.
    """
    L = len(channel_estimate)
    M = whitening_matrix.shape[0]

    if M >= L:
            # extract central LxL block from W
        start_idx = (M - L) // 2
        W_center = whitening_matrix[start_idx:start_idx+L, start_idx:start_idx+L]

        # effective channel is W_center * h
        h_effective = W_center @ channel_estimate
    else:
        # For L > M fall back to a scalar scaling defined by diagonal mean
        avg_scaling = np.mean(np.diag(whitening_matrix))
        h_effective = channel_estimate * avg_scaling

    # Normalize to preserve original channel energy
    original_norm = np.linalg.norm(channel_estimate)
    effective_norm = np.linalg.norm(h_effective)

    if effective_norm > 1e-10:
        h_effective = h_effective * (original_norm / effective_norm)

    return h_effective


# ====================================================================
# Full Pipeline Functions
# ====================================================================

def temporal_whitening_mahalanobis(
    received_signal,
    channel_estimate,
    known_training_sequence,
    regularization=1e-6,
    full_burst=True
):
    """Full temporal whitening pipeline (Mahalanobis method).

    Returns whitened signal and effective channel for equalization.
    """

    # Step 1: Estimate temporal covariance
    R_e, residual = estimate_temporal_covariance(
        received_signal,
        channel_estimate,
        known_training_sequence
    )

    # Step 2: Mahalanobis whitening matrix
    W, eigenvalues = mahalanobis_whitening_matrix(
        R_e,
        regularization
    )

    # Step 3: Apply whitening
    if full_burst:
        whitened_signal = extend_whitening_to_full_burst(
            received_signal,
            W,
            ts_start_index=61
        )
    else:
        # Only apply whitening to training symbols
        M = len(known_training_sequence)
        whitened_signal = np.copy(received_signal)
        whitened_signal[:M] = apply_temporal_whitening_to_ts(
            received_signal[:M],
            W
        )

    # Step 4: Effective channel
    effective_channel = compute_effective_channel_temporal(
        channel_estimate,
        W
    )

    return whitened_signal, effective_channel


def temporal_whitening_bias_removal(
    received_signal,
    channel_estimate,
    known_training_sequence,
    thermal_noise_variance=None,
    regularization=1e-10,
    full_burst=True
):
    """Full temporal whitening pipeline (bias-removal method).

    Similar to the Mahalanobis version but subtracts an estimate of
    the thermal noise variance from eigenvalues before inversion.
    """

    # Step 1: Estimate temporal covariance
    R_e, residual = estimate_temporal_covariance(
        received_signal,
        channel_estimate,
        known_training_sequence
    )

    # Step 2: Bias removal whitening matrix
    W, eigenvalues_corrected = bias_removal_whitening_matrix(
        R_e,
        thermal_noise_variance,
        regularization
    )

    # Step 3: Apply whitening
    if full_burst:
        whitened_signal = extend_whitening_to_full_burst(
            received_signal,
            W,
            ts_start_index=61
        )
    else:
        M = len(known_training_sequence)
        whitened_signal = np.copy(received_signal)
        whitened_signal[:M] = apply_temporal_whitening_to_ts(
            received_signal[:M],
            W
        )

    # Step 4: Effective channel
    effective_channel = compute_effective_channel_temporal(
        channel_estimate,
        W
    )

    return whitened_signal, effective_channel


# ====================================================================
# Unified Interface
# ====================================================================

def single_antenna_temporal_whitening(
    received_signal,
    channel_estimate,
    training_sequence,
    enable_temporal_whitening=True,
    method='bias_removal',  # 'mahalanobis' or 'bias_removal'
    regularization=1e-6,
    thermal_noise_variance=None,
    full_burst=True
):
    """Unified wrapper with SAIC-like interface.

    When disabled performs a simple matched-filter baseline.
    Returns the processed signal and an effective channel vector.
    """

    training_sequence = np.array(training_sequence, dtype=complex)

    if not enable_temporal_whitening:
        # baseline matched-filter processing
        matched_filter = channel_estimate[::-1].conj()
        processed_signal = np.convolve(
            received_signal,
            matched_filter,
            mode='same'
        )
        effective_channel = channel_estimate

        return processed_signal, effective_channel

    # run selected temporal whitening method
    if method == 'bias_removal':
        processed_signal, effective_channel = temporal_whitening_bias_removal(
            received_signal,
            channel_estimate,
            training_sequence,
            thermal_noise_variance,
            regularization,
            full_burst
        )
    else:  # method == 'mahalanobis'
        processed_signal, effective_channel = temporal_whitening_mahalanobis(
            received_signal,
            channel_estimate,
            training_sequence,
            regularization,
            full_burst
        )

    return processed_signal, effective_channel


# ====================================================================
# Diagnostic Functions
# ====================================================================

def compute_whitening_gain(
    R_before,
    R_after
):
    """Compute a simple metric of whitening effectiveness (dB).

    The metric is based on reduction of off-diagonal power in the
    covariance matrix (smaller off-diagonal power => better whitening).
    """
    # Metric: reduction in off-diagonal correlations
    M = R_before.shape[0]

    # Off-diagonal power
    off_diag_before = np.sum(np.abs(R_before)**2) - np.sum(np.abs(np.diag(R_before))**2)
    off_diag_after = np.sum(np.abs(R_after)**2) - np.sum(np.abs(np.diag(R_after))**2)

    if off_diag_after > 0:
        gain_db = 10 * np.log10(off_diag_before / off_diag_after)
    else:
        gain_db = float('inf')

    return gain_db


def analyze_eigenvalue_structure(eigenvalues):
    """Return simple diagnostics for an eigenvalue spectrum.

    The returned report contains condition number, signal/noise
    power and a rough SNR estimate in dB.
    """
    M = len(eigenvalues)

    # Condition number
    cond_number = eigenvalues[0] / eigenvalues[-1]

    # Signal/noise separation (simplified)
    # Assume последние 25% - noise subspace
    num_signal = int(0.75 * M)
    signal_eigenvalues = eigenvalues[:num_signal]
    noise_eigenvalues = eigenvalues[num_signal:]

    signal_power = np.sum(signal_eigenvalues)
    noise_power = np.sum(noise_eigenvalues)

    report = {
        'condition_number': cond_number,
        'max_eigenvalue': eigenvalues[0],
        'min_eigenvalue': eigenvalues[-1],
        'signal_power': signal_power,
        'noise_power': noise_power,
        'snr_estimate_db': 10 * np.log10(signal_power / noise_power)
    }

    return report
