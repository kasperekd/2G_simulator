import numpy as np
from scipy.linalg import orth
from scipy.signal import convolve

def estimate_interference_metric(r_ts, training_sequence, h_est, L):
    # Interference projection (IP) for multipath => hybrid IP & SP
    # Используем матрицу Toeplitz (Можно использовать просто свёртку)
    # X = np.zeros((len(r_ts), L), dtype=complex)
    # for i in range(L):
    #     X[i:i+len(training_sequence), i] = training_sequence
        
    # y_hat = X @ h_est
    y_hat = convolve(training_sequence, h_est, mode="full")[:-L+1]

    e = r_ts - y_hat

    P_signal = np.linalg.norm(y_hat)**2
    P_interf = np.linalg.norm(e)**2

    eta = P_interf / (P_signal + P_interf + 1e-12)
    # eta = 10 * np.log10(P_signal / P_interf + 1e-12)
    return eta, P_signal, P_interf

def interference_projection(received_signal, training_sequence):
    # Проблема с multipath - размывается
    if len(received_signal) != len(training_sequence):
        print(f"received signal = {len(received_signal)},\ntraining_sequence = {len(training_sequence)}")
        raise ValueError ("Received signal and training sequence must have the same length")
    
    u = training_sequence / np.linalg.norm(training_sequence)
    estimated_signal = np.dot(received_signal, u) * u
    residual = received_signal - estimated_signal
    signal_power = np.sum(np.abs(estimated_signal)**2)
    interference_power = np.sum(np.abs(residual))

    # sir_db = 10 * np.log10(signal_power / interference_power + 1e-12)

    sir_db = interference_power / (interference_power + signal_power + 1e-12)

    return sir_db, estimated_signal, residual

# @TODO надо решить проблему с индесацией в матрице
# НЕ РАБОТАЕТ!!!
def signal_projection(received_signal, training_sequence, channel_order):
    M = channel_order
    L_max = len(training_sequence) - M + 1
    
    r = received_signal[:L_max]
    L = len(r)

    # 1. Формируем матрицу Ганкеля (пространство сигнала)
    A = np.zeros((L, M), dtype=complex)
    for i in range(L):
        for j in range(M):
            A[i, j] = training_sequence[M - 1 - j + i]

    # 2. Находим ортонормированный базис
    Q, _ = np.linalg.qr(A)

    # 3. Проекция на пространство полезного сигнала
    estimated_signal = Q @ (Q.conj().T @ r)
    
    # 4. Остаток (помеха)
    residual = r - estimated_signal

    # 5. Расчет мощностей
    P_sig = np.linalg.norm(estimated_signal)**2
    P_int = np.linalg.norm(residual)**2

    # Метрика (от 0 до 1)
    eta = P_int / (P_sig + P_int + 1e-12)

    return eta, estimated_signal, residual

# @TODO: доделать метод на подпространстве 
# НЕ РАБОТАЕТ!!!
# недостаточна выборка данных для этого метода.
def subspace_based_sir(observations, signal_subspace_rank):
    L,K = observations.shape

    M = signal_subspace_rank

    # if M <= 0 or M >= L:
    #     raise ValueError("signal subspace rank must satisfy 0 < rank < observation dimension")

    covariance_matrix = (observations @ observations.T.conj()) / K

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    signal_eigenvalues = eigenvalues[:M]
    noise_eigenvalues = eigenvalues[M:]

    sigma_in = np.mean(noise_eigenvalues)
    sigma_s = np.sum(signal_eigenvalues - sigma_in)

    # sir_db = np.log10(sigma_s / sigma_in + 1e-12)
    sir_db = sigma_in / (sigma_s + sigma_in + 1e-12)

    return sir_db, signal_eigenvalues, noise_eigenvalues