import numpy as np
def add_thermal_noise(signal, bs_nf_db, fs_hz, temp_k, seed):
    # Physical noise calculation
    # np.random.seed(seed)
    k_boltzmann = 1.380649e-23
    nf_linear = 10**(bs_nf_db / 10)
    # TODO: Noise bandwidth should be channel bandwidth (e.g. 200e3 for GSM) not Fs
    noise_power = k_boltzmann * temp_k * fs_hz * nf_linear
    noise_std_dev = np.sqrt(noise_power / 2)
    noise = noise_std_dev * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise