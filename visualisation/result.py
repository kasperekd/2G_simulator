import numpy as np
import matplotlib.pyplot as plt
def plot_results(ratio_values, ber_values, config):
    plt.figure(figsize=(10, 6))
    ber_plot = np.where(ber_values == 0, 1e-6, ber_values)
    plt.semilogy(ratio_values, ber_plot, 'bo-', linewidth=2, markersize=6)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    title = f"BER vs {config.mode_selection.calculation_mode} for {config.core_simulation_parameters.modulation_type}\n"
    title += (
        f"(Estimation: {config.mode_selection.channel_estimation_method}, "
        f"Interferers: {config.num_interferers})"
    )
    plt.title(title)
    plt.xlabel(f'{config.mode_selection.calculation_mode} (dB)')
    plt.ylabel('BER')
    plt.ylim([1e-5, 1])
    plt.show()