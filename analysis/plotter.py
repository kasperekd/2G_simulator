import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(
    x_data,
    y_data=None,
    title="Scatter Plot",
    xlabel="In-Phase",
    ylabel="Quadrature",
    color="blue",
    alpha=0.6,
    show=True
):
    """Generates a scatter plot from input data.

    Can handle both regular x-y data and complex numbers. For complex input, plots
    real vs imaginary components.

    Args:
        x_data (array-like): Either complex numbers (uses real/imag parts) or
            x-coordinates. For complex data, y_data should be None.
        y_data (array-like, optional): y-coordinates (required if x_data isn't
            complex). Defaults to None.
        title (str, optional): Title of the plot. Defaults to "Scatter Plot".
        xlabel (str, optional): Label for x-axis. Defaults to "X-axis".
        ylabel (str, optional): Label for y-axis. Defaults to "Y-axis".
        color (str, optional): Point color. Defaults to "blue".
        alpha (float, optional): Point transparency (0-1). Defaults to 0.6.
        show (bool, optional): Whether to immediately display the plot.
            Defaults to True.

    Raises:
        ValueError: If neither complex data nor y_data is provided.
        TypeError: If input data types are incompatible.
    """
    plt.figure(figsize=(10, 6))
    
    if np.iscomplexobj(x_data):
        plt.scatter(x_data.real, x_data.imag, color=color, alpha=alpha)
    elif y_data is not None:
        plt.scatter(x_data, y_data, color=color, alpha=alpha)
    else:
        raise ValueError("Requires either complex data or both x_data and y_data")

    format_plot(title, xlabel, ylabel, show, grid=True)
    plt.axis('equal')


def plot_line(
    x_data,
    y_data=None,
    title="Line Plot",
    xlabel="X-axis",
    ylabel="Y-axis",
    color="blue",
    linewidth=2,
    show=True,
    grid=True
):
    """Generates a line plot from input data.

    Supports three modes:
    1. Regular x-y plot (when y_data provided)
    2. Complex signal plot (real and imaginary components)
    3. Simple y-values plot (automatic x-axis generation)

    Args:
        x_data (array-like): Input data or x-coordinates.
        y_data (array-like, optional): y-coordinates. Defaults to None.
        title (str, optional): Plot title. Defaults to "Line Plot".
        xlabel (str, optional): x-axis label. Defaults to "X-axis".
        ylabel (str, optional): y-axis label. Defaults to "Y-axis".
        color (str, optional): Line color for real/y data. Defaults to "red".
        linewidth (int, optional): Line width. Defaults to 2.
        show (bool, optional): Whether to display plot immediately.
            Defaults to True.
        grid (bool, optional): Whether to show grid. Defaults to True.

    Examples:
        >>> plot_line(np.arange(10), np.random.rand(10))
        >>> plot_line(np.exp(1j*np.linspace(0, 2*np.pi, 100)))
        >>> plot_line(np.sin(np.linspace(0, 2*np.pi, 100)))
    """
    plt.figure(figsize=(10, 6))
    
    if y_data is not None:
        plt.plot(x_data, y_data, color=color, linewidth=linewidth)
    elif np.iscomplexobj(x_data):
        plt.plot(x_data.real, color=color, linewidth=linewidth, label='Real part')
        plt.plot(x_data.imag, color='green', linewidth=linewidth, label='Imaginary part')
        plt.legend()
    else:
        plt.plot(x_data, color=color, linewidth=linewidth)
    
    format_plot(title, xlabel, ylabel, show, grid)


def format_plot(title, xlabel, ylabel, show=True, grid=True):
    """Applies consistent formatting to plots.

    Args:
        title (str): Plot title.
        xlabel (str): x-axis label.
        ylabel (str): y-axis label.
        show (bool, optional): Whether to display plot. Defaults to True.
        grid (bool, optional): Whether to show grid. Defaults to True.
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if grid:
        plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if show:
        plt.show()

def plot_channel_estimation_results(h_true, h_est, antenna_id):
    """Plots a detailed comparison of true and estimated channel responses."""
    plt.figure(figsize=(14, 6))
    plt.suptitle(f'Channel Estimation Comparison - Antenna {antenna_id}', fontsize=16)

    # Magnitude Comparison
    plt.subplot(1, 2, 1)
    markerline, stemlines, baseline = plt.stem(
        np.abs(h_true), linefmt='C0-', markerfmt='C0o', label='True Magnitude'
    )
    plt.setp(baseline, visible=False)
    
    markerline, stemlines, baseline = plt.stem(
        np.abs(h_est), linefmt='C1--', markerfmt='C1x', label='Estimated Magnitude'
    )
    plt.setp(baseline, visible=False)

    plt.title('Magnitude Response')
    plt.xlabel('Tap Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Phase Comparison
    plt.subplot(1, 2, 2)
    markerline, stemlines, baseline = plt.stem(
        np.angle(h_true), linefmt='C0-', markerfmt='C0o', label='True Phase'
    )
    plt.setp(baseline, visible=False)
    
    markerline, stemlines, baseline = plt.stem(
        np.angle(h_est), linefmt='C1--', markerfmt='C1x', label='Estimated Phase'
    )
    plt.setp(baseline, visible=False)
    
    plt.title('Phase Response')
    plt.xlabel('Tap Index')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_equalizer_constellations(y_hat_data, S_n, decoded_indices, constellation):
    """
    Visualizes the effectiveness of the equalizer by plotting received vs. reconstructed symbols.
    """
    # Approximate the equalized signal by removing the main tap's effect
    # This shows the signal cloud that the Viterbi decoder "sees" for each symbol.
    if S_n[0] == 0:
        print("Warning: S_n[0] is zero, cannot normalize equalizer constellation.")
        return
        
    equalized_symbols = y_hat_data / S_n[0]

    # Get the ideal constellation points corresponding to the decoded symbols
    reconstructed_symbols = constellation[decoded_indices]

    plt.figure(figsize=(8, 8))
    plt.scatter(equalized_symbols.real, equalized_symbols.imag, alpha=0.2, s=50, label='Equalized Symbols (y_hat/S0)')
    plt.scatter(reconstructed_symbols.real, reconstructed_symbols.imag, c='red', marker='x', s=100, label='Decoded Symbols (Ideal Target)')
    plt.title('Equalizer Output vs. Decoded Symbols')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

def visualize_mimo_reception(rx1, rx2, h11, h12, h21, h22):
        plt.figure(figsize=(15, 10))
        
        # Channels
        plt.subplot(321)
        plt.plot(np.abs(h11), label='h11')
        plt.plot(np.abs(h21), label='h21')
        plt.title('Channel Responses (to RX1)')
        plt.legend()
        
        plt.subplot(322)
        plt.plot(np.abs(h12), label='h12')
        plt.plot(np.abs(h22), label='h22')
        plt.title('Channel Responses (to RX2)')
        plt.legend()
        
        # Received Signals (time domain)
        plt.subplot(323)
        plt.plot(rx1.real, label='Real')
        plt.plot(rx1.imag, label='Imag')
        plt.title('Received Signal (Antenna 1)')
        plt.legend()
        
        plt.subplot(324)
        plt.plot(rx2.real, label='Real')
        plt.plot(rx2.imag, label='Imag')
        plt.title('Received Signal (Antenna 2)')
        plt.legend()
        
        # Received Signals (constellation)
        plt.subplot(325)
        plt.scatter(rx1.real, rx1.imag, alpha=0.5)
        plt.title('Constellation (Antenna 1)')
        plt.grid(True)
        
        plt.subplot(326)
        plt.scatter(rx2.real, rx2.imag, alpha=0.5)
        plt.title('Constellation (Antenna 2)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def plot_ber_vs_ci(ci_db_range, ber_results, receiver_types):
    """
    Plots Bit Error Rate (BER) as a function of Carrier-to-Interference (C/I) ratio.

    Args:
        ci_db_range (np.ndarray): Array of C/I values in dB.
        ber_results (dict): A dictionary where keys are receiver type names (str)
                            and values are the corresponding BER arrays.
        receiver_types (list): List of receiver names to plot, defining the order.
    """
    plt.figure(figsize=(10, 7))
    
    for receiver_type in receiver_types:
        if receiver_type in ber_results:
            plt.semilogy(ci_db_range, ber_results[receiver_type], 'o-', label=receiver_type)

    plt.title('BER vs. Carrier-to-Interference Ratio (C/I)', fontsize=16)
    plt.xlabel('C/I (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.ylim(1e-5, 1) # Set typical BER limits
    plt.xlim(min(ci_db_range), max(ci_db_range))
    plt.show()