import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(
    x_data,
    y_data=None,
    title="Scatter Plot",
    xlabel="X-axis",
    ylabel="Y-axis",
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
        if y_data is not None:
            plt.scatter(y_data.real, y_data.imag, color='red', alpha=alpha)
    elif y_data is not None:
        plt.scatter(x_data, y_data, color=color, alpha=alpha)
    else:
        raise ValueError("Requires either complex data or both x_data and y_data")

    format_plot(title, xlabel, ylabel, show)


def plot_line(
    x_data,
    y_data=None,
    title="Line Plot",
    xlabel="X-axis",
    ylabel="Y-axis",
    color="red",
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
        plt.plot(x_data.real, color=color, linewidth=linewidth,
                label='Real part')
        plt.plot(x_data.imag, color="blue", linewidth=linewidth,
                label='Imaginary part')
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
        plt.grid(alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()

# def plot_ber_vs_snr(snr_db, ber_values):
#     """Plots BER versus SNR on a logarithmic scale.

#     Args:
#         snr_db: List of SNR values in decibels.
#         ber_values: Corresponding BER values.
#     """
#     plt.figure(figsize=(8, 5))
#     plt.semilogy(snr_db, ber_values, marker='o', linestyle='-', color='b')
#     plt.xlabel("SNR (dB)")
#     plt.ylabel("BER (log scale)")
#     plt.title("Bit Error Rate vs Signal-to-Noise Ratio")
#     plt.grid(True, which="both", ls="--")
#     plt.show()