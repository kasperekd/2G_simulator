import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(
    x_data,
    y_data=None,
    title="Scatter Plot",
    xlabel="X-axis",
    ylabel="Y-axis",
    color="blue",
    alpha=0.6
):
    """Plots a scatter chart from input data.

    Args:
        x_data: Either complex numbers (uses real/imag parts) or x-coordinates
        y_data: y-coordinates (required if x_data isn't complex). Default None
        title: Title of the plot. Default "Scatter Plot"
        xlabel: Label for x-axis. Default "X-axis"
        ylabel: Label for y-axis. Default "Y-axis"
        color: Point color. Default "blue"
        alpha: Point transparency (0-1). Default 0.6

    Raises:
        ValueError: If neither complex data nor y_data is provided
    """
    plt.figure(figsize=(10, 6))
    
    if hasattr(x_data, "real") and hasattr(x_data, "imag"):
        plt.scatter(x_data.real, x_data.imag, color=color, alpha=alpha)
    elif y_data is not None:
        plt.scatter(x_data, y_data, color=color, alpha=alpha)
    else:
        raise ValueError("Requires either complex data or both x_data and y_data")

    format_plot(title, xlabel, ylabel)

def plot_line(
    x_data,
    y_data,
    title="Line Plot",
    xlabel="X-axis",
    ylabel="Y-axis",
    color="red",
    linewidth=2
):
    """Plots a line chart from x-y data.

    Args:
        x_data: x-coordinates data
        y_data: y-coordinates data
        title: Title of the plot. Default "Line Plot"
        xlabel: Label for x-axis. Default "X-axis"
        ylabel: Label for y-axis. Default "Y-axis"
        color: Line color. Default "red"
        linewidth: Line width. Default 2
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, color=color, linewidth=linewidth)
    format_plot(title, xlabel, ylabel)

def format_plot(title, xlabel, ylabel):
    """Helper function to format plots consistently."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
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