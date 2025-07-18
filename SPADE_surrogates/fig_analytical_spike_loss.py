"""
Module to plot analytical spike count reduction for PPD and Gamma spike trains
and their uniform-dithered surrogates.

This module provides functions to calculate and visualize the theoretical
spike count reduction that occurs when continuous spike trains are discretized
into bins, for both:
- Poisson Process with Dead time (PPD) 
- Gamma distributed inter-spike intervals

The analysis includes both original spike trains and their uniform-dithered
surrogates used for statistical testing.
"""

import math
import os
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormaps
import quantities as pq
from scipy.special import gamma, gammainc

# Constants
DEFAULT_FIGURE_SIZE = (6.5/2, 2.5)
DEFAULT_DPI = 300
DEFAULT_N_TERMS = 50  # Number of terms for series expansion
COLORMAP_NAME = 'Greys'
COLOR_INTENSITIES = (0.3, 0.5, 0.7, 0.9)


def calculate_clipped_firing_rate_gamma(firing_rate: Union[np.ndarray, float], 
                                      shape_factor: float, 
                                      bin_size: float) -> Union[np.ndarray, float]:
    """
    Calculate the effective firing rate after binning for Gamma-distributed ISIs.
    
    For Gamma-distributed inter-spike intervals, this function computes how
    the effective firing rate changes when continuous spike trains are
    discretized into bins of fixed size.
    
    Parameters
    ----------
    firing_rate : np.ndarray or float
        Original firing rate(s) in Hz
    shape_factor : float
        Shape parameter of the Gamma distribution (γ)
    bin_size : float
        Size of discretization bins in seconds
        
    Returns
    -------
    np.ndarray or float
        Effective firing rate after binning
    """
    # Handle array inputs recursively
    if isinstance(firing_rate, np.ndarray):
        clipped_rates = np.zeros_like(firing_rate)
        for i, rate in enumerate(firing_rate):
            clipped_rates[i] = calculate_clipped_firing_rate_gamma(
                rate, shape_factor, bin_size)
        return clipped_rates
    
    # Calculate the scaling parameter for the Gamma distribution
    scale_param = shape_factor * firing_rate * bin_size
    
    # Calculate the correction factor using incomplete gamma functions
    gamma_term = gamma(shape_factor)
    incomplete_gamma_k = gammainc(shape_factor, scale_param)
    incomplete_gamma_k_plus_1 = gammainc(shape_factor + 1, scale_param)
    
    correction_factor = (
        1.0 - (1.0 / gamma_term) * (
            gamma_term * incomplete_gamma_k - 
            gamma(shape_factor + 1) * incomplete_gamma_k_plus_1 / scale_param
        )
    )
    
    return firing_rate * correction_factor


def calculate_clipped_firing_rate_ppd(firing_rate: Union[np.ndarray, float], 
                                    dead_time: float, 
                                    bin_size: float) -> Union[np.ndarray, float]:
    """
    Calculate the effective firing rate after binning for PPD spike trains.
    
    For Poisson Process with Dead time (PPD), this function computes the
    effective firing rate when continuous spike trains are discretized,
    accounting for the refractory period.
    
    Parameters
    ----------
    firing_rate : np.ndarray or float
        Original firing rate(s) in Hz
    dead_time : float
        Refractory period in seconds
    bin_size : float
        Size of discretization bins in seconds
        
    Returns
    -------
    np.ndarray or float
        Effective firing rate after binning
    """
    # Handle array inputs recursively
    if isinstance(firing_rate, np.ndarray):
        clipped_rates = np.zeros_like(firing_rate)
        for i, rate in enumerate(firing_rate):
            clipped_rates[i] = calculate_clipped_firing_rate_ppd(
                rate, dead_time, bin_size)
        return clipped_rates
    
    # If bin size is smaller than dead time, no correction needed
    if bin_size <= dead_time:
        return firing_rate
    
    # Calculate effective rate accounting for dead time
    effective_rate = firing_rate / (1.0 - firing_rate * dead_time)
    active_time = bin_size - dead_time
    
    # Calculate clipped firing rate using exponential decay
    exponential_term = np.exp(-effective_rate * active_time)
    
    clipped_rate = (
        firing_rate * dead_time / bin_size * exponential_term +
        (1.0 / bin_size) * (1.0 - exponential_term)
    )
    
    return clipped_rate


def calculate_clipped_firing_rate_ppd_surrogates(firing_rate: pq.Quantity,
                                               dead_time: pq.Quantity,
                                               dither_amount: pq.Quantity,
                                               bin_size: pq.Quantity) -> pq.Quantity:
    """
    Calculate effective firing rate for uniform-dithered PPD surrogates.
    
    This function computes the theoretical firing rate for uniform-dithered
    surrogates of PPD spike trains, using a series expansion approach.
    
    Parameters
    ----------
    firing_rate : pq.Quantity
        Original firing rate with units
    dead_time : pq.Quantity
        Refractory period with units
    dither_amount : pq.Quantity
        Amount of uniform dithering applied with units
    bin_size : pq.Quantity
        Size of discretization bins with units
        
    Returns
    -------
    pq.Quantity
        Effective firing rate for dithered surrogates
    """
    # Calculate effective firing rate without dead time effects
    effective_rate = firing_rate / (1.0 - (dead_time * firing_rate).simplified.magnitude)
    
    # Calculate the upper limit for the series expansion
    max_terms = math.floor(2 * dither_amount / dead_time) + 1
    
    # Initialize the lambda_u parameter
    lambda_u = np.zeros(firing_rate.shape)
    
    # Series expansion for dithered spike trains
    for i in range(1, max_terms):
        # Time window for this term
        time_window = (2 * dither_amount - i * dead_time).simplified.magnitude
        rate_time_product = (effective_rate * time_window).simplified.magnitude
        
        # Add contribution from this term
        lambda_u += (
            time_window * gammainc(i, rate_time_product) -
            (i / effective_rate).simplified.magnitude * 
            gammainc(i + 1, rate_time_product)
        )
    
    # Normalize by dithering window
    dither_normalization = 1.0 / (2 * dither_amount.simplified.magnitude) ** 2
    lambda_u *= dither_normalization
    
    # Calculate final firing rate using Taylor expansion
    bin_size_magnitude = bin_size.simplified.magnitude
    lambda_u_bin = lambda_u * bin_size_magnitude
    
    correction_factor = (
        1.0 - 0.5 * lambda_u_bin + (1.0/6.0) * lambda_u_bin ** 2
    )
    
    return firing_rate * correction_factor


def calculate_clipped_firing_rate_gamma_surrogates(firing_rate: pq.Quantity,
                                                 shape_factor: float,
                                                 dither_amount: pq.Quantity,
                                                 bin_size: pq.Quantity,
                                                 n_terms: int = DEFAULT_N_TERMS) -> pq.Quantity:
    """
    Calculate effective firing rate for uniform-dithered Gamma surrogates.
    
    This function computes the theoretical firing rate for uniform-dithered
    surrogates of Gamma spike trains using series expansion.
    
    Parameters
    ----------
    firing_rate : pq.Quantity
        Original firing rate with units
    shape_factor : float
        Shape parameter of the Gamma distribution
    dither_amount : pq.Quantity
        Amount of uniform dithering applied with units
    bin_size : pq.Quantity
        Size of discretization bins with units
    n_terms : int, optional
        Number of terms in series expansion (default: 50)
        
    Returns
    -------
    pq.Quantity
        Effective firing rate for dithered surrogates
    """
    # Initialize the lambda_u parameter
    lambda_u = np.zeros(firing_rate.shape)
    
    # Series expansion for dithered Gamma spike trains
    for i in range(1, n_terms + 1):
        # Calculate rate-time product for this term
        dither_magnitude = (2 * dither_amount).simplified.magnitude
        rate_dither_product = (
            shape_factor * firing_rate * dither_magnitude
        ).simplified.magnitude
        
        # Add contribution from this term
        lambda_u += (
            dither_magnitude * gammainc(i * shape_factor, rate_dither_product) -
            (i / firing_rate).simplified.magnitude * 
            gammainc(i * shape_factor + 1, rate_dither_product)
        )
    
    # Normalize by dithering window
    dither_normalization = 1.0 / (2 * dither_amount.simplified.magnitude) ** 2
    lambda_u *= dither_normalization
    
    # Calculate final firing rate using Taylor expansion
    bin_size_magnitude = bin_size.simplified.magnitude
    lambda_u_bin = lambda_u * bin_size_magnitude
    
    correction_factor = (
        1.0 - 0.5 * lambda_u_bin + (1.0/6.0) * lambda_u_bin ** 2
    )
    
    return firing_rate * correction_factor


def create_analytical_spike_count_figure(firing_rates: pq.Quantity,
                                       bin_size: pq.Quantity,
                                       dither_amount: pq.Quantity,
                                       dead_times: pq.Quantity,
                                       shape_factors: np.ndarray,
                                       output_path: str) -> None:
    """
    Create figure showing analytical spike count reduction for different models.
    
    This function generates a two-panel figure comparing spike count reduction
    for PPD and Gamma spike train models, showing both original and
    uniform-dithered surrogate data.
    
    Parameters
    ----------
    firing_rates : pq.Quantity
        Array of firing rates to analyze
    bin_size : pq.Quantity
        Size of discretization bins
    dither_amount : pq.Quantity
        Amount of uniform dithering for surrogates
    dead_times : pq.Quantity
        Array of dead times for PPD analysis
    shape_factors : np.ndarray
        Array of shape factors for Gamma analysis
    output_path : str
        Path where to save the output figure
    """
    # Set up color scheme
    colormap = plt.get_cmap(COLORMAP_NAME)
    colors = [colormap(intensity) for intensity in COLOR_INTENSITIES]
    
    # Create figure with two subplots
    fig, (ax_ppd, ax_gamma) = plt.subplots(
        1, 2, figsize=DEFAULT_FIGURE_SIZE, sharey=True, dpi=DEFAULT_DPI)
    
    # Adjust layout
    fig.subplots_adjust(left=0.17, bottom=0.17, wspace=0.1, right=0.98)
    
    # Plot PPD analysis
    _plot_ppd_analysis(ax_ppd, firing_rates, bin_size, dither_amount, 
                      dead_times, colors)
    
    # Plot Gamma analysis  
    _plot_gamma_analysis(ax_gamma, firing_rates, bin_size, dither_amount,
                        shape_factors, colors)
    
    # Configure axes
    _configure_axes(ax_ppd, ax_gamma)
    
    # Save figure
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    plt.show()


def _plot_ppd_analysis(ax: plt.Axes, 
                      firing_rates: pq.Quantity,
                      bin_size: pq.Quantity,
                      dither_amount: pq.Quantity,
                      dead_times: pq.Quantity,
                      colors: list) -> None:
    """
    Plot PPD spike count reduction analysis.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    firing_rates : pq.Quantity
        Array of firing rates
    bin_size : pq.Quantity
        Size of discretization bins
    dither_amount : pq.Quantity
        Amount of uniform dithering
    dead_times : pq.Quantity
        Array of dead times to analyze
    colors : list
        List of colors for different dead times
    """
    for dead_time_idx, dead_time in enumerate(dead_times):
        # Calculate spike count reduction for original data
        clipped_rate = calculate_clipped_firing_rate_ppd(
            firing_rates.magnitude,
            dead_time.rescale(pq.s).magnitude,
            bin_size.rescale(pq.s).magnitude
        )
        spike_count_ratio = clipped_rate / firing_rates.magnitude
        spike_count_reduction = 1.0 - spike_count_ratio
        
        # Plot original data
        ax.plot(firing_rates, spike_count_reduction,
               label=f'd = {dead_time}',
               color=colors[dead_time_idx])
        
        # Calculate and plot surrogate data
        surrogate_clipped_rate = calculate_clipped_firing_rate_ppd_surrogates(
            firing_rates, dead_time, dither_amount, bin_size)
        surrogate_ratio = surrogate_clipped_rate / firing_rates
        surrogate_reduction = 1.0 - surrogate_ratio
        
        ax.plot(firing_rates, surrogate_reduction,
               linestyle='--', color=colors[dead_time_idx])


def _plot_gamma_analysis(ax: plt.Axes,
                        firing_rates: pq.Quantity,
                        bin_size: pq.Quantity,
                        dither_amount: pq.Quantity,
                        shape_factors: np.ndarray,
                        colors: list) -> None:
    """
    Plot Gamma spike count reduction analysis.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    firing_rates : pq.Quantity
        Array of firing rates
    bin_size : pq.Quantity
        Size of discretization bins
    dither_amount : pq.Quantity
        Amount of uniform dithering
    shape_factors : np.ndarray
        Array of shape factors to analyze
    colors : list
        List of colors for different shape factors
    """
    for shape_idx, shape_factor in enumerate(shape_factors):
        # Calculate spike count reduction for original data
        clipped_rate = calculate_clipped_firing_rate_gamma(
            firing_rates.magnitude, 
            shape_factor,
            bin_size.rescale(pq.s).magnitude
        )
        spike_count_ratio = clipped_rate / firing_rates.magnitude
        spike_count_reduction = 1.0 - spike_count_ratio
        
        # Plot original data
        ax.plot(firing_rates, spike_count_reduction,
               label=f'γ = {shape_factor:.1f}',
               color=colors[shape_idx])
        
        # Calculate and plot surrogate data
        surrogate_clipped_rate = calculate_clipped_firing_rate_gamma_surrogates(
            firing_rates, shape_factor, dither_amount, bin_size)
        surrogate_ratio = surrogate_clipped_rate / firing_rates
        surrogate_reduction = 1.0 - surrogate_ratio
        
        ax.plot(firing_rates, surrogate_reduction,
               linestyle='--', color=colors[shape_idx])


def _configure_axes(ax_ppd: plt.Axes, ax_gamma: plt.Axes) -> None:
    """
    Configure the appearance of both subplot axes.
    
    Parameters
    ----------
    ax_ppd : plt.Axes
        PPD subplot axes
    ax_gamma : plt.Axes
        Gamma subplot axes
    """
    # Set labels and titles
    ax_ppd.set_ylabel(r'$1 - N_{clip}$/$N$', labelpad=1.5, fontsize='small')
    
    subplot_configs = [
        (ax_ppd, 'PPD'),
        (ax_gamma, 'Gamma')
    ]
    
    for ax, title in subplot_configs:
        ax.set_xlabel(r'$\lambda$ (Hz)', labelpad=1.5, fontsize='small')
        ax.tick_params(axis='x', labelsize='small')
        ax.tick_params(axis='y', labelsize='small')
        ax.legend(fontsize='x-small')
        ax.set_title(title, fontsize=10)


def main() -> None:
    """
    Main function to generate the analytical spike count reduction figure.
    
    This function sets up the default parameters and generates the figure
    showing theoretical spike count reduction for PPD and Gamma models.
    """
    # Ensure output directory exists
    output_dir = '../plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define analysis parameters
    shape_factors = np.arange(1.0, 3.0, 0.5)
    firing_rates = np.arange(0.01, 101.0, 1.0) * pq.Hz
    bin_size = 5.0 * pq.ms
    dead_times = np.arange(1.5, 3.1, 0.5) * pq.ms
    dither_amount = 25 * pq.ms
    output_path = os.path.join(output_dir, 'fig_analytical_spike_count_reduction.svg')
    
    # Generate the figure
    create_analytical_spike_count_figure(
        firing_rates=firing_rates,
        bin_size=bin_size,
        dither_amount=dither_amount,
        dead_times=dead_times,
        shape_factors=shape_factors,
        output_path=output_path
    )
    
    print(f"Figure saved to: {output_path}")


if __name__ == '__main__':
    main()