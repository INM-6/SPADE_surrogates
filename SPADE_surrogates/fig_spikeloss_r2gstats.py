"""
Script to create the figure showing spike count reduction in experimental data.

This script generates a multi-panel figure that visualizes:
- Spike count reduction from binning and uniform dithering (UD) surrogates
- Inter-spike interval (ISI) distributions for original and surrogate data
- CV2 and dead time distributions across neurons
"""

import itertools
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from elephant import conversion, spike_train_surrogates
import elephant.statistics as stat

from generate_artificial_data import estimate_rate_deadtime, create_st_list
from rgutils import load_processed_spike_trains

# Constants
XLABEL_PADDING = -0.25
DEFAULT_FONTSIZE = 9
FIGURE_SIZE = (5.2, 5.5)
FIGURE_DPI = 300


def count_spikes_in_bins(binned_spike_train):
    """
    Count the number of bins containing spikes in a binned spike train.
    
    Parameters
    ----------
    binned_spike_train : elephant.conversion.BinnedSpikeTrain
        The binned spike train to analyze
        
    Returns
    -------
    int
        Number of bins containing at least one spike
    """
    spike_array = binned_spike_train.to_array()[0, :]
    binary_array = np.array(spike_array > 0, dtype=int)
    return np.sum(binary_array)


def calculate_spike_count_statistics(spike_trains, bin_size, mean_list, std_list):
    """
    Calculate mean and standard deviation of spike counts after binning.
    
    Parameters
    ----------
    spike_trains : list
        List of spike trains to analyze
    bin_size : pq.Quantity
        Size of bins for spike train discretization
    mean_list : list
        List to append mean spike counts (modified in place)
    std_list : list
        List to append standard deviations (modified in place)
    """
    spike_counts = []
    for spike_train in spike_trains:
        binned_train = conversion.BinnedSpikeTrain(spike_train, bin_size=bin_size)
        spike_counts.append(count_spikes_in_bins(binned_train))
    
    mean_list.append(np.mean(spike_counts))
    std_list.append(np.std(spike_counts))


def calculate_residual_statistics(original_spike_count, surrogate_trains, bin_size,
                                mean_residuals, std_residuals):
    """
    Calculate statistics for residuals between original and surrogate spike counts.
    
    Parameters
    ----------
    original_spike_count : int
        Spike count from the original neuron
    surrogate_trains : list
        List of uniformly dithered surrogate spike trains
    bin_size : pq.Quantity
        Size of bins for spike train discretization
    mean_residuals : list
        List to append mean residuals (modified in place)
    std_residuals : list
        List to append standard deviations of residuals (modified in place)
    """
    residuals = []
    for surrogate_train in surrogate_trains:
        binned_surrogate = conversion.BinnedSpikeTrain(surrogate_train, bin_size=bin_size)
        surrogate_count = count_spikes_in_bins(binned_surrogate)
        residuals.append(original_spike_count - surrogate_count)
    
    mean_residuals.append(np.mean(residuals))
    std_residuals.append(np.std(residuals))


def analyze_spike_count_reduction(spike_trains, dither_amount, bin_size, n_surrogates,
                                trial_length, window_length):
    """
    Analyze spike count reduction from binning and uniform dithering.
    
    Parameters
    ----------
    spike_trains : list
        List of spike trains to analyze
    dither_amount : pq.Quantity
        Amount of dithering for surrogate generation
    bin_size : pq.Quantity
        Size of bins for discretization
    n_surrogates : int
        Number of surrogate trains to generate
    trial_length : pq.Quantity
        Length of each trial
    window_length : int
        Window length for analysis
        
    Returns
    -------
    tuple
        (firing_rates, mean_residuals, std_residuals)
    """
    # Calculate separation between trials
    trial_separation = 2 * bin_size * window_length
    
    original_spike_counts = []
    binned_spike_counts = []
    mean_residuals = []
    std_residuals = []
    
    # Calculate number of trials
    n_trials = int(spike_trains[0].t_stop / (trial_length + trial_separation))
    
    for spike_train in spike_trains:
        # Process original spike train
        binned_train = conversion.BinnedSpikeTrain(spike_train, bin_size=bin_size)
        original_count = len(spike_train)
        binned_count = count_spikes_in_bins(binned_train)
        
        original_spike_counts.append(original_count)
        binned_spike_counts.append(binned_count)
        
        # Generate and analyze surrogates
        surrogate_trains = spike_train_surrogates.dither_spikes(
            spike_train, dither_amount, n_surrogates=n_surrogates, edges=True)
        
        calculate_residual_statistics(
            binned_count, surrogate_trains, bin_size, mean_residuals, std_residuals)
    
    # Calculate firing rates
    firing_rates = np.array(original_spike_counts) / (n_trials * trial_length)
    
    return firing_rates, mean_residuals, std_residuals


def plot_spike_count_reduction(ax_loss, ax_residuals, spike_trains, dither_amount,
                             bin_size, n_surrogates, trial_length, window_length,
                             fontsize):
    """
    Plot spike count reduction from binning and uniform dithering.
    
    Creates two panels:
    1. Top panel: Spike count reduction vs firing rate
    2. Bottom panel: Residuals between original and surrogate reductions
    
    Parameters
    ----------
    ax_loss : matplotlib.axes.Axes
        Axes for the spike count reduction plot
    ax_residuals : matplotlib.axes.Axes
        Axes for the residuals plot
    spike_trains : list
        List of spike trains to analyze
    dither_amount : pq.Quantity
        Amount of dithering for surrogate generation
    bin_size : pq.Quantity
        Size of bins for discretization
    n_surrogates : int
        Number of surrogate trains to generate
    trial_length : pq.Quantity
        Length of each trial
    window_length : int
        Window length for analysis
    fontsize : int
        Font size for labels
    """
    # Calculate separation between trials
    trial_separation = 2 * bin_size * window_length
    
    # Initialize data containers
    original_spike_counts = []
    binned_spike_counts = []
    mean_dithered_counts = []
    std_dithered_counts = []
    
    # Calculate number of trials
    n_trials = int(spike_trains[0].t_stop / (trial_length + trial_separation))
    
    # Process each spike train
    for spike_train in spike_trains:
        # Original and binned counts
        binned_train = conversion.BinnedSpikeTrain(spike_train, bin_size=bin_size)
        original_spike_counts.append(len(spike_train))
        binned_spike_counts.append(count_spikes_in_bins(binned_train))
        
        # Generate surrogates and calculate statistics
        surrogate_trains = spike_train_surrogates.dither_spikes(
            spike_train, dither_amount, n=n_surrogates, edges=True)
        calculate_spike_count_statistics(
            surrogate_trains, bin_size, mean_dithered_counts, std_dithered_counts)
    
    # Convert to numpy arrays for calculations
    original_counts = np.array(original_spike_counts)
    binned_counts = np.array(binned_spike_counts)
    mean_dithered = np.array(mean_dithered_counts)
    std_dithered = np.array(std_dithered_counts)
    
    # Calculate firing rates and losses
    firing_rates = original_counts / (n_trials * trial_length)
    binned_loss = 1.0 - binned_counts / original_counts
    mean_dithered_loss = 1.0 - mean_dithered / original_counts
    std_dithered_loss = std_dithered / original_counts
    
    # Plot spike count reduction
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    ax_loss.scatter(firing_rates, binned_loss, label='Original',
                   color=colors[0], marker='x')
    ax_loss.errorbar(firing_rates, mean_dithered_loss, yerr=std_dithered_loss,
                    fmt='o', label='UD Surrogate', color=colors[1], marker='x')
    
    ax_loss.set_ylabel('Spike count\ndecrease', fontsize=fontsize)
    ax_loss.set_xlim(left=2.0 / trial_length.magnitude, right=65)
    ax_loss.set_ylim(bottom=-0.01, top=0.23)
    ax_loss.tick_params(axis="x", labelsize=8)
    ax_loss.tick_params(axis="y", labelsize=8)
    ax_loss.legend(fontsize=fontsize - 2, loc='upper right')
    
    # Plot residuals
    ax_residuals.errorbar(firing_rates, -binned_loss + mean_dithered_loss,
                         yerr=std_dithered_loss, fmt='o',
                         label='UD Surrogate + clipping', color='grey', marker='x')
    
    ax_residuals.set_xlabel('Average Firing rate (Hz)', fontsize=fontsize,
                           labelpad=XLABEL_PADDING)
    ax_residuals.set_ylabel('Residuals', fontsize=fontsize)
    ax_residuals.set_xlim(left=0, right=65)
    ax_residuals.set_ylim(bottom=-0.05, top=0.125)
    ax_residuals.tick_params(axis="x", labelsize=8)
    ax_residuals.tick_params(axis="y", labelsize=8)
    ax_residuals.set_xticks(np.arange(0, 65, 20))


def plot_isi_distribution(spike_train, ax, dither_amount, n_surrogates=500,
                         show_ylabel=True, show_xlabel=True, fontsize=14,
                         show_legend=False):
    """
    Plot inter-spike interval (ISI) distributions for original and surrogate data.
    
    Parameters
    ----------
    spike_train : neo.SpikeTrain
        The spike train to analyze
    ax : matplotlib.axes.Axes
        Axes for plotting
    dither_amount : pq.Quantity
        Amount of dithering for surrogate generation
    n_surrogates : int, optional
        Number of surrogates to generate (default: 500)
    show_ylabel : bool, optional
        Whether to show y-axis label (default: True)
    show_xlabel : bool, optional
        Whether to show x-axis label (default: True)
    fontsize : int, optional
        Font size for labels (default: 14)
    show_legend : bool, optional
        Whether to show legend (default: False)
    """
    # Calculate ISI for original spike train
    original_isi = stat.isi(spike_train)
    
    # Generate surrogates and calculate their ISIs
    surrogate_trains = spike_train_surrogates.dither_spikes(
        spike_train, n_surrogates=n_surrogates, dither=dither_amount * pq.s)
    surrogate_isis = [stat.isi(surrogate) for surrogate in surrogate_trains]
    
    # Set up histogram parameters
    bin_width = 0.001
    bins = np.arange(0, 0.05, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    
    # Calculate histogram for original data
    original_hist, _ = np.histogram(original_isi, bins=bins, density=True)
    
    # Calculate histograms for surrogates
    surrogate_hists = [np.histogram(isi, bins=bins, density=True)[0] 
                      for isi in surrogate_isis]
    surrogate_mean = np.mean(surrogate_hists, axis=0)
    surrogate_std = np.std(surrogate_hists, axis=0)
    
    # Plot original ISI distribution
    ax.plot(bin_centers, original_hist, label='Original')
    
    # Plot surrogate ISI distribution with confidence band
    ax.fill_between(bin_centers, surrogate_mean - surrogate_std,
                   surrogate_mean + surrogate_std, color='lightgrey')
    ax.plot(bin_centers, surrogate_mean, label='UD surrogates', color='grey')
    
    # Add bin size indicator
    ax.axvline(x=bin_centers[5], color='navy', linestyle='--')
    
    # Configure axes
    ax.set_xlim(0, 0.05)
    ax.set_ylim(-1, 81)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    
    if show_legend:
        ax.legend(fontsize=fontsize - 2, loc='upper right')
    if show_xlabel:
        ax.set_xlabel('ISI (s)', fontsize=fontsize, labelpad=XLABEL_PADDING)
    if show_ylabel:
        ax.set_ylabel('Count', fontsize=fontsize)


def create_trial_based_spike_trains(spike_trains, trial_separation, trial_length):
    """
    Convert concatenated spike trains into trial-based format.
    
    Parameters
    ----------
    spike_trains : list
        List of concatenated spike trains
    trial_separation : pq.Quantity
        Buffer time between trials
    trial_length : pq.Quantity
        Length of each trial
        
    Returns
    -------
    list
        List of spike trains organized by trials
    """
    trial_based_trains = []
    
    for spike_train in spike_trains:
        single_trial_trains = create_st_list(
            spiketrain=spike_train,
            sep=trial_separation,
            epoch_length=trial_length
        )
        trial_based_trains.append(single_trial_trains)
    
    return trial_based_trains


def calculate_cv2(isis_list):
    """
    Calculate CV2 (coefficient of variation) from ISI list.
    
    Implementation based on Van Vreeswijk 2010.
    
    Parameters
    ----------
    isis_list : list
        List of ISI arrays for different trials
        
    Returns
    -------
    float
        CV2 value
    """
    numerator = np.sum([
        2 * np.sum(np.abs(isis[:-1] - isis[1:]) / (isis[:-1] + isis[1:]))
        for isis in isis_list
    ])
    
    denominator = np.sum([
        len(isis) - 1 if len(isis) > 0 else 0
        for isis in isis_list
    ])
    
    return numerator / denominator if denominator > 0 else np.nan


def plot_cv2_distribution(spike_trains, ax, trial_length, trial_separation,
                         show_xlabel=True, show_ylabel=True, fontsize=14):
    """
    Plot distribution of CV2 values across neurons.
    
    Parameters
    ----------
    spike_trains : list
        List of spike trains
    ax : matplotlib.axes.Axes
        Axes for plotting
    trial_length : pq.Quantity
        Length of each trial
    trial_separation : pq.Quantity
        Separation time between trials
    show_xlabel : bool, optional
        Whether to show x-axis label (default: True)
    show_ylabel : bool, optional
        Whether to show y-axis label (default: True)
    fontsize : int, optional
        Font size for labels (default: 14)
    """
    # Convert to trial-based format
    trial_based_trains = create_trial_based_spike_trains(
        spike_trains, trial_separation, trial_length)
    
    # Calculate CV2 for each neuron
    cv2_values = []
    for trial_trains in trial_based_trains:
        isis_list = [np.diff(train.magnitude) for train in trial_trains if len(train) > 1]
        cv2 = calculate_cv2(isis_list)
        cv2_values.append(cv2)
    
    # Create histogram
    bin_width = 0.1
    bins = np.arange(0, 2, bin_width)
    ax.hist(cv2_values, bins, alpha=1)
    
    # Configure axes
    ax.set_xticks(np.arange(0, 1.9, 0.5))
    ax.set_xlim(0.2, 1.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    
    if show_xlabel:
        ax.set_xlabel('CV2', fontsize=fontsize, labelpad=XLABEL_PADDING)
    if show_ylabel:
        ax.set_ylabel('Count', fontsize=fontsize)


def plot_dead_time_distribution(spike_trains, ax, sorting_dead_time, trial_separation,
                               max_refractory=4 * pq.ms, show_xlabel=True,
                               show_ylabel=True, fontsize=14):
    """
    Plot distribution of dead times (minimum ISI) across neurons.
    
    Parameters
    ----------
    spike_trains : list
        List of spike trains
    ax : matplotlib.axes.Axes
        Axes for plotting
    sorting_dead_time : pq.Quantity
        Dead time used during spike sorting
    trial_separation : pq.Quantity
        Separation time between trials
    max_refractory : pq.Quantity, optional
        Maximum refractory period (default: 4 ms)
    show_xlabel : bool, optional
        Whether to show x-axis label (default: True)
    show_ylabel : bool, optional
        Whether to show y-axis label (default: True)
    fontsize : int, optional
        Font size for labels (default: 14)
    """
    # Calculate dead times for each neuron
    dead_times = []
    for spike_train in spike_trains:
        _, dead_time, _ = estimate_rate_deadtime(
            spike_train,
            max_refractory=max_refractory,
            sampling_period=1 * pq.ms,
            sep=trial_separation
        )
        dead_times.append(dead_time * 1000)  # Convert to ms
    
    # Remove NaN values
    dead_times = np.array(dead_times)
    valid_dead_times = dead_times[~np.isnan(dead_times)]
    
    # Create histogram
    bin_width = 0.1
    bins = np.arange(0, 4, bin_width)
    ax.hist(valid_dead_times, bins, alpha=1)
    
    # Add sorting dead time reference line
    sorting_dead_time_ms = sorting_dead_time.rescale(pq.ms).magnitude
    ax.axvline(x=sorting_dead_time_ms, color='grey', linestyle='--')
    
    # Configure axes
    ax.set_xticks(np.arange(0, 4, 1))
    ax.set_xlim(-0.2, 3.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    
    if show_xlabel:
        ax.set_xlabel('d (ms)', fontsize=fontsize, labelpad=XLABEL_PADDING)
    if show_ylabel:
        ax.set_ylabel('Count', fontsize=fontsize)


def create_spike_count_reduction_figure(data_folder, sessions, epoch, trial_type,
                                       dither_amount, bin_size, n_surrogates,
                                       window_length, trial_length, sorting_dead_times,
                                       trial_separation, fontsize, data_type='original'):
    """
    Create the complete spike count reduction figure.
    
    Parameters
    ----------
    data_folder : str
        Path to data folder
    sessions : list
        List of session names
    epoch : str
        Epoch name
    trial_type : str
        Trial type identifier
    dither_amount : float
        Amount of dithering for surrogates
    bin_size : pq.Quantity
        Size of bins for discretization
    n_surrogates : int
        Number of surrogates to generate
    window_length : int
        Window length for analysis
    trial_length : pq.Quantity
        Length of each trial
    sorting_dead_times : dict
        Dead times for each session
    trial_separation : pq.Quantity
        Separation between trials
    fontsize : int
        Font size for labels
    data_type : str, optional
        Type of data to analyze (default: 'original')
    """
    # Create figure and main grid
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    main_grid = gridspec.GridSpec(
        nrows=3, ncols=1, figure=fig, hspace=0.4, wspace=0.2,
        left=0.12, right=0.97, bottom=0.075, top=0.95,
        height_ratios=(1.25, 1., 0.75)
    )
    
    # Set random seed for reproducibility
    np.random.seed(0)
    
    # Panel A: Spike count reduction analysis
    column_spacing = 0.15
    panel_a_grid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=main_grid[0], wspace=column_spacing)
    
    # Load and process data for both sessions
    session_data = {}
    for i, session in enumerate(sessions):
        data_prefix = f'{data_type}_' if data_type != 'original' else ''
        filename = f'{data_folder}{session}/{data_prefix}{epoch}_{trial_type}.npy'
        session_data[session] = load_processed_spike_trains(filename)
    
    # Create subplots for each session
    session_names = ['Monkey N', 'Monkey L']
    for i, (session, session_name) in enumerate(zip(sessions, session_names)):
        # Create subplot grid
        subplot_grid = gridspec.GridSpecFromSubplotSpec(
            nrows=2, ncols=1, subplot_spec=panel_a_grid[i],
            hspace=0, height_ratios=[2, 1])
        
        ax_loss = fig.add_subplot(subplot_grid[0])
        ax_residuals = fig.add_subplot(subplot_grid[1], sharex=ax_loss)
        
        # Plot spike count reduction
        plot_spike_count_reduction(
            ax_loss, ax_residuals, session_data[session], dither_amount * pq.s,
            bin_size, n_surrogates, trial_length, window_length, fontsize)
        
        # Configure subplot appearance
        if i == 1:  # Right panel
            ax_loss.set_ylabel('')
            ax_loss.get_legend().remove()
            ax_residuals.set_ylabel('')
        
        plt.setp(ax_loss.get_xticklabels(), visible=False)
        ax_loss.set_title(session_name, fontsize=10)
    
    # Add panel label
    plt.figtext(x=0.055, y=0.95, s='A', fontsize=12, multialignment='center')
    
    # Panel B: ISI distributions
    panel_b_grid = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2, subplot_spec=main_grid[1],
        wspace=column_spacing, hspace=0.8)
    
    plt.figtext(x=0.055, y=0.57, s='B', fontsize=12, multialignment='center')
    
    # Example neuron indices for ISI plots
    example_neurons = {'N': [10, 27], 'L': [7, 8]}
    
    for i, (session, session_name) in enumerate(zip(sessions, session_names)):
        # Create subplot grid for ISI distributions
        isi_grid = gridspec.GridSpecFromSubplotSpec(
            nrows=2, ncols=1, subplot_spec=panel_b_grid[i], hspace=0)
        
        for j, neuron_idx in enumerate(example_neurons[session_name[-1]]):
            ax = fig.add_subplot(isi_grid[j])
            
            show_xlabel = (j == 1)  # Only bottom subplot gets x-label
            show_ylabel = (i == 0)  # Only left column gets y-label
            show_legend = (i == 0 and j == 0)  # Only top-left gets legend
            
            plot_isi_distribution(
                session_data[session][neuron_idx], ax, dither_amount,
                show_xlabel=show_xlabel, show_ylabel=show_ylabel,
                fontsize=fontsize, show_legend=show_legend)
            
            if not show_xlabel:
                plt.setp(ax.get_xticklabels(), visible=False)
    
    # Panel C: CV2 and dead time distributions
    panel_c_grid = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2, subplot_spec=main_grid[2],
        wspace=column_spacing, hspace=0.8)
    
    plt.figtext(x=0.055, y=0.25, s='C', fontsize=12, multialignment='center')
    
    # Create shared axes for consistent scaling
    cv2_axes = []
    dt_axes = []
    
    for i, (session, session_name) in enumerate(zip(sessions, session_names)):
        # Create subplot grid for statistics
        stats_grid = gridspec.GridSpecFromSubplotSpec(
            nrows=1, ncols=2, subplot_spec=panel_c_grid[i], wspace=0.35)
        
        # CV2 distribution
        if i == 0:
            ax_cv2 = fig.add_subplot(stats_grid[0])
            cv2_axes.append(ax_cv2)
        else:
            ax_cv2 = fig.add_subplot(stats_grid[0], sharex=cv2_axes[0], sharey=cv2_axes[0])
            cv2_axes.append(ax_cv2)
        
        show_ylabel = (i == 0)
        plot_cv2_distribution(
            session_data[session], ax_cv2, trial_length, trial_separation,
            fontsize=fontsize, show_ylabel=show_ylabel)
        
        # Dead time distribution
        if i == 0:
            ax_dt = fig.add_subplot(stats_grid[1])
            dt_axes.append(ax_dt)
        else:
            ax_dt = fig.add_subplot(stats_grid[1], sharex=dt_axes[0], sharey=dt_axes[0])
            dt_axes.append(ax_dt)
        
        plot_dead_time_distribution(
            session_data[session], ax_dt, sorting_dead_times[session_name[-1]],
            trial_separation, fontsize=fontsize, show_ylabel=show_ylabel)
    
    # Align y-labels for better appearance
    fig.align_ylabels()
    
    # Save figure
    plt.rcParams.update({'font.size': 10})
    fig.savefig('../plots/fig_spikeloss_r2gstats.png', dpi=300)
    fig.savefig('../plots/fig_spikeloss_r2gstats.pdf', dpi=300)


def main():
    """Main function to run the figure generation."""
    import yaml
    from yaml import Loader
    
    # Load configuration
    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    
    # Extract configuration parameters
    sessions = config['reduced_sessions']
    bin_size = config['binsize'] * pq.s
    window_length = config['winlen']
    dither_amount = config['dither']
    
    # Analysis parameters
    epoch = 'movement'
    trial_type = 'PGLF'
    trial_separation = 2 * window_length * bin_size
    sorting_dead_times = {
        'N': 1.26666 * pq.ms,
        'L': 1.6 * pq.ms
    }
    data_path = '../data/concatenated_spiketrains/'
    trial_length = 0.5 * pq.s
    n_surrogates = 100
    fontsize = DEFAULT_FONTSIZE
    
    # Prepare data types
    data_types = ['original']
    data_types.extend(config['processes'])
    
    # Fixed parameters for specific analysis
    fixed_data_type = 'original'
    fixed_epoch = 'movement'
    fixed_trial_type = 'PGHF'
    
    # Generate figures for specified conditions
    for current_epoch, current_trial_type, current_data_type in itertools.product(
            config['epochs'], config['trialtypes'], data_types):
        
        # Skip unless matching fixed conditions
        if not (current_data_type == fixed_data_type and 
                current_trial_type == fixed_trial_type and 
                current_epoch == fixed_epoch):
            continue
        
        # Determine data folder
        if current_data_type == 'original':
            data_folder = '../data/concatenated_spiketrains/'
        else:
            data_folder = f'../data/artificial_data/{current_data_type}/'
        
        # Create figure
        create_spike_count_reduction_figure(
            data_folder, sessions, current_epoch, current_trial_type,
            dither_amount, bin_size, n_surrogates, window_length,
            trial_length, sorting_dead_times, trial_separation,
            fontsize, data_type='original')


if __name__ == '__main__':
    main()