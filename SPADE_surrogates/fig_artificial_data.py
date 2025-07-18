"""
Enhanced script for creating publication-quality figures showing statistics of artificial data
and false positives when analyzing data with SPADE (Spike Pattern Detection and Evaluation).

This script generates a two-panel figure optimized for DIN A4 landscape format:
- Panel A: Comparison of spike train statistics (original vs artificial data)
- Panel B: False positive analysis across different surrogate methods

The script also provides detailed console output with comprehensive pattern analysis.
"""

import itertools
import os
import warnings
from collections import defaultdict

import numpy as np
import quantities as pq
import elephant

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab
from matplotlib import font_manager

# Configure modern plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'pdf.fonttype': 42,  # Avoid Type 3 fonts
    'ps.fonttype': 42,   # Avoid Type 3 fonts
    'svg.fonttype': 'none'
})

# Import project-specific modules
from fig_spikeloss_r2gstats import calculate_cv2 as get_cv2
from generate_artificial_data import estimate_rate_deadtime, create_st_list, estimate_deadtime
from rgutils import load_processed_spike_trains
from SPADE_surrogates.analyse_data_utils.filter_results import load_filtered_results

# ========================================================================================
# CONFIGURATION CONSTANTS
# ========================================================================================

# Layout parameters
XLABEL_PAD = 0.5
YLABEL_PAD = -0.3

# Data configuration
EXCLUDED_NEURONS = np.load('analysis_artificial_data/excluded_neurons.npy', allow_pickle=True).item()

# Surrogate method definitions
SURROGATE_METHODS = [
    'dither_spikes', 
    'dither_spikes_with_refractory_period',
    'joint_isi_dithering', 
    'isi_dithering',
    'trial_shifting', 
    'bin_shuffling'
]

SURROGATE_TAGS = ('UD', 'UDD', 'JISI-D', 'ISI-D', 'TR-SHIFT', 'WIN-SHUFF')

# Analysis sessions and processes
SESSIONS = ['i140703s001', 'l101210s001']
PROCESSES = ['ppd', 'gamma']

# Color scheme (matching original matplotlib defaults)
COLORS = {
    'original': '#1f77b4',      # Blue (C0)
    'dither_spikes': '#ff7f0e',  # Orange (C1) 
    'dither_spikes_with_refractory_period': '#2ca02c',  # Green (C2)
    'isi_dithering': '#d62728',  # Red (C4)
    'joint_isi_dithering': '#9467bd',  # Purple (C6)
    'trial_shifting': '#8c564b',  # Brown (C3)
    'bin_shuffling': '#e377c2'   # Pink (C5)
}

# Display labels for methods
LABELS = {
    'original': 'Original',
    'dither_spikes': 'UD',
    'dither_spikes_with_refractory_period': 'UDD',
    'isi_dithering': 'ISI-D',
    'joint_isi_dithering': 'JISI-D',
    'trial_shifting': 'TR-SHIFT',
    'bin_shuffling': 'WIN-SHUFF'
}

# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def print_section_header(title, char='=', width=80):
    """Print a formatted section header for console output."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")


def print_subsection_header(title, char='-', width=60):
    """Print a formatted subsection header for console output."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")


def safe_load_patterns(filepath):
    """
    Safely load pattern results with error handling.
    
    Parameters
    ----------
    filepath : str
        Path to the filtered results file
        
    Returns
    -------
    patterns : list
        List of detected patterns (empty if loading fails)
    success : bool
        Whether loading was successful
    """
    try:
        patterns, _, _, _, _ = load_filtered_results(filepath)
        return patterns, True
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return [], False


def safe_load_spiketrains(spiketrain_path, session):
    """
    Safely load spike trains with error handling and neuron exclusion.
    
    Parameters
    ----------
    spiketrain_path : str
        Path to the spike train file
    session : str
        Session identifier for neuron exclusion
        
    Returns
    -------
    spiketrains : list or None
        List of spike trains (None if loading fails)
    """
    try:
        spiketrains = load_processed_spike_trains(spiketrain_path)
        # Remove excluded neurons
        for neuron in EXCLUDED_NEURONS[session]:
            spiketrains.pop(int(neuron))
        return spiketrains
    except Exception as e:
        print(f"Could not load spiketrains for {spiketrain_path}: {e}")
        return None


# ========================================================================================
# DATA PROCESSING FUNCTIONS
# ========================================================================================

def cut_firing_rate_into_trials(rate, epoch_length, sep, sampling_period):
    """
    Convert concatenated firing rate into trial-by-trial format.
    
    Parameters
    ----------
    rate : neo.AnalogSignal
        Firing rate estimated for a single neuron on concatenated data
    epoch_length : pq.Quantity
        Length of each trial
    sep : pq.Quantity
        Buffer time between concatenated trials
    sampling_period : pq.Quantity
        Sampling period of the rate estimation
        
    Returns
    -------
    trials_rate : np.array
        Array of trial rates, where each item corresponds to one trial
    """
    len_rate = len(rate)
    start_index = 0
    trials_rate = []
    
    # Convert to consistent units
    epoch_length = epoch_length.rescale(pq.ms)
    sep = sep.rescale(pq.ms)
    sampling_period = sampling_period.rescale(pq.ms)
    
    # Calculate indices
    epoch_samples = int((epoch_length / sampling_period).simplified.magnitude)
    sep_samples = int((sep / sampling_period).simplified.magnitude)
    stop_index = start_index + epoch_samples
    
    # Extract trials
    while stop_index < len_rate:
        cut_rate = rate[start_index:stop_index]
        trials_rate.append(cut_rate)
        start_index += epoch_samples + sep_samples
        stop_index = start_index + epoch_samples
        
    return np.array(trials_rate)


def estimate_neuron_rate(spiketrain, binsize, epoch_length, winlen):
    """
    Estimate average firing rate for a neuron in concatenated spike train.
    
    Parameters
    ----------
    spiketrain : neo.Spiketrain
        Spike train for rate estimation
    binsize : pq.Quantity
        Bin size of the SPADE analysis
    epoch_length : pq.Quantity
        Length of epoch segmentation within one trial
    winlen : int
        Window length of the SPADE analysis
        
    Returns
    -------
    rate : float
        Average firing rate (spikes per second)
    """
    sep = 2 * binsize * winlen
    epoch_length = epoch_length.rescale(pq.s)
    sep = sep.rescale(pq.s)
    
    number_of_trials = int(spiketrain.t_stop.rescale(pq.s) / (epoch_length + sep))
    spike_count = len(spiketrain)
    rate = spike_count / (number_of_trials * epoch_length)
    
    return rate


# ========================================================================================
# DETAILED PATTERN ANALYSIS FUNCTIONS
# ========================================================================================

def print_detailed_pattern_stats(patterns, surrogate_method, process, session, 
                                behavioral_context, spiketrains):
    """
    Print comprehensive statistics about detected patterns.
    
    Parameters
    ----------
    patterns : list
        List of detected patterns
    surrogate_method : str
        Name of the surrogate method
    process : str
        Process type ('ppd' or 'gamma')
    session : str
        Session identifier
    behavioral_context : str
        Behavioral context identifier
    spiketrains : list
        List of spike trains
    """
    if len(patterns) == 0:
        return
    
    print_section_header("DETAILED PATTERN ANALYSIS")
    print(f"Process: {process.upper()}, Method: {surrogate_method.upper()}")
    print(f"Session: {session}, Context: {behavioral_context}")
    print_section_header("", char='=')
    
    # Basic pattern statistics
    pattern_sizes = [len(pattern['neurons']) for pattern in patterns]
    print(f"Total patterns found: {len(patterns)}")
    print(f"Pattern sizes: {pattern_sizes}")
    print(f"Average pattern size: {np.mean(pattern_sizes):.2f} ± {np.std(pattern_sizes):.2f}")
    print(f"Pattern size range: {min(pattern_sizes)} - {max(pattern_sizes)}")
    
    # Neuron participation analysis
    all_neurons = []
    for pattern in patterns:
        all_neurons.extend(pattern['neurons'])
    
    unique_neurons = np.unique(all_neurons)
    print(f"\nNeuron participation:")
    print(f"  Unique neurons in patterns: {len(unique_neurons)}")
    print(f"  Total neuron participations: {len(all_neurons)}")
    
    # Participation frequency analysis
    neuron_counts = defaultdict(int)
    for neuron in all_neurons:
        neuron_counts[neuron] += 1
    
    participation_freq = list(neuron_counts.values())
    print(f"  Average participations per neuron: {np.mean(participation_freq):.2f}")
    
    # Most active neurons
    sorted_neurons = sorted(neuron_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 5 most active neurons: {sorted_neurons[:5]}")
    
    # Temporal analysis (if available)
    if patterns and 'times' in patterns[0]:
        all_times = []
        for pattern in patterns:
            all_times.extend(pattern['times'])
        
        print(f"\nTemporal analysis:")
        print(f"  Pattern occurrence times (first 10): {all_times[:10]}")
        print(f"  Time range: {min(all_times):.3f} - {max(all_times):.3f}")
    
    # Statistical significance analysis (if available)
    if patterns and 'pvalue' in patterns[0]:
        pvalues = [pattern['pvalue'] for pattern in patterns]
        print(f"\nStatistical significance:")
        print(f"  P-value range: {min(pvalues):.6f} - {max(pvalues):.6f}")
        print(f"  Mean p-value: {np.mean(pvalues):.6f}")
        
        # Count significant patterns at different thresholds
        significant_001 = sum(1 for p in pvalues if p < 0.001)
        significant_01 = sum(1 for p in pvalues if p < 0.01)
        significant_05 = sum(1 for p in pvalues if p < 0.05)
        
        print(f"  Patterns with p < 0.001: {significant_001}")
        print(f"  Patterns with p < 0.01: {significant_01}")
        print(f"  Patterns with p < 0.05: {significant_05}")
    
    print_section_header("", char='=')


# ========================================================================================
# PLOTTING FUNCTIONS - PANEL A (STATISTICS COMPARISON)
# ========================================================================================

def plot_trial_firing_rate(ax, sts, gamma, ppd, neuron, max_refractory, sep,
                           epoch_length, sampling_period, fontsize):
    """
    Plot average firing rate across trials for different data types.
    
    Shows firing rate profiles with confidence bands for original, PPD, and gamma data.
    """
    # Extract spike trains for the selected neuron
    st = sts[neuron]
    gamma_st = gamma[neuron]
    ppd_st = ppd[neuron]
    
    # Estimate rates and refractory periods
    rate_dict = {}
    list_st = [st, gamma_st, ppd_st]
    processes = ['original', 'gamma', 'ppd']
    
    for order, process in enumerate(processes):
        rate, refractory_period, _ = estimate_rate_deadtime(
            list_st[order],
            max_refractory=max_refractory,
            sampling_period=sampling_period,
            sep=sep
        )
        rate_dict[process] = rate
    
    # Convert rates to trial format
    trial_rates = {}
    for process in processes:
        trial_rates[process] = cut_firing_rate_into_trials(
            rate_dict[process], epoch_length, sep, sampling_period
        )
    
    # Time axis
    x = np.arange(0, int(epoch_length.rescale(pq.ms).magnitude /
                         sampling_period.rescale(pq.ms).magnitude))
    
    # Plot each data type with confidence bands
    plot_configs = [
        ('original', COLORS['original'], 'Original'),
        ('ppd', '#ff7f0e', 'PPD'),
        ('gamma', '#2ca02c', 'Gamma')
    ]
    
    for process, color, label in plot_configs:
        mean_rate = np.squeeze(np.mean(trial_rates[process], axis=0))
        std_rate = np.squeeze(np.std(trial_rates[process], axis=0))
        
        # Plot mean line
        ax.plot(mean_rate, label=label, color=color, linewidth=2, alpha=0.8)
        
        # Plot confidence band
        ax.fill_between(x, mean_rate - std_rate, mean_rate + std_rate,
                       alpha=0.2, color=color)
    
    # Set plot limits based on data range
    all_means = [np.mean(trial_rates[p], axis=0) for p in processes]
    all_stds = [np.std(trial_rates[p], axis=0) for p in processes]
    
    y_min = min(np.min(m - s) for m, s in zip(all_means, all_stds)) - 10
    y_max = max(np.max(m + s) for m, s in zip(all_means, all_stds)) + 10
    ax.set_ylim(y_min, y_max)
    
    # Format axes and labels
    ax.set_xlabel('Time (ms)', fontsize=fontsize, labelpad=XLABEL_PAD)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=fontsize)
    ax.set_title('Single Unit Average FR', fontsize=fontsize, pad=10)
    
    # Add legend only to the first plot
    # ax.legend(frameon=True, fancybox=False, shadow=False, loc='upper left', 
    #           bbox_to_anchor=(0, 1))
    
    # Convert x-axis to milliseconds
    xticks = ax.get_xticks().tolist()
    rescaled_xticks = [int(int(lab) / 10) for lab in xticks]
    ax.set_xticklabels(rescaled_xticks)


def plot_dead_time_distribution(ax, sts, gamma, ppd, max_refractory, fontsize):
    """
    Plot dead time distribution for all neurons across data types.
    
    Uses step histograms to show overlapping distributions clearly.
    """
    processes = {'Original': sts, 'PPD': ppd, 'Gamma': gamma}
    colors = {'Original': COLORS['original'], 'PPD': '#ff7f0e', 'Gamma': '#2ca02c'}
    
    # Calculate dead times for each neuron and process
    dead_time_dict = {'Original': [], 'PPD': [], 'Gamma': []}
    
    for neuron in range(len(sts)):
        for key, dead_time_list in dead_time_dict.items():
            dead_time = estimate_deadtime(
                processes[key][neuron],
                max_dead_time=max_refractory
            )
            dead_time_list.append(dead_time.magnitude * 1000)  # Convert to ms
    
    # Create histogram
    bins = np.arange(0, 4, 0.1)
    for key, dead_times in dead_time_dict.items():
        ax.hist(dead_times, bins=bins, alpha=1.0, label=key, 
                histtype='step', color=colors[key], linewidth=2)
    
    # Format plot
    ax.set_title('Dead Time Distribution', fontsize=fontsize, pad=10)
    ax.set_ylabel('Count', fontsize=fontsize, labelpad=YLABEL_PAD)
    ax.set_xlabel('Dead Time (ms)', fontsize=fontsize, labelpad=XLABEL_PAD)
    
    # Set y-limit based on data
    max_count = max([max(np.histogram(dt, bins)[0]) for dt in dead_time_dict.values()])
    ax.set_ylim([0, max_count + 5])


def plot_isi_distribution(ax, sts, gamma, ppd, neuron, fontsize):
    """
    Plot inter-spike interval distribution for a single neuron.
    
    Shows ISI distributions for original, PPD, and gamma data.
    """
    # Extract spike trains
    spike_trains = {
        'Original': sts[neuron],
        'PPD': ppd[neuron], 
        'Gamma': gamma[neuron]
    }
    
    colors = {'Original': COLORS['original'], 'PPD': '#ff7f0e', 'Gamma': '#2ca02c'}
    
    # Calculate ISI for each data type
    isi_dict = {}
    for key, st in spike_trains.items():
        isi = elephant.statistics.isi(st)
        isi_dict[key] = isi
    
    # Create histogram
    bins = np.arange(0, 0.3, 0.01)
    for key, isi in isi_dict.items():
        ax.hist(isi, bins, alpha=1.0, label=key, 
                histtype='step', color=colors[key], linewidth=2)
    
    # Format plot
    ax.set_xlabel('ISI (s)', fontsize=fontsize, labelpad=XLABEL_PAD)
    ax.set_ylabel('Count', fontsize=fontsize, labelpad=YLABEL_PAD)
    ax.set_title('Single Unit ISI Distribution', fontsize=fontsize, pad=10)
    
    # Add legend to A2 (upper right corner)
    ax.legend(frameon=True, fancybox=False, shadow=False, loc='upper right')


def plot_cv2_distribution(ax, sts, gamma, ppd, sep, fontsize):
    """
    Plot coefficient of variation (CV2) distribution for all neurons.
    
    CV2 is calculated trial-wise and then aggregated across neurons.
    """
    cv2_dict = {'Original': [], 'PPD': [], 'Gamma': []}
    colors = {'Original': COLORS['original'], 'PPD': '#ff7f0e', 'Gamma': '#2ca02c'}
    
    # Calculate CV2 for each process
    for process, conc_spiketrains in zip(
            ('Original', 'PPD', 'Gamma'),
            (sts, ppd, gamma)):
        
        cv2_list = []
        for conc_st in conc_spiketrains:
            # Split into trials
            trial_list = create_st_list(conc_st, sep=sep)
            
            # Calculate ISIs for trials with enough spikes
            isis = [np.diff(st.magnitude) for st in trial_list if len(st) > 1]
            
            # Calculate CV2
            cv2 = get_cv2(isis)
            cv2_list.append(cv2)
        
        # Filter out NaN values
        cv2_array = np.array(cv2_list)[~np.isnan(np.array(cv2_list))]
        cv2_dict[process] = cv2_array
    
    # Create histogram
    bins = np.arange(0, 1.5, 0.1)
    for key, cv2 in cv2_dict.items():
        ax.hist(cv2, bins, alpha=1.0, label=key, 
                histtype='step', color=colors[key], linewidth=2)
    
    # Format plot
    ax.set_title('CV2 Distribution', fontsize=fontsize, pad=10)
    ax.set_ylabel('Count', fontsize=fontsize, labelpad=YLABEL_PAD)
    ax.set_xlabel('CV2', fontsize=fontsize, labelpad=XLABEL_PAD)


def create_panel_a(axes, sts, gamma, ppd, neuron, max_refractory, sep,
                   epoch_length, sampling_period, fontsize):
    """
    Create Panel A: Statistical comparison of original vs artificial data.
    
    Contains four subplots:
    1. Firing rate profile
    2. ISI distribution  
    3. CV2 distribution
    4. Dead time distribution
    """
    ax1, ax2, ax3, ax4 = axes
    
    # Generate each subplot
    plot_trial_firing_rate(
        ax1, sts=sts, gamma=gamma, ppd=ppd, neuron=neuron,
        max_refractory=max_refractory, sep=sep, epoch_length=epoch_length,
        sampling_period=sampling_period, fontsize=fontsize
    )
    ax1.set_ylim([0, 50])
    
    plot_isi_distribution(
        ax2, sts=sts, gamma=gamma, ppd=ppd, neuron=neuron, fontsize=fontsize
    )
    
    plot_cv2_distribution(
        ax3, sts=sts, gamma=gamma, ppd=ppd, sep=sep, fontsize=fontsize
    )
    
    plot_dead_time_distribution(
        ax4, sts=sts, gamma=gamma, ppd=ppd, 
        max_refractory=max_refractory, fontsize=fontsize
    )


# ========================================================================================
# FALSE POSITIVE ANALYSIS FUNCTIONS
# ========================================================================================

def calculate_false_positives(surrogate_methods, sessions):
    """
    Calculate false positives with comprehensive statistical analysis.
    
    Returns detailed statistics for each surrogate method and process type.
    """
    print_section_header("COMPREHENSIVE FALSE POSITIVE ANALYSIS")
    
    # Initialize result containers
    fps_results = {'ppd': {}, 'gamma': {}}
    detailed_stats = {'ppd': {}, 'gamma': {}}
    
    # Analyze each process type
    for process in ['ppd', 'gamma']:
        print_subsection_header(f"{process.upper()} ANALYSIS")
        
        for surrogate in surrogate_methods:
            # Initialize tracking variables
            fps_results[process][surrogate] = 0
            detailed_stats[process][surrogate] = {
                'total_patterns': 0,
                'datasets_with_patterns': 0,
                'pattern_sizes': [],
                'sessions_breakdown': {}
            }
            
            pattern_counts = []
            
            # Process each session
            for session in sessions:
                detailed_stats[process][surrogate]['sessions_breakdown'][session] = 0
                
                # Find result directories
                result_base_path = f'../results/artificial_data/{surrogate}/{process}/{session}/'
                
                if not os.path.exists(result_base_path):
                    print(f"Warning: Path does not exist: {result_base_path}")
                    continue
                
                folder_results = [f.path for f in os.scandir(result_base_path) if f.is_dir()]
                
                # Process each result directory
                for result_dir in folder_results:
                    filepath = os.path.join(result_dir, 'filtered_res.npy')
                    
                    patterns, success = safe_load_patterns(filepath)
                    
                    if success and len(patterns) > 0:
                        # Update counters
                        fps_results[process][surrogate] += len(patterns)
                        pattern_counts.append(len(patterns))
                        detailed_stats[process][surrogate]['total_patterns'] += len(patterns)
                        detailed_stats[process][surrogate]['datasets_with_patterns'] += 1
                        detailed_stats[process][surrogate]['sessions_breakdown'][session] += len(patterns)
                        
                        # Store pattern sizes
                        pattern_sizes = [len(pattern['neurons']) for pattern in patterns]
                        detailed_stats[process][surrogate]['pattern_sizes'].extend(pattern_sizes)
                        
                        # Generate detailed analysis if requested
                        behavioral_context = os.path.basename(result_dir)
                        spiketrain_path = f'../data/artificial_data/{process}/{session}/{process}_{behavioral_context}.npy'
                        spiketrains = safe_load_spiketrains(spiketrain_path, session)
                        
                        if spiketrains is not None:
                            print_detailed_pattern_stats(
                                patterns, surrogate, process, session, 
                                behavioral_context, spiketrains
                            )
            
            # Print summary for this surrogate method
            print_subsection_header(f"{process.upper()} - {surrogate.upper()}")
            print(f"Total FP patterns: {np.sum(pattern_counts)}")
            print(f"Datasets with patterns: {len(pattern_counts)}")
            print(f"Family-wise error rate: {len(pattern_counts)/48:.3f}")
            
            if pattern_counts:
                print(f"Average patterns per dataset with patterns: {np.mean(pattern_counts):.2f}")
            
            if detailed_stats[process][surrogate]['pattern_sizes']:
                sizes = detailed_stats[process][surrogate]['pattern_sizes']
                print(f"Average pattern size: {np.mean(sizes):.2f}")
                print(f"Pattern size range: {min(sizes)}-{max(sizes)}")
            
            # Session breakdown
            for session, count in detailed_stats[process][surrogate]['sessions_breakdown'].items():
                print(f"  {session}: {count} patterns")
    
    # Print comparison summary
    print_section_header("SUMMARY COMPARISON")
    print(f"{'Method':<20} {'PPD FPs':<10} {'Gamma FPs':<12} {'PPD FWER':<10} {'Gamma FWER':<12}")
    print(f"{'-'*70}")
    
    for surrogate in surrogate_methods:
        ppd_fwer = detailed_stats['ppd'][surrogate]['datasets_with_patterns'] / 48
        gamma_fwer = detailed_stats['gamma'][surrogate]['datasets_with_patterns'] / 48
        print(f"{surrogate:<20} {fps_results['ppd'][surrogate]:<10} "
              f"{fps_results['gamma'][surrogate]:<12} {ppd_fwer:<10.3f} {gamma_fwer:<12.3f}")
    
    return fps_results['ppd'], fps_results['gamma']


def calculate_firing_rate_stats(surrogate_methods, sessions, binsize, winlen, epoch_length):
    """
    Calculate firing rate statistics for neurons involved in false positive patterns.
    
    Returns firing rates organized by process and surrogate method.
    """
    neurons_fr = {'gamma': {}, 'ppd': {}}
    
    for process, surrogate in itertools.product(['gamma', 'ppd'], surrogate_methods):
        neurons_fr[process][surrogate] = []
        
        for session in sessions:
            result_base_path = f'../results/artificial_data/{surrogate}/{process}/{session}/'
            
            if not os.path.exists(result_base_path):
                continue
                
            folder_results = [f.path for f in os.scandir(result_base_path) if f.is_dir()]
            
            for result_dir in folder_results:
                filepath = os.path.join(result_dir, 'filtered_res.npy')
                patterns, success = safe_load_patterns(filepath)
                
                if not success:
                    continue
                
                # Load corresponding spike train data
                behavioral_context = os.path.basename(result_dir)
                spiketrain_path = f'../data/artificial_data/{process}/{session}/{process}_{behavioral_context}.npy'
                spiketrains = safe_load_spiketrains(spiketrain_path, session)
                
                if spiketrains is None:
                    continue
                
                # Extract firing rates for neurons in patterns
                for pattern in patterns:
                    for neuron in pattern['neurons']:
                        rate = estimate_neuron_rate(
                            spiketrains[int(neuron)],
                            binsize=binsize,
                            winlen=winlen,
                            epoch_length=epoch_length
                        )
                        neurons_fr[process][surrogate].append(rate.magnitude)
        
        neurons_fr[process][surrogate] = np.array(neurons_fr[process][surrogate]).flatten()
    
    return neurons_fr


# ========================================================================================
# PLOTTING FUNCTIONS - PANEL B (FALSE POSITIVE ANALYSIS)
# ========================================================================================

def plot_false_positive_bars(ax_num_fps, index, process, sessions, surrogate_methods, tick_size):
    """
    Create bar plot showing false positives per dataset for each surrogate method.
    
    Includes value labels on bars and proper formatting.
    """
    # Get false positive data
    fps_data = calculate_false_positives(sessions=sessions, surrogate_methods=surrogate_methods)
    fps = fps_data[index]  # Select PPD (0) or Gamma (1) results
    
    number_datasets = 48
    bars = []
    
    # Create bars for each surrogate method
    for index_surr, surrogate in enumerate(surrogate_methods):
        bar = ax_num_fps.bar(
            index_surr,
            fps[surrogate] / number_datasets,
            width=0.6,
            color=COLORS[surrogate],
            label=LABELS[surrogate], 
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
        bars.append(bar)
    
    # Add value labels on bars
    for i, (surrogate, bar) in enumerate(zip(surrogate_methods, bars)):
        height = bar[0].get_height()
        ax_num_fps.text(
            bar[0].get_x() + bar[0].get_width()/2., 
            height + 0.01,
            f'{fps[surrogate]}',
            ha='center', va='bottom', fontsize=tick_size-1
        )

    # Format axes
    ax_num_fps.set_xticks(range(len(surrogate_methods)))
    ax_num_fps.set_xticklabels(
        [LABELS[surrogate] for surrogate in surrogate_methods], 
        rotation=45, ha='right'
    )
    ax_num_fps.tick_params(axis='both', which='major', labelsize=tick_size)
    ax_num_fps.grid(axis='y', alpha=0.3)
    
    # Return the fps values to use for consistent y-axis scaling
    return [fps[surrogate] / number_datasets for surrogate in surrogate_methods]


# ========================================================================================
# MAIN FIGURE CREATION FUNCTION
# ========================================================================================

def create_artificial_data_figure(sts, gamma, ppd, neuron, max_refractory,
                                 sep, sampling_period, epoch_length,
                                 sessions, surrogate_methods, processes,
                                 label_size, tick_size):
    """
    Create the complete figure with modern styling optimized for DIN A4 landscape format.
    
    Generates a two-panel figure:
    - Panel A: Statistical comparison (4 subplots)
    - Panel B: False positive analysis (2 subplots)
    
    Parameters
    ----------
    sts : list
        Original spike trains
    gamma : list
        Gamma model spike trains
    ppd : list
        PPD model spike trains
    neuron : int
        Index of neuron for single-unit analysis
    max_refractory : pq.Quantity
        Maximum refractory period
    sep : pq.Quantity
        Separation time between trials
    sampling_period : pq.Quantity
        Sampling period
    epoch_length : pq.Quantity
        Trial duration
    sessions : list
        List of session identifiers
    surrogate_methods : list
        List of surrogate method names
    processes : list
        List of process types ['ppd', 'gamma']
    label_size : int
        Font size for labels
    tick_size : int
        Font size for tick labels
    """
    
    print_section_header("GENERATING PUBLICATION-QUALITY FIGURE")
    
    # Create figure sized for DIN A4 landscape top half with margins
    # DIN A4 landscape: 297mm x 210mm, top half: 297mm x 105mm
    # Convert to inches: 11.69" x 4.13"
    fig = plt.figure(figsize=(11.69, 4.5), dpi=300, facecolor='white')
    
    # Adjust margins for proper DIN A4 formatting
    plt.subplots_adjust(
        top=0.88, bottom=0.15, left=0.08, right=0.95, 
        hspace=0.45, wspace=0.25
    )
    
    # Create main grid layout (2 rows only)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 0.8], hspace=0.45)

    # ==================================================================================
    # PANEL A: STATISTICS COMPARISON
    # ==================================================================================
    
    print_subsection_header("GENERATING PANEL A: STATISTICS COMPARISON")
    
    # Create subplot grid for Panel A (4 columns)
    gs0 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=4, subplot_spec=gs[0], wspace=0.3
    )
    
    # Create subplots
    ax01 = fig.add_subplot(gs0[0])  # Firing rate
    ax02 = fig.add_subplot(gs0[1])  # ISI distribution
    ax03 = fig.add_subplot(gs0[2])  # CV2 distribution
    ax04 = fig.add_subplot(gs0[3])  # Dead time distribution
    axes_panel_a = (ax01, ax02, ax03, ax04)
    
    # Generate Panel A plots
    create_panel_a(
        axes=axes_panel_a, sts=sts, gamma=gamma, ppd=ppd, neuron=neuron,
        max_refractory=max_refractory, sep=sep, fontsize=label_size,
        sampling_period=sampling_period, epoch_length=epoch_length
    )
    
    # Set specific limits for better visualization
    ax03.set_xlim(-0.05, 1.75)  # CV2 range
    ax02.set_xlim(-0.005, 0.2)  # ISI range
    ax01.set_ylim(0, 40)  # Firing rate range

    # ==================================================================================
    # PANEL B: FALSE POSITIVE ANALYSIS
    # ==================================================================================
    
    print_subsection_header("GENERATING PANEL B: FALSE POSITIVES ANALYSIS")
    
    # Create subplot grid for Panel B (2 columns)
    gs1 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2, subplot_spec=gs[1], wspace=0.3
    )
    
    ax_b_left = fig.add_subplot(gs1[0])   # PPD results
    ax_b_right = fig.add_subplot(gs1[1])  # Gamma results

    # Generate false positive plots for both processes and collect y-axis data
    all_fps_values = []
    for index, (process, ax_num_fps) in enumerate(zip(processes, (ax_b_left, ax_b_right))):
        fps_values = plot_false_positive_bars(
            ax_num_fps=ax_num_fps,
            index=index,
            process=process,
            sessions=sessions,
            surrogate_methods=surrogate_methods,
            tick_size=tick_size
        )
        all_fps_values.extend(fps_values)

    # Set consistent y-axis limits for both plots
    max_fps = max(all_fps_values) if all_fps_values else 1
    y_max = max_fps * 1.1  # Add 10% padding
    
    # Format plots with consistent y-axis
    for index, (process, ax_num_fps) in enumerate(zip(processes, (ax_b_left, ax_b_right))):
        ax_num_fps.set_ylabel('FPs per Dataset', size=label_size, labelpad=5)
        ax_num_fps.set_ylim(0, y_max)
        
        if process == 'ppd':
            ax_num_fps.set_title('PPD Model', fontsize=label_size+2, pad=-5,  # Increased pad
                                fontweight='bold')
        else:
            ax_num_fps.set_title('Gamma Model', fontsize=label_size+2, pad=-5,  # Increased pad
                                fontweight='bold')
            # No legend needed anymore

    # Add panel describtions
    fig = ax01.get_figure()
    # Calculate the absolute position for both labels
    label_x_fig = ax01.get_position().x0 - 0.02  # Adjust offset as needed

    fig.text(label_x_fig, ax01.get_position().y1 + 0.02, 'A', 
            fontsize=18, fontweight='bold', va='bottom', ha='center')
    fig.text(label_x_fig, ax_b_left.get_position().y1 + 0.02, 'B', 
            fontsize=18, fontweight='bold', va='bottom', ha='center')

    # ==================================================================================
    # SAVE FIGURE
    # ==================================================================================
    
    print_subsection_header("SAVING FIGURE")
    
    # Ensure output directory exists
    output_dir = '../figures/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Save in multiple high-quality formats
    base_filename = 'fig_artificial_data_modern'
    
    formats = [
        ('pdf', 'PDF (publication quality)'),
        ('png', 'PNG (presentations)'),
        ('svg', 'SVG (vector graphics)')
    ]
    
    saved_files = []
    for fmt, description in formats:
        filepath = os.path.join(output_dir, f'{base_filename}.{fmt}')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        saved_files.append(f"- {os.path.basename(filepath)} ({description})")
        print(f"Saved: {filepath}")
    
    print_section_header("FIGURE GENERATION COMPLETE")
    print("Files saved:")
    for file_info in saved_files:
        print(file_info)
    print_section_header("", char='=')
    
    plt.show()


# ========================================================================================
# ANALYSIS SUMMARY AND REPORTING
# ========================================================================================

def print_analysis_summary(surrogate_methods, sessions, processes):
    """
    Print comprehensive summary of the analysis configuration and outputs.
    """
    print_section_header("COMPREHENSIVE ANALYSIS SUMMARY")
    
    print("Analysis Configuration:")
    print(f"  Sessions analyzed: {sessions}")
    print(f"  Processes: {processes}")
    print(f"  Surrogate methods: {len(surrogate_methods)}")
    print(f"  Total datasets: {len(sessions) * 6 * 4}")  # sessions × epochs × trial types
    
    print("\nSurrogate Methods:")
    for i, method in enumerate(surrogate_methods):
        print(f"  {i+1:2d}. {method} ({LABELS[method]})")
    
    print("\nStatistical Measures Analyzed:")
    measures = [
        "Firing rate profiles across trials",
        "Inter-spike interval (ISI) distributions", 
        "Coefficient of variation (CV2)",
        "Dead time distributions",
        "False positive detection rates",
        "Neuron participation patterns"
    ]
    for measure in measures:
        print(f"  - {measure}")
    
    print("\nOutput Specifications:")
    specifications = [
        "High-resolution PDF (publication quality)",
        "PNG format (presentations)", 
        "SVG format (vector graphics)",
        "DIN A4 landscape top-half sizing",
        "Type 42 fonts (non-embedded)",
        "Modern color scheme with accessibility"
    ]
    for spec in specifications:
        print(f"  - {spec}")
    
    print_section_header("", char='=')


# ========================================================================================
# MAIN EXECUTION
# ========================================================================================

if __name__ == "__main__":
    print_section_header("MODERN ARTIFICIAL DATA ANALYSIS SCRIPT", width=100)
    print("Enhanced with detailed pattern statistics and modern visualization")
    print("Optimized for publication-quality figures in DIN A4 landscape format")
    print_section_header("", char='=', width=100)
    
    # ==================================================================================
    # PARAMETER CONFIGURATION
    # ==================================================================================
    
    # Analysis parameters
    binsize = 5 * pq.ms
    winlen = 12
    sep = 2 * winlen * binsize
    max_refractory = 4 * pq.ms
    epoch_length = 500 * pq.ms
    sampling_period = 0.1 * pq.ms
    neuron = 39  # Selected neuron for single-unit analysis
    
    # Font and layout parameters
    label_size = 12
    title_size = 14
    tick_size = 11

    print("Analysis Parameters:")
    parameter_info = [
        f"Bin size: {binsize}",
        f"Window length: {winlen}",
        f"Separation time: {sep}",
        f"Max refractory period: {max_refractory}",
        f"Epoch length: {epoch_length}",
        f"Sampling period: {sampling_period}",
        f"Selected neuron: {neuron}"
    ]
    for info in parameter_info:
        print(f"  {info}")

    # Print comprehensive analysis summary
    print_analysis_summary(SURROGATE_METHODS, SESSIONS, PROCESSES)

    # ==================================================================================
    # DATA LOADING
    # ==================================================================================
    
    # Data configuration for visualization
    session = 'i140703s001'
    epoch = 'movement'
    trialtype = 'PGHF'
    
    print_subsection_header("LOADING DATA FOR VISUALIZATION")
    print(f"Session: {session}")
    print(f"Epoch: {epoch}")
    print(f"Trial type: {trialtype}")
    
    try:
        # Load spike train data
        data_paths = {
            'original': f'../data/concatenated_spiketrains/{session}/{epoch}_{trialtype}.npy',
            'gamma': f'../data/artificial_data/gamma/{session}/gamma_{epoch}_{trialtype}.npy',
            'ppd': f'../data/artificial_data/ppd/{session}/ppd_{epoch}_{trialtype}.npy'
        }
        
        data = {}
        for data_type, path in data_paths.items():
            if os.path.exists(path):
                data[data_type] = load_processed_spike_trains(path)
                print(f"  {data_type.capitalize()} data: {len(data[data_type])} neurons")
            else:
                raise FileNotFoundError(f"Data file not found: {path}")
        
        # ==================================================================================
        # FIGURE GENERATION
        # ==================================================================================
        
        # Generate the enhanced figure
        create_artificial_data_figure(
            sts=data['original'],
            gamma=data['gamma'],
            ppd=data['ppd'],
            neuron=neuron,
            max_refractory=max_refractory,
            sep=sep,
            sampling_period=sampling_period,
            epoch_length=epoch_length,
            sessions=SESSIONS,
            surrogate_methods=SURROGATE_METHODS,
            processes=PROCESSES,
            label_size=label_size,
            tick_size=tick_size
        )
            
    except Exception as e:
        print_section_header("ERROR IN DATA LOADING OR PROCESSING", char='!', width=80)
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check that all data files exist in the specified paths")
        print("2. Verify that the required modules are properly imported")
        print("3. Ensure sufficient memory for large datasets")
        print("4. Check file permissions for the output directory")
        print_section_header("", char='!', width=80)