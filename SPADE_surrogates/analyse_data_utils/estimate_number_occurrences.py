"""
Improved occurrence estimation for SPADE analysis.

This module estimates the minimum number of occurrences for patterns of different sizes
using an improved non-stationary Poisson model with geometric mean rate estimation.
"""

import os
import numpy as np
from scipy.stats import poisson
from scipy.special import binom as binom_coeff
from scipy.stats.mstats import gmean

import quantities as pq
import yaml
from yaml import Loader

from SPADE_surrogates.analyse_data_utils import spade_utils as utils
from SPADE_surrogates.rgutils import save_processed_data, load_processed_spike_trains

def create_rate_dict(session, epoch, trialtype, rates_path, binsize, process='original'):
    """
    Create rate dictionary for estimating expected pattern occurrences.

    Parameters
    ----------
    session : str
        Recording session identifier
    epoch : str
        Epoch of the trial (e.g., 'start', 'cue', 'movement')
    trialtype : str
        Trial type (e.g., 'PGHF', 'PGLF')
    rates_path : str
        Path where to store the rate profiles
    binsize : pq.Quantity
        Bin size for the SPADE analysis
    process : str, optional
        Point process model being analyzed ('original', 'ppd', 'gamma')
        Default: 'original'

    Returns
    -------
    dict
        Dictionary containing:
        - 'rates': sorted rates in descending order
        - 'n_bins': total number of bins
        - 'rates_ordered_by_neuron': rates ordered by neuron ID
    """
    # Define data paths based on process type
    data_paths = {
        'original': '../../data/concatenated_spiketrains/',
        'ppd': '../../data/artificial_data/ppd/',
        'gamma': '../../data/artificial_data/gamma/'
    }
    
    if process not in data_paths:
        raise ValueError(f"Process must be one of {list(data_paths.keys())}, got '{process}'")
    
    data_path = data_paths[process]
    
    # Load spike train data
    if process == 'original':
        filename = f'{data_path}{session}/{epoch}_{trialtype}.npy'
    else:
        filename = f'{data_path}{session}/{process}_{epoch}_{trialtype}.npy'
    
    spike_trains = load_processed_spike_trains(filename)
    
    # Calculate basic statistics
    data_duration = spike_trains[0].t_stop
    n_bins = int(data_duration / binsize)
    
    # Compute firing rates for each neuron
    firing_rates = []
    for spike_train in spike_trains:
        spike_count = len(spike_train)
        firing_rate = spike_count / float(data_duration)
        firing_rates.append(firing_rate)
    
    # Create rate dictionary
    sorted_rates = sorted(firing_rates, reverse=True)  # Descending order
    rates_dict = {
        'rates': sorted_rates,
        'n_bins': n_bins,
        'rates_ordered_by_neuron': firing_rates
    }
    
    # Ensure output directory exists
    _create_directory_if_needed(rates_path)
    
    # Save rates dictionary
    np.save(os.path.join(rates_path, 'rates.npy'), rates_dict)
    
    return rates_dict


def _create_directory_if_needed(path):
    """Create directory structure if it doesn't exist."""
    current_path = './'
    for folder in utils.split_path(path):
        current_path = os.path.join(current_path, folder)
        utils.mkdirp(current_path)


def _get_non_stationarity_factor(epoch, process='original'):
    """
    Get non-stationarity factor based on epoch and process type.
    
    Different epochs have different levels of firing rate variability:
    - Start/Wait: Stationary (factor = 1.0)
    - Movement: Highly variable (factor = 2.5)
    - Others: Moderately variable (factor = 1.5)
    
    Parameters
    ----------
    epoch : str
        Epoch name
    process : str
        Process type
        
    Returns
    -------
    float
        Non-stationarity factor
    """
    stationary_conditions = (
        epoch in ("start", "wait") or 
        process in ('homogeneous_poisson', 'stationary_poisson', 'stationary_gamma')
    )
    
    if stationary_conditions:
        return 1.0
    elif epoch == "movement":
        return 2.5
    else:
        return 1.5


def _calculate_min_occurrence_threshold(rate_ref, binsize, pattern_size, winlen, 
                                       n_neurons, n_bins, percentile_poiss, 
                                       non_stationarity=1.0):
    """
    Calculate minimum occurrence threshold using improved non-stationary Poisson model.
    
    This method accounts for non-stationary firing patterns by modeling high and low
    rate periods, uses proper Poisson probabilities, and includes multiple comparisons
    correction.
    
    Parameters
    ----------
    rate_ref : float
        Reference firing rate (geometric mean of top neurons)
    binsize : float
        Bin size in seconds
    pattern_size : int
        Number of spikes in pattern
    winlen : int
        Window length for pattern detection
    n_neurons : int
        Number of neurons with similar firing rates
    n_bins : int
        Total number of bins in the data
    percentile_poiss : float
        Percentile for significance threshold (0-100)
    non_stationarity : float
        Non-stationarity factor (1.0 = stationary)
        
    Returns
    -------
    int
        Minimum occurrence threshold
    """
    # Model non-stationary behavior with high and low rate periods
    normalization_factor = 2 / (non_stationarity + 1)
    rate_high = non_stationarity * normalization_factor * rate_ref
    rate_low = normalization_factor * rate_ref
    
    # Use proper Poisson probability: P(spike) = 1 - exp(-rate * dt)
    prob_high = (1 - np.exp(-rate_high * binsize)) ** pattern_size
    prob_low = (1 - np.exp(-rate_low * binsize)) ** pattern_size
    
    # Calculate number of possible pattern combinations
    # Accounts for temporal arrangements and neuron combinations
    temporal_combinations = (pattern_size * (pattern_size - 1)) // 2 * winlen ** (pattern_size - 2)
    neuron_combinations = binom_coeff(n_neurons, pattern_size)
    total_combinations = temporal_combinations * neuron_combinations
    
    # Use average probability for threshold calculation
    mean_probability = (prob_high + prob_low) / 2
    expected_count = mean_probability * n_bins
    
    # Calculate significance threshold using Poisson distribution
    # with multiple comparisons correction
    if expected_count <= 0:
        return 1
    
    # Define range around expected count for threshold search
    search_range = max(5 * np.sqrt(expected_count), 20)
    min_count = max(0, int(expected_count - search_range))
    max_count = int(expected_count + search_range) + 20
    
    count_values = np.arange(min_count, max_count + 1)
    
    # Calculate cumulative distribution with multiple comparisons correction
    log_cdf = total_combinations * poisson.logcdf(count_values, expected_count)
    corrected_cdf = np.exp(log_cdf)
    
    # Find threshold at specified percentile
    target_percentile = percentile_poiss / 100.0
    threshold_idx = np.searchsorted(corrected_cdf, target_percentile, side="right")
    
    if threshold_idx >= len(count_values):
        return int(count_values[-1])
    
    return int(count_values[threshold_idx])


def _store_analysis_parameters(param_dict, session, context, job_id, 
                              analysis_params, process='original'):
    """
    Store analysis parameters for a specific job.
    
    Parameters
    ----------
    param_dict : dict
        Main parameter dictionary
    session : str
        Session identifier
    context : str
        Analysis context (epoch_trialtype)
    job_id : int
        Job counter/identifier
    analysis_params : dict
        Dictionary containing all analysis parameters
    process : str
        Process type
    """
    # Navigate to correct dictionary location
    if process == 'original':
        target_dict = param_dict[session][context]
    else:
        target_dict = param_dict[session][process][context]
    
    # Store core parameters
    target_dict[job_id] = {
        'session': session,
        'trialtype': analysis_params['trialtype'],
        'binsize': (analysis_params['binsize'] * pq.s).rescale(analysis_params['unit']),
        'epoch': analysis_params['epoch'],
        'min_spikes': analysis_params['pattern_size'],
        'max_spikes': analysis_params['pattern_size'],
        'min_occ': max(analysis_params['min_occ'], analysis_params['abs_min_occ'])
    }
    
    # Add process type for non-original data
    if process != 'original':
        target_dict[job_id]['process'] = process
    
    # Store additional analysis parameters
    additional_params = [
        'percentile_poiss', 'winlen', 'correction', 'psr_param', 'alpha',
        'n_surr', 'abs_min_occ', 'dither', 'spectrum', 'abs_min_spikes', 'surr_method'
    ]
    
    for param in additional_params:
        if param in analysis_params:
            target_dict[job_id][param] = analysis_params[param]


def _process_single_condition(process, session, epoch, trialtype, config_params, 
                             excluded_neurons, param_dict, job_counter):
    """
    Process a single experimental condition (session, epoch, trialtype combination).
    
    Parameters
    ----------
    process : str
        Process type ('original', 'ppd', 'gamma')
    session : str
        Session identifier
    epoch : str
        Epoch name
    trialtype : str
        Trial type
    config_params : dict
        Configuration parameters
    excluded_neurons : dict
        Dictionary tracking excluded neurons per session
    param_dict : dict
        Main parameter dictionary
    job_counter : int
        Current job counter
        
    Returns
    -------
    int
        Updated job counter
    """
    # Determine rates file path
    if process == 'original':
        rates_path = f'../../results/experimental_data/{session}/rates/{epoch}_{trialtype}'
    else:
        rates_path = f'../../results/artificial_data/{process}/rates/{session}/{epoch}_{trialtype}'
    
    # Load or create rate dictionary
    rates_file = os.path.join(rates_path, 'rates.npy')
    if os.path.exists(rates_file):
        rates_dict = np.load(rates_file, allow_pickle=True).item()
        print(f"  {trialtype}: max rate = {np.max(rates_dict['rates']):.3f} Hz")
    else:
        rates_dict = create_rate_dict(
            session=session,
            epoch=epoch,
            trialtype=trialtype,
            rates_path=rates_path,
            binsize=config_params['binsize'],
            process=process
        )
    
    # Extract rate information
    all_rates = rates_dict['rates']
    n_bins = rates_dict['n_bins']
    rates_by_neuron = np.array(rates_dict['rates_ordered_by_neuron'])
    
    # Apply firing rate threshold if specified
    firing_rate_threshold = config_params.get('firing_rate_threshold')
    if firing_rate_threshold is not None:
        high_rate_neurons = np.where(rates_by_neuron > firing_rate_threshold)[0]
        excluded_neurons[session] = np.append(excluded_neurons[session], high_rate_neurons)
        
        if len(excluded_neurons[session]) > 0:
            n_excluded = len(excluded_neurons[session])
            all_rates = all_rates[:-n_excluded]
    
    # Prepare analysis context
    context = f"{epoch}_{trialtype}"
    
    # Initialize parameter storage for this context
    if process == 'original':
        param_dict[session][context] = {}
    else:
        param_dict[session][process][context] = {}
    
    # Filter out neurons with zero firing rates
    active_rates = np.array([rate for rate in all_rates if rate > 0])
    if len(active_rates) == 0:
        print(f"  Warning: No active neurons for {session} {epoch} {trialtype}")
        return job_counter
    
    sorted_active_rates = np.sort(active_rates)
    non_stationarity = _get_non_stationarity_factor(epoch, process)
    
    print(f"  Processing pattern sizes 2-6 (non-stationarity: {non_stationarity})")
    
    # Process each pattern size from 2 to 6 (inclusive)
    for pattern_size in range(2, 7):
        # Calculate reference rate using geometric mean of top neurons
        if len(sorted_active_rates) >= pattern_size:
            rate_ref = gmean(sorted_active_rates[-pattern_size:])
        else:
            rate_ref = np.mean(sorted_active_rates)
        
        # Determine number of neurons with similar firing rates
        similar_rate_threshold = 0.8 * rate_ref
        n_similar_neurons = max(
            np.sum(sorted_active_rates > similar_rate_threshold),
            pattern_size
        )
        
        # Calculate minimum occurrence threshold
        min_occ = _calculate_min_occurrence_threshold(
            rate_ref=rate_ref,
            binsize=config_params['binsize'],
            pattern_size=pattern_size,
            winlen=config_params['winlen'],
            n_neurons=n_similar_neurons,
            n_bins=n_bins,
            percentile_poiss=config_params['percentile_poiss'],
            non_stationarity=non_stationarity
        )
        
        # Prepare parameters for storage
        analysis_params = {
            'trialtype': trialtype,
            'epoch': epoch,
            'pattern_size': pattern_size,
            'min_occ': min_occ,
            'binsize': config_params['binsize'],
            'unit': config_params['unit'],
            'abs_min_occ': config_params['abs_min_occ'],
            'percentile_poiss': config_params['percentile_poiss'],
            'winlen': config_params['winlen'],
            'correction': config_params['correction'],
            'psr_param': config_params['psr_param'],
            'alpha': config_params['alpha'],
            'n_surr': config_params['n_surr'],
            'dither': config_params['dither'],
            'spectrum': config_params['spectrum'],
            'abs_min_spikes': config_params['abs_min_spikes'],
            'surr_method': config_params['surr_method']
        }
        
        # Store parameters
        _store_analysis_parameters(
            param_dict=param_dict,
            session=session,
            context=context,
            job_id=job_counter,
            analysis_params=analysis_params,
            process=process
        )
        
        job_counter += 1
        print(f"    Pattern size {pattern_size}: min_occ = {max(min_occ, config_params['abs_min_occ'])}")
    
    return job_counter


def estimate_pattern_occurrences(sessions, epochs, trialtypes, config_params, processes=('original',)):
    """
    Estimate minimum occurrence thresholds for patterns across all conditions.
    
    Uses improved non-stationary Poisson model with geometric mean rate estimation
    and proper multiple comparisons correction.

    Parameters
    ----------
    sessions : list of str
        Recording sessions to analyze
    epochs : list of str
        Trial epochs to analyze
    trialtypes : list of str
        Trial types to analyze
    config_params : dict
        Configuration parameters containing analysis settings
    processes : tuple of str, optional
        Process types to analyze ('original', 'ppd', 'gamma')
        Default: ('original',)

    Returns
    -------
    tuple
        (param_dict, excluded_neurons) where:
        - param_dict: Dictionary containing analysis parameters and thresholds
        - excluded_neurons: Array of neuron IDs excluded from analysis
    """
    print("Starting pattern occurrence estimation...")
    print(f"Sessions: {sessions}")
    print(f"Epochs: {epochs}")
    print(f"Trial types: {trialtypes}")
    print(f"Processes: {processes}")
    
    # Initialize storage
    param_dict = {}
    firing_rate_threshold = config_params.get('firing_rate_threshold')
    
    if firing_rate_threshold is not None:
        excluded_neurons = {}
        print(f"Applying firing rate threshold: {firing_rate_threshold} Hz")
    else:
        excluded_neurons = None
        print("No firing rate threshold applied")
    
    # Process each session and process type
    for session in sessions:
        print(f"\nProcessing session: {session}")
        param_dict[session] = {}
        
        for process in processes:
            print(f"  Process: {process}")
            
            # Initialize process-specific storage
            if process != 'original':
                param_dict[session][process] = {}
            
            if firing_rate_threshold is not None:
                excluded_neurons[session] = np.array([])
            
            job_counter = 0
            
            # Process each epoch and trial type
            for epoch in epochs:
                print(f"  Epoch: {epoch}")
                
                for trialtype in trialtypes:
                    job_counter = _process_single_condition(
                        process=process,
                        session=session,
                        epoch=epoch,
                        trialtype=trialtype,
                        config_params=config_params,
                        excluded_neurons=excluded_neurons,
                        param_dict=param_dict,
                        job_counter=job_counter
                    )
            
            # Clean up excluded neurons list
            if firing_rate_threshold is not None:
                excluded_neurons[session] = np.unique(excluded_neurons[session])
                excluded_neurons[session] = np.sort(excluded_neurons[session])[::-1]  # Descending order
                print(f"  Excluded {len(excluded_neurons[session])} neurons with high firing rates")
    
    # Save excluded neurons
    if excluded_neurons is not None:
        output_dir = '../analysis_experimental_data/' if processes[0] == 'original' else '../analysis_artificial_data/'
        excluded_neurons_file = os.path.join(output_dir, 'excluded_neurons.npy')
        
        if not os.path.exists(excluded_neurons_file):
            os.makedirs(output_dir, exist_ok=True)
            np.save(excluded_neurons_file, excluded_neurons)
            print(f"Saved excluded neurons to: {excluded_neurons_file}")
    
    print("\nPattern occurrence estimation completed!")
    return param_dict, excluded_neurons


def load_configuration(config_file="../configfile.yaml"):
    """
    Load configuration parameters from YAML file.
    
    Parameters
    ----------
    config_file : str
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration parameters
    """
    try:
        with open(config_file, 'r') as stream:
            config = yaml.load(stream, Loader=Loader)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")


def execute_occurrence_estimation(analyze_original=True):
    """
    Execute the occurrence estimation process using configuration file.
    
    Parameters
    ----------
    analyze_original : bool
        If True, analyze original data. If False, analyze artificial data.
    """
    print("Loading configuration...")
    config = load_configuration()
    
    # Extract configuration parameters
    required_params = [
        'sessions', 'epochs', 'trialtypes', 'abs_min_occ', 'binsize',
        'percentile_poiss', 'abs_min_spikes', 'winlen', 'spectrum', 'dither',
        'n_surr', 'alpha', 'correction', 'psr_param', 'unit', 'surr_method'
    ]
    
    config_params = {}
    for param in required_params:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' not found in configuration")
        config_params[param] = config[param]
    
    # Add optional parameters
    config_params['firing_rate_threshold'] = config.get('firing_rate_threshold')
    
    # Determine processes to analyze
    if analyze_original:
        processes = ('original',)
        print("Analyzing original experimental data")
    else:
        processes = tuple(config.get('processes', ['ppd', 'gamma']))
        print(f"Analyzing artificial data: {processes}")
    
    # Run estimation
    param_dict, excluded_neurons = estimate_pattern_occurrences(
        sessions=config_params['sessions'],
        epochs=config_params['epochs'],
        trialtypes=config_params['trialtypes'],
        config_params=config_params,
        processes=processes
    )
    
    print("Occurrence estimation completed successfully!")
    return param_dict, excluded_neurons


if __name__ == "__main__":
    # Example usage
    execute_occurrence_estimation(analyze_original=True)