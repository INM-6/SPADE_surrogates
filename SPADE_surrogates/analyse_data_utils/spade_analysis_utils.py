"""
Utility functions for SPADE analysis scripts.

This module contains shared functionality for both experimental and artificial data
SPADE analysis, including reproducible seeding, parameter loading, data handling,
and result saving.
"""

import numpy as np
import quantities as pq
import argparse
import hashlib
from mpi4py import MPI
import yaml
from yaml import Loader
from elephant.spade import spade
from SPADE_surrogates.analyse_data_utils.spade_utils import mkdirp, split_path
from SPADE_surrogates.rgutils import save_processed_data, load_processed_spike_trains


def generate_reproducible_seed(session, context, job_id, surr_method, mpi_rank=0):
    """
    Generate a reproducible but unique seed for each SPADE job and MPI rank.
    
    The seed is generated using a hash of the job parameters plus MPI rank, ensuring:
    1. Same parameters and rank always produce the same seed (reproducible)
    2. Different parameter combinations produce different seeds (unique across jobs)
    3. Different MPI ranks produce different seeds (independent surrogates)
    4. Seeds are well-distributed across the random number space
    
    Parameters
    ----------
    session : str
        Recording session identifier
    context : str
        Analysis context (epoch_trialtype)
    job_id : int
        Job identifier
    surr_method : str
        Surrogate method name
    mpi_rank : int, optional
        MPI rank (process ID). Default: 0
        
    Returns
    -------
    int
        Reproducible seed value (0 to 2^32-1)
    """
    # Create a unique string from job parameters AND MPI rank
    seed_string = f"{session}_{context}_{job_id}_{surr_method}_rank{mpi_rank}"
    
    # Generate hash and convert to integer seed
    hash_object = hashlib.md5(seed_string.encode())
    hash_hex = hash_object.hexdigest()
    
    # Convert first 8 hex characters to integer (32-bit seed)
    seed = int(hash_hex[:8], 16)
    
    return seed


def parse_experimental_arguments():
    """Parse command line arguments for experimental data analysis."""
    parser = argparse.ArgumentParser(
        description='Perform SPADE analysis on experimental data')
    
    parser.add_argument('job_id', metavar='job_id', type=int,
                        help='ID of the job to perform in this call')
    parser.add_argument('context', metavar='context', type=str,
                        help='Behavioral context (epoch_trialtype) to analyze')
    parser.add_argument('session', metavar='session', type=str,
                        help='Recording session to analyze')
    parser.add_argument('surrogate_method', metavar='surr_method', type=str,
                        help='Surrogate method to use')
    
    return parser.parse_args()


def parse_artificial_arguments():
    """Parse command line arguments for artificial data analysis."""
    parser = argparse.ArgumentParser(
        description='Perform SPADE analysis on artificial data')
    
    parser.add_argument('job_id', metavar='job_id', type=int,
                        help='ID of the job to perform in this call')
    parser.add_argument('context', metavar='context', type=str,
                        help='Behavioral context (epoch_trialtype) to analyze')
    parser.add_argument('session', metavar='session', type=str,
                        help='Recording session to analyze')
    parser.add_argument('process', metavar='process', type=str,
                        help='Point process to analyze')
    parser.add_argument('surrogate_method', metavar='surr_method', type=str,
                        help='Surrogate method to use')
    
    return parser.parse_args()


def load_job_parameters(session, context, job_id, process=None):
    """
    Load job-specific parameters from parameter dictionary.
    
    Parameters
    ----------
    session : str
        Session identifier
    context : str
        Analysis context
    job_id : int
        Job identifier
    process : str, optional
        Process type (for artificial data). If None, assumes experimental data.
        
    Returns
    -------
    dict
        Job-specific parameters
    """
    try:
        param_dict = np.load('./param_dict.npy', encoding='latin1',
                           allow_pickle=True).item()
        
        if process is None:
            # Experimental data: session -> context -> job_id
            return param_dict[session][context][job_id]
        else:
            # Artificial data: session -> process -> context -> job_id
            return param_dict[session][process][context][job_id]
            
    except (FileNotFoundError, KeyError) as e:
        raise ValueError(f"Could not load parameters for job {job_id}, "
                        f"session {session}, context {context}, process {process}: {e}")


def load_configuration():
    """Load configuration from YAML file."""
    try:
        with open("configfile.yaml", 'r') as stream:
            config = yaml.load(stream, Loader=Loader)
        return config
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file 'configfile.yaml' not found")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")


def validate_job_context(context, epoch, trialtype):
    """Validate that job context matches epoch and trialtype."""
    expected_context = f"{epoch}_{trialtype}"
    if context != expected_context:
        raise ValueError(
            f"Job context '{context}' does not match expected "
            f"'{expected_context}' from epoch '{epoch}' and trialtype '{trialtype}'")


def load_experimental_data(session, epoch, trialtype):
    """
    Load experimental spike train data.
    
    Parameters
    ----------
    session : str
        Session identifier
    epoch : str
        Epoch name
    trialtype : str
        Trial type
        
    Returns
    -------
    tuple
        (spike_trains_list, annotations_dict)
    """
    data_file = f'../../data/concatenated_spiketrains/{session}/{epoch}_{trialtype}.npy'
    
    try:
        spike_trains_list = load_processed_spike_trains(data_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Experimental data not found: {data_file}")
    
    # Generate annotations dictionary
    annotations_dict = {}
    for st_idx, spike_train in enumerate(spike_trains_list):
        annotations_dict[st_idx] = spike_train.annotations
    
    return spike_trains_list, annotations_dict


def load_artificial_data(session, epoch, trialtype, process):
    """
    Load artificial spike train data.
    
    Parameters
    ----------
    session : str
        Session identifier
    epoch : str
        Epoch name
    trialtype : str
        Trial type
    process : str
        Process type (e.g., 'ppd', 'gamma')
        
    Returns
    -------
    tuple
        (spike_trains_list, annotations_dict)
    """
    data_file = f'../../data/artificial_data/{process}/{session}/{process}_{epoch}_{trialtype}.npy'
    
    try:
        spike_trains_list = load_processed_spike_trains(data_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Artificial data not found: {data_file}")
    
    # Generate annotations dictionary
    annotations_dict = {}
    for st_idx, spike_train in enumerate(spike_trains_list):
        annotations_dict[st_idx] = spike_train.annotations
    
    return spike_trains_list, annotations_dict


def apply_firing_rate_threshold(spike_trains, session, firing_rate_threshold):
    """
    Remove neurons that exceed the firing rate threshold.
    
    Parameters
    ----------
    spike_trains : list
        List of spike trains
    session : str
        Session identifier
    firing_rate_threshold : float or None
        Maximum allowed firing rate
        
    Returns
    -------
    list
        Filtered spike trains list
    """
    if firing_rate_threshold is None:
        return spike_trains
    
    try:
        excluded_neurons_dict = np.load('excluded_neurons.npy', allow_pickle=True).item()
        excluded_neurons = excluded_neurons_dict.get(session, [])
        
        # Remove neurons in reverse order to maintain correct indices
        for neuron_idx in sorted(excluded_neurons, reverse=True):
            if int(neuron_idx) < len(spike_trains):
                spike_trains.pop(int(neuron_idx))
                
        print(f"Excluded {len(excluded_neurons)} neurons with high firing rates")
        
    except FileNotFoundError:
        print("Warning: excluded_neurons.npy not found. "
              "Run parameter estimation first.")
    
    return spike_trains


def extract_spade_parameters(config, job_params):
    """
    Extract and format SPADE analysis parameters.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    job_params : dict
        Job-specific parameters
        
    Returns
    -------
    dict
        Formatted SPADE parameters
    """
    # Extract parameters
    spectrum = config['spectrum']
    winlen = config['winlen']
    unit = config['unit']
    dither = (config['dither'] * pq.s).rescale(unit)
    n_surr = config['n_surr']
    binsize = (config['binsize'] * pq.s).rescale(unit)
    
    min_occ = job_params['min_occ']
    min_spikes = job_params['min_spikes']
    max_spikes = job_params['max_spikes']
    
    # Fixed parameters for individual job analysis
    min_neu = min_spikes
    alpha = 1  # No statistical correction at job level
    psr_param = None  # No PSR at job level
    
    return {
        'binsize': binsize,
        'winlen': winlen,
        'min_spikes': min_spikes,
        'max_spikes': max_spikes,
        'min_neu': min_neu,
        'min_occ': min_occ,
        'n_surr': n_surr,
        'alpha': alpha,
        'psr_param': psr_param,
        'dither': dither,
        'spectrum': spectrum
    }


def create_loading_parameters(session, epoch, trialtype, config, process=None):
    """
    Create loading parameters dictionary.
    
    Parameters
    ----------
    session : str
        Session identifier
    epoch : str
        Epoch name
    trialtype : str
        Trial type
    config : dict
        Configuration dictionary
    process : str, optional
        Process type (for artificial data)
        
    Returns
    -------
    dict
        Loading parameters
    """
    winlen = config['winlen']
    unit = config['unit']
    binsize = (config['binsize'] * pq.s).rescale(unit)
    
    loading_params = {
        'session': session,
        'epoch': epoch,
        'trialtype': trialtype,
        'SNR_thresh': config['SNR_thresh'],
        'synchsize': config['synchsize'],
        'sep': 2 * winlen * binsize,
        'firing_rate_threshold': config['firing_rate_threshold']
    }
    
    if process is not None:
        loading_params['process'] = process
    
    return loading_params


def run_spade_analysis(spike_trains, spade_params, surr_method):
    """
    Run SPADE analysis with appropriate parameters.
    
    Parameters
    ----------
    spike_trains : list
        List of spike trains
    spade_params : dict
        SPADE parameters
    surr_method : str
        Surrogate method
        
    Returns
    -------
    dict
        SPADE analysis results
    """
    print(f"Running SPADE analysis with {len(spike_trains)} spike trains")
    print(f"Parameters: min_spikes={spade_params['min_spikes']}, "
          f"max_spikes={spade_params['max_spikes']}, "
          f"min_occ={spade_params['min_occ']}, "
          f"n_surr={spade_params['n_surr']}")
    
    # Special handling for trial_shifting surrogate method
    if surr_method == 'trial_shifting':
        # Get separation parameter from spade_params (calculated in create_loading_parameters)
        sep = spade_params.get('sep', 2 * spade_params['winlen'] * spade_params['binsize'])
        
        results = spade(
            spiketrains=spike_trains,
            binsize=spade_params['binsize'],
            winlen=spade_params['winlen'],
            min_occ=spade_params['min_occ'],
            min_spikes=spade_params['min_spikes'],
            max_spikes=spade_params['max_spikes'],
            min_neu=spade_params['min_neu'],
            dither=spade_params['dither'],
            n_surr=spade_params['n_surr'],
            alpha=spade_params['alpha'],
            psr_param=spade_params['psr_param'],
            surr_method=surr_method,
            spectrum=spade_params['spectrum'],
            output_format='concepts',
            trial_length=500 * pq.ms,
            trial_separation=sep
        )
    else:
        results = spade(
            spiketrains=spike_trains,
            binsize=spade_params['binsize'],
            winlen=spade_params['winlen'],
            min_occ=spade_params['min_occ'],
            min_spikes=spade_params['min_spikes'],
            max_spikes=spade_params['max_spikes'],
            min_neu=spade_params['min_neu'],
            dither=spade_params['dither'],
            n_surr=spade_params['n_surr'],
            alpha=spade_params['alpha'],
            psr_param=spade_params['psr_param'],
            surr_method=surr_method,
            spectrum=spade_params['spectrum'],
            output_format='concepts'
        )
    
    print("SPADE analysis completed")
    return results


def create_output_directory(output_path):
    """Create output directory structure if it doesn't exist."""
    current_path = './'
    for folder in split_path(output_path):
        current_path = current_path + '/' + folder
        mkdirp(current_path)


def save_experimental_results(spade_results, loading_params, spade_params, config,
                             session, epoch, trialtype, job_id, surr_method,
                             annotations_dict, mpi_rank, mpi_size, random_seed):
    """
    Save experimental data analysis results.
    
    Parameters
    ----------
    spade_results : dict
        SPADE analysis results
    loading_params : dict
        Loading parameters
    spade_params : dict
        SPADE parameters
    config : dict
        Configuration dictionary
    session : str
        Session identifier
    epoch : str
        Epoch name
    trialtype : str
        Trial type
    job_id : int
        Job identifier
    surr_method : str
        Surrogate method
    annotations_dict : dict
        Annotations dictionary
    mpi_rank : int
        MPI rank
    mpi_size : int
        MPI size
    random_seed : int
        Random seed used
    """
    # Add MPI and seed information to spade_params
    spade_params_with_info = spade_params.copy()
    spade_params_with_info.update({
        'surr_method': surr_method,
        'correction': config['correction'],
        'random_seed': random_seed,
        'mpi_rank': mpi_rank,
        'mpi_size': mpi_size
    })
    
    # Create output directory
    output_path = (f'../../results/experimental_data/{surr_method}/'
                  f'{session}/{epoch}_{trialtype}/{job_id}')
    create_output_directory(output_path)
    
    # Save SPADE results
    results_file = f'{output_path}/results.npy'
    np.save(results_file, [spade_results, loading_params, spade_params_with_info, config])
    print(f"Saved SPADE results: {results_file}")
    
    # Save annotations dictionary
    annotations_file = (f'../../results/experimental_data/{surr_method}/'
                       f'{session}/{epoch}_{trialtype}/annotations.npy')
    np.save(annotations_file, annotations_dict)
    print(f"Saved annotations: {annotations_file}")


def save_artificial_results(spade_results, loading_params, spade_params, config,
                           session, epoch, trialtype, process, job_id, surr_method,
                           annotations_dict, mpi_rank, mpi_size, random_seed):
    """
    Save artificial data analysis results.
    
    Parameters
    ----------
    spade_results : dict
        SPADE analysis results
    loading_params : dict
        Loading parameters
    spade_params : dict
        SPADE parameters
    config : dict
        Configuration dictionary
    session : str
        Session identifier
    epoch : str
        Epoch name
    trialtype : str
        Trial type
    process : str
        Process type
    job_id : int
        Job identifier
    surr_method : str
        Surrogate method
    annotations_dict : dict
        Annotations dictionary
    mpi_rank : int
        MPI rank
    mpi_size : int
        MPI size
    random_seed : int
        Random seed used
    """
    # Add MPI and seed information to spade_params
    spade_params_with_info = spade_params.copy()
    spade_params_with_info.update({
        'surr_method': surr_method,
        'random_seed': random_seed,
        'mpi_rank': mpi_rank,
        'mpi_size': mpi_size
    })
    
    # Create output directory
    output_path = (f'../../results/artificial_data/{surr_method}/'
                  f'{process}/{session}/{epoch}_{trialtype}/{job_id}')
    create_output_directory(output_path)
    
    # Save SPADE results
    results_file = f'{output_path}/results.npy'
    np.save(results_file, [spade_results, loading_params, spade_params_with_info, config])
    print(f"Saved SPADE results: {results_file}")
    
    # Save annotations dictionary
    annotations_file = (f'../../results/artificial_data/{surr_method}/'
                       f'{process}/{session}/{epoch}_{trialtype}/annotations.npy')
    np.save(annotations_file, annotations_dict)
    print(f"Saved annotations: {annotations_file}")


def initialize_mpi():
    """
    Initialize MPI and return communicator information.
    
    Returns
    -------
    tuple
        (comm, rank, size)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print(f"Starting SPADE analysis with {size} MPI processes (rank {rank})")
    return comm, rank, size