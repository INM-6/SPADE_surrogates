"""
Spike Train Data Concatenation Script

This script loads experimental spike train data from the 'multielectrode grasp' 
dataset and saves them as concatenated spike trains for further analysis.

The script processes multiple sessions, epochs, and trial types, concatenating
trials within each combination and saving the results as numpy files.
"""

import os
import numpy as np
import quantities as pq
import yaml
from yaml import Loader
import rgutils


def load_configuration(config_file="configfile.yaml"):
    """
    Load configuration parameters from YAML file.
    
    Parameters
    ----------
    config_file : str
        Path to the configuration YAML file
        
    Returns
    -------
    dict
        Configuration dictionary containing all parameters
    """
    print(f"Loading configuration from {config_file}...")
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    
    print(f"Configuration loaded successfully.")
    return config


def create_output_directory(session):
    """
    Create output directory for concatenated spike trains if it doesn't exist.
    
    Parameters
    ----------
    session : str
        Session identifier for creating session-specific directory
    """
    output_dir = f'../data/concatenated_spiketrains/{session}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir


def process_session_data(session, epochs, trialtypes, config_params):
    """
    Process all epoch and trial type combinations for a given session.
    
    Parameters
    ----------
    session : str
        Session identifier
    epochs : list
        List of epochs to process
    trialtypes : list
        List of trial types to process
    config_params : dict
        Dictionary containing processing parameters
    """
    print(f"\n--- Processing session: {session} ---")
    output_dir = create_output_directory(session)
    
    # Extract parameters for readability
    snr_threshold = config_params['SNR_thresh']
    synchrony_size = config_params['synchsize']
    trial_separation = config_params['sep']
    
    total_combinations = len(epochs) * len(trialtypes)
    current_combination = 0
    
    for epoch in epochs:
        for trialtype in trialtypes:
            current_combination += 1
            print(f"Processing [{current_combination}/{total_combinations}]: "
                  f"{session} | {epoch} | {trialtype}")
            
            # Load concatenated spike trains for this combination
            spike_trains = rgutils.load_epoch_concatenated_trials(
                session=session,
                epoch=epoch,
                trialtypes=trialtype,
                SNRthresh=snr_threshold,
                synchsize=synchrony_size,
                sep=trial_separation
            )
            
            # Save concatenated spike trains
            output_filename = f'{output_dir}/{epoch}_{trialtype}.npy'
            np.save(output_filename, spike_trains)
            print(f"Saved: {output_filename}")
    
    print(f"Session {session} processing complete!")


def main():
    """
    Main execution function that orchestrates the entire concatenation process.
    """
    print("=== Spike Train Concatenation Script ===")
    print("Loading experimental data and creating concatenated spike trains...\n")
    
    # Load configuration
    config = load_configuration()
    
    # Extract configuration parameters
    sessions = config['sessions']
    epochs = config['epochs']
    trialtypes = config['trialtypes']
    window_length = config['winlen']
    time_unit = config['unit']
    bin_size = (config['binsize'] * pq.s).rescale(time_unit)
    snr_threshold = config['SNR_thresh']
    synchrony_size = config['synchsize']
    
    # Calculate trial separation time
    trial_separation = 2 * window_length * bin_size
    
    # Package parameters for passing to processing function
    processing_params = {
        'SNR_thresh': snr_threshold,
        'synchsize': synchrony_size,
        'sep': trial_separation
    }
    
    # Display processing summary
    print(f"Processing Summary:")
    print(f"  Sessions: {len(sessions)} ({', '.join(sessions)})")
    print(f"  Epochs: {len(epochs)} ({', '.join(epochs)})")
    print(f"  Trial types: {len(trialtypes)} ({', '.join(trialtypes)})")
    print(f"  SNR threshold: {snr_threshold}")
    print(f"  Synchrony size: {synchrony_size}")
    print(f"  Trial separation: {trial_separation}")
    print(f"  Total combinations: {len(sessions) * len(epochs) * len(trialtypes)}")
    
    # Process each session
    for session_idx, session in enumerate(sessions, 1):
        print(f"\n>>> Processing session {session_idx}/{len(sessions)} <<<")
        process_session_data(
            session=session,
            epochs=epochs,
            trialtypes=trialtypes,
            config_params=processing_params
        )
    
    print("\n=== All processing complete! ===")
    print("Concatenated spike trains have been saved to ../data/concatenated_spiketrains/")


if __name__ == '__main__':
    main()