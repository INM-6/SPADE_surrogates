#!/usr/bin/env python3
"""
Simplified SPADE analysis script for artificial data.

This script uses the shared utility functions to perform SPADE analysis
on artificial spike train data with reproducible seeding and MPI parallelization.

Usage:
    mpirun -n <n_ranks> python spade_analysis.py <pattern_size> <context> <session> <process> <surrogate_method>
"""

import numpy as np
import os
import sys
import argparse
from SPADE_surrogates.analyse_data_utils.spade_analysis_utils import (
    initialize_mpi,
    generate_reproducible_seed,
    load_configuration,
    validate_job_context,
    load_artificial_data,
    apply_firing_rate_threshold,
    extract_spade_parameters,
    create_loading_parameters,
    run_spade_analysis,
    save_artificial_results
)


def parse_artificial_arguments():
    """Parse command line arguments for artificial data SPADE analysis."""
    parser = argparse.ArgumentParser(
        description='SPADE analysis for artificial data with calibrated parameters')
    
    parser.add_argument('pattern_size', type=int,
                        help='Pattern size (number of spikes)')
    parser.add_argument('context', type=str,
                        help='Behavioral context (epoch_trialtype)')
    parser.add_argument('session', type=str,
                        help='Recording session')
    parser.add_argument('process', type=str,
                        help='Point process type')
    parser.add_argument('surrogate_method', type=str,
                        help='Surrogate method to use')
    
    return parser.parse_args()


def load_calibrated_job_parameters(session, context, process, pattern_size):
    """
    Load job parameters for artificial data, preferring calibrated parameters if available.
    """
    # Check if calibrated parameters exist
    calibrated_file = 'param_dict_calibrated_integrated.npy'
    
    if os.path.exists(calibrated_file):
        print(f"Loading calibrated parameters from {calibrated_file}")
        param_dict = np.load(calibrated_file, allow_pickle=True).item()
    else:
        print(f"Calibrated parameters not found, using original parameters")
        param_dict = np.load('param_dict.npy', allow_pickle=True).item()
    
    # Extract job parameters for artificial data structure
    if session not in param_dict:
        raise ValueError(f"Session {session} not found in parameter dictionary")
    
    if process not in param_dict[session]:
        raise ValueError(f"Process {process} not found for session {session}")
    
    if context not in param_dict[session][process]:
        raise ValueError(f"Context {context} not found for session {session}, process {process}")
    
    # Get job parameters directly by pattern_size key
    jobs = param_dict[session][process][context]
    
    if pattern_size in jobs:
        job_params = jobs[pattern_size]
    else:
        # Fallback: search by min_spikes if pattern_size key doesn't exist
        job_params = None
        for job_id, job_data in jobs.items():
            if job_data.get('min_spikes', 2) == pattern_size:
                job_params = job_data
                break
        
        if job_params is None:
            raise ValueError(f"No job found with pattern size {pattern_size} for session {session}, process {process}, context {context}")
    
    # Log parameter source and values
    if job_params.get('calibration_applied', False):
        original_min_occ = job_params.get('original_min_occ', 'unknown')
        current_min_occ = job_params.get('min_occ', 'unknown')
        print(f"Using calibrated parameters for pattern size {pattern_size}:")
        print(f"  min_occ: {original_min_occ} → {current_min_occ}")
        print(f"  calibration_applied: {job_params.get('calibration_applied', False)}")
    else:
        print(f"Using original parameters for pattern size {pattern_size}")
        print(f"  min_occ: {job_params.get('min_occ', 'unknown')}")
    
    return job_params


def main():
    """Main execution function for artificial data analysis."""
    # Initialize MPI
    comm, rank, size = initialize_mpi()
    
    if rank == 0:
        print("=== SPADE Analysis with Calibrated Parameters - Artificial Data ===")
    
    # Parse command line arguments
    args = parse_artificial_arguments()
    pattern_size = args.pattern_size
    context = args.context
    session = args.session
    process = args.process
    surr_method = args.surrogate_method
    
    if rank == 0:
        print(f"Job parameters: session={session}, context={context}, process={process}, "
              f"pattern_size={pattern_size}, surr_method={surr_method}")
    
    # Generate reproducible seed for this job and MPI rank
    random_seed = generate_reproducible_seed(session, context, f"{process}_{pattern_size}", surr_method, rank)
    np.random.seed(random_seed)
    if rank == 0:
        print(f"Using reproducible seed: {random_seed}")
    
    # Load configuration and job parameters
    config = load_configuration()
    
    # Load job parameters (calibrated if available)
    job_params = load_calibrated_job_parameters(session, context, process, pattern_size)
    
    # Extract job-specific parameters
    epoch = job_params['epoch']
    trialtype = job_params['trialtype']
    
    if rank == 0:
        print(f"Extracted context: epoch={epoch}, trialtype={trialtype}")
    
    # Validate job context
    validate_job_context(context, epoch, trialtype)
    
    # Load artificial spike train data
    spike_trains, annotations_dict = load_artificial_data(session, epoch, trialtype, process)
    if rank == 0:
        print(f"Loaded {len(spike_trains)} spike trains")
    
    # Apply firing rate threshold if specified
    firing_rate_threshold = config['firing_rate_threshold']
    spike_trains = apply_firing_rate_threshold(spike_trains, session, firing_rate_threshold)
    if rank == 0:
        print(f"Using {len(spike_trains)} spike trains after filtering")
    
    # Extract SPADE parameters (using calibrated job_params)
    spade_params = extract_spade_parameters(config, job_params)
    
    if rank == 0:
        print(f"SPADE parameters:")
        for key, value in spade_params.items():
            print(f"  {key}: {value}")
    
    # Create loading parameters
    loading_params = create_loading_parameters(session, epoch, trialtype, config, process=process)
    
    # Run SPADE analysis
    if rank == 0:
        print(f"Starting SPADE analysis with {size} MPI ranks...")
    
    spade_results = run_spade_analysis(spike_trains, spade_params, surr_method)
    
    # Save results (only rank 0 writes files)
    if rank == 0:
        print(f"Saving results...")
        save_artificial_results(
            spade_results=spade_results,
            loading_params=loading_params,
            spade_params=spade_params,
            config=config,
            session=session,
            epoch=epoch,
            trialtype=trialtype,
            process=process,
            job_id=pattern_size,
            surr_method=surr_method,
            annotations_dict=annotations_dict,
            mpi_rank=rank,
            mpi_size=size,
            random_seed=random_seed
        )
        print(f"Pattern size {pattern_size} completed successfully!")
        print(f"Results saved to: ../../results/artificial_data/{surr_method}/{process}/{session}/{context}/{pattern_size}/")


if __name__ == "__main__":
    main()