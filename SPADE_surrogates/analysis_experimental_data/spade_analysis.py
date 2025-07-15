#!/usr/bin/env python3
"""
Simplified SPADE analysis script for experimental data.

This script uses the shared utility functions to perform SPADE analysis
on experimental spike train data with reproducible seeding and MPI parallelization.
"""

import numpy as np
import os
from SPADE_surrogates.analyse_data_utils.spade_analysis_utils import (
    initialize_mpi,
    generate_reproducible_seed,
    parse_experimental_arguments,
    load_configuration,
    load_job_parameters,
    validate_job_context,
    load_experimental_data,
    apply_firing_rate_threshold,
    extract_spade_parameters,
    create_loading_parameters,
    run_spade_analysis,
    save_experimental_results
)


def load_calibrated_job_parameters(session, context, job_id):
    """
    Load job parameters, preferring calibrated parameters if available.
    """
    # Check if calibrated parameters exist
    calibrated_file = 'param_dict_calibrated_integrated.npy'
    
    if os.path.exists(calibrated_file):
        print(f"Loading calibrated parameters from {calibrated_file}")
        param_dict = np.load(calibrated_file, allow_pickle=True).item()
    else:
        print(f"Calibrated parameters not found, using original parameters")
        param_dict = np.load('param_dict.npy', allow_pickle=True).item()
    
    # Extract job parameters
    if session not in param_dict:
        raise ValueError(f"Session {session} not found in parameter dictionary")
    
    if context not in param_dict[session]:
        raise ValueError(f"Context {context} not found for session {session}")
    
    if job_id not in param_dict[session][context]:
        raise ValueError(f"Job {job_id} not found for session {session}, context {context}")
    
    job_params = param_dict[session][context][job_id]
    
    # Log parameter source and values
    if job_params.get('calibration_applied', False):
        original_min_occ = job_params.get('original_min_occ', 'unknown')
        current_min_occ = job_params.get('min_occ', 'unknown')
        print(f"Using calibrated parameters for job {job_id}:")
        print(f"  min_occ: {original_min_occ} → {current_min_occ}")
        print(f"  calibration_applied: {job_params.get('calibration_applied', False)}")
    else:
        print(f"Using original parameters for job {job_id}")
        print(f"  min_occ: {job_params.get('min_occ', 'unknown')}")
    
    return job_params


def main():
    """Main execution function for experimental data analysis."""
    # Initialize MPI
    comm, rank, size = initialize_mpi()
    
    if rank == 0:
        print("=== SPADE Analysis with Calibrated Parameters ===")
    
    # Parse command line arguments
    args = parse_experimental_arguments()
    session = args.session
    context = args.context
    job_id = args.job_id
    surr_method = args.surrogate_method
    
    if rank == 0:
        print(f"Job parameters: session={session}, context={context}, "
              f"job_id={job_id}, surr_method={surr_method}")
    
    # Generate reproducible seed for this job and MPI rank
    random_seed = generate_reproducible_seed(session, context, job_id, surr_method, rank)
    np.random.seed(random_seed)
    if rank == 0:
        print(f"Using reproducible seed: {random_seed}")
    
    # Load configuration and job parameters
    config = load_configuration()
    
    # Load job parameters (calibrated if available)
    job_params = load_calibrated_job_parameters(session, context, job_id)
    
    # Extract job-specific parameters
    epoch = job_params['epoch']
    trialtype = job_params['trialtype']
    
    if rank == 0:
        print(f"Extracted context: epoch={epoch}, trialtype={trialtype}")
    
    # Validate job context
    validate_job_context(context, epoch, trialtype)
    
    # Load spike train data
    spike_trains, annotations_dict = load_experimental_data(session, epoch, trialtype)
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
    loading_params = create_loading_parameters(session, epoch, trialtype, config)
    
    # Run SPADE analysis
    if rank == 0:
        print(f"Starting SPADE analysis with {size} MPI ranks...")
    
    spade_results = run_spade_analysis(spike_trains, spade_params, surr_method)
    
    # Save results (only rank 0 writes files)
    if rank == 0:
        print(f"Saving results...")
        save_experimental_results(
            spade_results=spade_results,
            loading_params=loading_params,
            spade_params=spade_params,
            config=config,
            session=session,
            epoch=epoch,
            trialtype=trialtype,
            job_id=job_id,
            surr_method=surr_method,
            annotations_dict=annotations_dict,
            mpi_rank=rank,
            mpi_size=size,
            random_seed=random_seed
        )
        print(f"Job {job_id} completed successfully!")
        print(f"Results saved to: ../../results/experimental_data/{surr_method}/{session}/{context}/{job_id}/")


if __name__ == "__main__":
    main()