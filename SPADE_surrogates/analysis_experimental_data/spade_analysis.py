#!/usr/bin/env python3
"""
Simplified SPADE analysis script for experimental data.

This script uses the shared utility functions to perform SPADE analysis
on experimental spike train data with reproducible seeding and MPI parallelization.
"""

import numpy as np
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


def main():
    """Main execution function for experimental data analysis."""
    # Initialize MPI
    comm, rank, size = initialize_mpi()
    
    # Parse command line arguments
    args = parse_experimental_arguments()
    session = args.session
    context = args.context
    job_id = args.job_id
    surr_method = args.surrogate_method
    
    print(f"Job parameters: session={session}, context={context}, "
          f"job_id={job_id}, surr_method={surr_method}")
    
    # Generate reproducible seed for this job and MPI rank
    random_seed = generate_reproducible_seed(session, context, job_id, surr_method, rank)
    np.random.seed(random_seed)
    print(f"Rank {rank}: Using reproducible seed: {random_seed}")
    
    # Load configuration and job parameters
    config = load_configuration()
    job_params = load_job_parameters(session, context, job_id)
    
    # Extract job-specific parameters
    epoch = job_params['epoch']
    trialtype = job_params['trialtype']
    
    # Validate job context
    validate_job_context(context, epoch, trialtype)
    
    # Load spike train data
    spike_trains, annotations_dict = load_experimental_data(session, epoch, trialtype)
    print(f"Loaded {len(spike_trains)} spike trains")
    
    # Apply firing rate threshold if specified
    firing_rate_threshold = config['firing_rate_threshold']
    spike_trains = apply_firing_rate_threshold(spike_trains, session, firing_rate_threshold)
    print(f"Using {len(spike_trains)} spike trains after filtering")
    
    # Extract SPADE parameters
    spade_params = extract_spade_parameters(config, job_params)
    
    # Create loading parameters
    loading_params = create_loading_parameters(session, epoch, trialtype, config)
    
    # Run SPADE analysis
    spade_results = run_spade_analysis(spike_trains, spade_params, surr_method)
    
    # Save results (only rank 0 writes files)
    if rank == 0:
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


if __name__ == "__main__":
    main()