#!/usr/bin/env python3
"""
SPADE analysis filtering script for experimental data.
Fixes the bug where test_signature_significance was called multiple times.
"""

import argparse
from SPADE_surrogates.analyse_data_utils.filter_results import filter_spade_results


def parse_arguments():
    """Parse command line arguments for experimental data SPADE analysis."""
    parser = argparse.ArgumentParser(
        description='Define parameter for filtering the results'
                    ' of the SPADE analysis on R2G experimental data')
    
    parser.add_argument('context', metavar='context', type=str,
                        help='behavioral context (epoch_trialtype) to analyze')
    parser.add_argument('session', metavar='session', type=str,
                        help='Recording session to analyze')
    parser.add_argument('surr_method', metavar='surr_method', type=str,
                        help='Surrogate method to use')
    
    return parser.parse_args()


def main():
    """Main function for experimental data analysis."""
    # Parse command line arguments
    args = parse_arguments()
    session_name = args.session
    context = args.context
    surr_method = args.surr_method
    
    print(f"Analyzing experimental data:")
    print(f"  Session: {session_name}")
    print(f"  Context: {context}")
    print(f"  Surrogate method: {surr_method}")
    
    # Define paths
    directory = f'../../results/experimental_data/{surr_method}/{session_name}/{context}'
    annotations_path = f'../results/{surr_method}/{session_name}/{context}/annotations.npy'
    output_path = f'../../results/experimental_data/{surr_method}/{session_name}/{context}/filtered_res.npy'
    
    # Process the results using shared utility functions
    filter_spade_results(directory, annotations_path, output_path)
    
    print('Experimental data analysis completed successfully!')


if __name__ == '__main__':
    main()