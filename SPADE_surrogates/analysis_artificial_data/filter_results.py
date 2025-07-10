#!/usr/bin/env python3
"""
SPADE analysis filtering script for artificial data.
Fixes the bug where test_signature_significance was called multiple times.
"""

import argparse
from SPADE_surrogates.analyse_data_utils.filter_results import filter_spade_results


def parse_arguments():
    """Parse command line arguments for artificial data SPADE analysis."""
    parser = argparse.ArgumentParser(
        description='Define parameter for filtering the results'
                    ' of the SPADE analysis on R2G artificial data')
    
    parser.add_argument('context', metavar='context', type=str,
                        help='behavioral context (epoch_trialtype) to analyze')
    parser.add_argument('session', metavar='session', type=str,
                        help='Recording session to analyze')
    parser.add_argument('process', metavar='process', type=str,
                        help='Point process to analyze')
    parser.add_argument('surr_method', metavar='surr_method', type=str,
                        help='Surrogate method to use')
    
    return parser.parse_args()


def main():
    """Main function for artificial data analysis."""
    # Parse command line arguments
    args = parse_arguments()
    session_name = args.session
    context = args.context
    process = args.process
    surr_method = args.surr_method
    
    print(f"Analyzing artificial data:")
    print(f"  Session: {session_name}")
    print(f"  Context: {context}")
    print(f"  Process: {process}")
    print(f"  Surrogate method: {surr_method}")
    
    # Define paths for artificial data
    directory = f'../../results/artificial_data/{surr_method}/{process}/{session_name}/{context}'
    annotations_path = f'../../results/artificial_data/{surr_method}/{process}/{session_name}/{context}/annotations.npy'
    output_path = f'../../results/artificial_data/{surr_method}/{process}/{session_name}/{context}/filtered_res.npy'
    
    # Process the results using shared utility functions
    filter_spade_results(directory, annotations_path, output_path)
    
    print('Artificial data analysis completed successfully!')


if __name__ == '__main__':
    main()