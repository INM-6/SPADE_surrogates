"""
Utility functions for SPADE analysis filtering and processing.
Contains shared functionality between experimental and artificial data analysis scripts.
"""

import numpy as np
import os
import elephant.spade as spade
import quantities as pq


def load_results_from_directory(directory):
    """
    Load and merge SPADE results from all files in the directory.
    
    Parameters:
    -----------
    directory : str
        Path to directory containing SPADE results
        
    Returns:
    --------
    tuple: (concepts, pv_spec_list, config_params, spade_params, loading_params)
    """
    concepts = []
    pv_spec_list = []
    config_params = {}
    spade_params = {}
    loading_params = {}
    
    # Get all subdirectories
    subdirectories = [x[0] for x in os.walk(directory)][1:]
    
    for path in subdirectories:
        print(f"Processing: {path}")
        
        # Get files in this directory
        for dirpath, dirnames, filenames in os.walk(path):
            if len(filenames) > 1:
                raise ValueError(f'More than 1 file in folder: {path}')
            if len(filenames) == 0:
                continue
                
            # Load the results file
            filename = filenames[0]
            filepath = os.path.join(path, filename)
            results, loading_param, spade_param, configfile_param = np.load(
                filepath, encoding='latin1', allow_pickle=True)
            
            print(f"  Found {len(results['patterns'])} concepts")
            
            # Merge concepts from this file
            concepts.extend(results['patterns'])
            
            # Store parameters (assuming consistent across files)
            config_params = configfile_param
            spade_params = spade_param
            loading_params = loading_param
            
            # Collect p-value spectrum data if surrogate testing was performed
            n_surr = configfile_param['n_surr']
            if n_surr > 0:
                pv_spec = results['pvalue_spectrum']
                print(f"  P-value spectrum length: {len(pv_spec)}")
                
                if len(pv_spec) > 0:
                    # Ensure pv_spec is in the correct format before extending
                    # Format: [pattern_size, pattern_occ, p_value] for spectrum '#'
                    #         [pattern_size, pattern_occ, pattern_dur, p_value] for spectrum '3d#'
                    pv_spec_list.extend(pv_spec)
    
    return concepts, pv_spec_list, config_params, spade_params, loading_params


def validate_pv_spec_format(pv_spec_list, spectrum):
    """
    Validate that pv_spec entries have the correct format.
    
    Parameters:
    -----------
    pv_spec_list : list
        List of p-value spectrum entries
    spectrum : str
        Spectrum type ('#' or '3d#')
    """
    if len(pv_spec_list) == 0:
        return
        
    first_entry = pv_spec_list[0]
    print(f'P-value spectrum entry format: {first_entry} (length: {len(first_entry)})')
    
    expected_length = 3 if spectrum == '#' else 4
    if len(first_entry) != expected_length:
        raise ValueError(
            f"Expected pv_spec entries of length {expected_length} for spectrum '{spectrum}', "
            f"got {len(first_entry)}")


def compute_non_significant_signatures(pv_spec_list, concepts, config_params):
    """
    Compute non-significant signatures using statistical testing.
    
    Parameters:
    -----------
    pv_spec_list : list
        Merged p-value spectrum data
    concepts : list
        List of concepts
    config_params : dict
        Configuration parameters
        
    Returns:
    --------
    list: Non-significant signatures
    """
    alpha = config_params['alpha']
    correction = config_params['correction']
    winlen = config_params['winlen']
    n_surr = config_params['n_surr']
    spectrum = config_params['spectrum']
    
    # Skip if no surrogate testing or alpha is 1 (all significant)
    if n_surr in {0, None} or len(pv_spec_list) == 0 or alpha in {None, 1}:
        return []
    
    print(f'Total merged p-value spectrum length: {len(pv_spec_list)}')
    
    # Validate format
    validate_pv_spec_format(pv_spec_list, spectrum)
    
    # Compute non-significant signatures with statistical correction
    ns_signatures = spade.test_signature_significance(
        pv_spec=pv_spec_list,
        concepts=concepts,
        alpha=alpha,
        winlen=winlen,
        corr=correction,
        report='non_significant',
        spectrum=spectrum)
    
    return ns_signatures


def apply_pattern_spectrum_filter(concepts, ns_signatures, config_params):
    """
    Apply pattern spectrum filter (PSF) to concepts.
    
    Parameters:
    -----------
    concepts : list
        List of concepts to filter
    ns_signatures : list
        Non-significant signatures
    config_params : dict
        Configuration parameters
        
    Returns:
    --------
    list: Filtered concepts
    """
    spectrum = config_params['spectrum']
    winlen = config_params['winlen']
    n_surr = config_params['n_surr']
    
    if n_surr not in {0, None} and len(ns_signatures) > 0:
        filtered_concepts = [
            concept for concept in concepts
            if spade._pattern_spectrum_filter(
                concept=concept,
                ns_signatures=ns_signatures,
                spectrum=spectrum,
                winlen=winlen)
        ]
        print(f'Concepts after PSF filtering: {len(filtered_concepts)}')
        return filtered_concepts
    
    return concepts


def apply_pattern_set_reduction(concepts, ns_signatures, config_params):
    """
    Apply pattern set reduction (PSR) filtering if configured.
    
    Parameters:
    -----------
    concepts : list
        List of concepts to filter
    ns_signatures : list
        Non-significant signatures
    config_params : dict
        Configuration parameters
        
    Returns:
    --------
    list: Filtered concepts
    """
    psr_param = config_params['psr_param']
    
    if psr_param is None:
        return concepts
    
    spectrum = config_params['spectrum']
    winlen = config_params['winlen']
    min_spikes = config_params['abs_min_spikes']
    min_occ = config_params['abs_min_occ']
    
    filtered_concepts = spade.pattern_set_reduction(
        concepts=concepts,
        ns_signatures=ns_signatures,
        winlen=winlen,
        spectrum=spectrum,
        h_subset_filtering=psr_param[0],
        k_superset_filtering=psr_param[1],
        l_covered_spikes=psr_param[2],
        min_spikes=min_spikes,
        min_occ=min_occ)
    
    return filtered_concepts


def add_neuron_annotations(patterns, annotations_path):
    """
    Add channel, unit, and connector-aligned IDs to patterns.
    
    Parameters:
    -----------
    patterns : list
        List of patterns to annotate
    annotations_path : str
        Path to annotations file
    """
    annotations = np.load(annotations_path, allow_pickle=True).item()
    
    for pattern in patterns:
        pattern['channel_ids'] = []
        pattern['unit_ids'] = []
        pattern['connector_aligned_ids'] = []
        
        # Process each neuron in the pattern
        for neuron_id in pattern['neurons']:
            annotation = annotations[neuron_id]
            pattern['channel_ids'].append(annotation['channel_id'])
            pattern['unit_ids'].append(annotation['unit_id'])
            pattern['connector_aligned_ids'].append(annotation['connector_aligned_id'])


def create_filter_params(config_params):
    """
    Create filter parameters dictionary for saving.
    
    Parameters:
    -----------
    config_params : dict
        Configuration parameters
        
    Returns:
    --------
    dict: Filter parameters
    """
    return {
        'alpha': config_params['alpha'],
        'psr_param': config_params['psr_param'],
        'correction': config_params['correction'],
        'winlen': config_params['winlen'],
        'binsize': config_params['binsize'],
        'spectrum': config_params['spectrum'],
        'n_surr': config_params['n_surr']
    }


def filter_spade_results(directory, annotations_path, output_path):
    """
    Main processing function for SPADE results.
    
    Parameters:
    -----------
    directory : str
        Input directory containing SPADE results
    annotations_path : str
        Path to annotations file
    output_path : str
        Output path for filtered results
        
    Returns:
    --------
    tuple: (patterns, loading_params, filter_params, config_params, spade_params)
    """
    print(f"Processing directory: {directory}")
    
    # Load and merge results from all files
    concepts, pv_spec_list, config_params, spade_params, loading_params = (
        load_results_from_directory(directory))
    
    print(f'Total concepts loaded: {len(concepts)}')
    
    # Extract parameters for convenience
    spectrum = config_params['spectrum']
    binsize = config_params['binsize'] * pq.s
    winlen = config_params['winlen']
    n_surr = config_params['n_surr']
    
    # Compute non-significant signatures (statistical testing)
    ns_signatures = compute_non_significant_signatures(
        pv_spec_list, concepts, config_params)
    
    # Apply pattern spectrum filter (PSF)
    concepts = apply_pattern_spectrum_filter(concepts, ns_signatures, config_params)
    
    # Apply pattern set reduction (PSR) if configured
    concepts = apply_pattern_set_reduction(concepts, ns_signatures, config_params)
    
    # Convert concepts to patterns format
    # Handle pv_spec properly based on surrogate testing
    pv_spec_for_output = pv_spec_list if (n_surr > 0 and len(pv_spec_list) > 0) else None
    
    patterns = spade.concept_output_to_patterns(
        concepts=concepts,
        pv_spec=pv_spec_for_output,
        winlen=winlen,
        binsize=binsize,
        spectrum=spectrum)
    
    print(f'Final number of patterns: {len(patterns)}')
    
    # Add neuron annotation information
    add_neuron_annotations(patterns, annotations_path)
    
    # Create filter parameters
    filter_params = create_filter_params(config_params)
    
    # Save filtered results
    np.save(output_path, [patterns, loading_params, filter_params, 
                         config_params, spade_params])
    
    print(f'Results saved to: {output_path}')
    
    return patterns, loading_params, filter_params, config_params, spade_params