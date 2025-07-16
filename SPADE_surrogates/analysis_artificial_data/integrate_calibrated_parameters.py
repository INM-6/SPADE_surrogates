#!/usr/bin/env python3
"""
Integration script to combine all calibrated parameters into the main parameter dictionary for artificial data.
"""

import numpy as np
import os
import glob
import yaml
from yaml import Loader

def integrate_calibrated_parameters():
    """
    Integrate all calibrated parameters from individual session/context/process files
    into the main parameter dictionary for artificial data.
    """
    print("=== Integrating Calibrated Parameters for Artificial Data ===")
    
    # Load configuration to get all sessions/contexts/processes
    with open("../configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    
    sessions = config['sessions']
    epochs = config['epochs']
    trialtypes = config['trialtypes']
    processes = config['processes']
    surr_method = config['surr_method']
    
    # Load original parameter dictionary
    original_param_dict = np.load('param_dict.npy', allow_pickle=True).item()
    
    # Create new parameter dictionary with calibrated values
    calibrated_param_dict = original_param_dict.copy()
    
    contexts = [f"{epoch}_{trialtype}" for epoch in epochs for trialtype in trialtypes]
    
    calibration_stats = {
        'total_contexts': 0,
        'calibrated_contexts': 0,
        'total_jobs': 0,
        'calibrated_jobs': 0,
        'failed_contexts': []
    }
    
    print(f"Looking for calibrated parameters...")
    print(f"Expected pattern: ../../results/empirical_calibration/{surr_method}/[process]/[session]/[context]_calibrated_params.npy")
    print()
    
    # Process each session, process, and context
    for session in sessions:
        if session not in calibrated_param_dict:
            print(f"Warning: Session {session} not in original parameter dictionary")
            continue
            
        print(f"Processing session: {session}")
        
        for process in processes:
            if process not in calibrated_param_dict[session]:
                print(f"  Process {process}: Not in original parameters")
                continue
                
            print(f"  Process: {process}")
            
            for context in contexts:
                calibration_stats['total_contexts'] += 1
                
                if context not in calibrated_param_dict[session][process]:
                    print(f"    {context}: Not in original parameters")
                    continue
                
                # Count total jobs from original dictionary
                original_job_count = len(calibrated_param_dict[session][process][context])
                calibration_stats['total_jobs'] += original_job_count
                
                # Look for calibrated parameters file
                calibration_file = f'../../results/empirical_calibration/{surr_method}/{process}/{session}/{context}_calibrated_params.npy'
                
                if os.path.exists(calibration_file):
                    try:
                        # Load calibrated parameters
                        calibrated_jobs = np.load(calibration_file, allow_pickle=True).item()
                        
                        # Replace jobs in the main dictionary
                        calibrated_param_dict[session][process][context] = calibrated_jobs
                        
                        calibration_stats['calibrated_contexts'] += 1
                        calibration_stats['calibrated_jobs'] += len(calibrated_jobs)
                        
                        print(f"    {context}: ✅ Integrated {len(calibrated_jobs)} calibrated jobs")
                        
                        # Show changes
                        for job_id, job_params in calibrated_jobs.items():
                            if job_params.get('calibration_applied', False):
                                original = job_params['original_min_occ']
                                calibrated = job_params['min_occ']
                                pattern_size = job_params['min_spikes']
                                change_pct = ((calibrated - original) / original) * 100
                                print(f"      Job {job_id} (size {pattern_size}): {original} → {calibrated} ({change_pct:+.1f}%)")
                        
                    except Exception as e:
                        print(f"    {context}: ❌ Failed to load calibration: {e}")
                        calibration_stats['failed_contexts'].append(f"{session}/{process}/{context}")
                else:
                    print(f"    {context}: ⚠️  No calibration file found")
        
        print()
    
    # Save integrated parameter dictionary
    np.save('param_dict_calibrated_integrated.npy', calibrated_param_dict)
    print(f"Saved integrated parameters to: param_dict_calibrated_integrated.npy")
    
    # Create backup of original and replace
    if not os.path.exists('param_dict_original_backup.npy'):
        np.save('param_dict_original_backup.npy', original_param_dict)
        print(f"Backed up original parameters to: param_dict_original_backup.npy")
    
    np.save('param_dict.npy', calibrated_param_dict)
    print(f"Updated main parameter dictionary: param_dict.npy")
    
    # Print integration summary
    print(f"\n=== Integration Summary ===")
    print(f"Total contexts: {calibration_stats['total_contexts']}")
    print(f"Calibrated contexts: {calibration_stats['calibrated_contexts']}")
    print(f"Total jobs: {calibration_stats['total_jobs']}")
    print(f"Calibrated jobs: {calibration_stats['calibrated_jobs']}")
    
    # Safe division for calibration coverage
    if calibration_stats['total_jobs'] > 0:
        coverage = calibration_stats['calibrated_jobs'] / calibration_stats['total_jobs'] * 100
        print(f"Calibration coverage: {coverage:.1f}%")
    else:
        print(f"Calibration coverage: N/A (no jobs found)")
    
    if calibration_stats['failed_contexts']:
        print(f"\nFailed contexts:")
        for context in calibration_stats['failed_contexts']:
            print(f"  {context}")
    
    # Save integration summary
    integration_summary = {
        'timestamp': str(np.datetime64('now')),  # Convert to string for JSON compatibility
        'calibration_method': 'empirical_no_surrogates',
        'data_type': 'artificial',
        'stats': calibration_stats,
        'integration_complete': True
    }
    
    np.save('calibration_integration_summary.npy', integration_summary)
    print(f"\nIntegration complete! Summary saved to: calibration_integration_summary.npy")
    
    return calibrated_param_dict, calibration_stats

if __name__ == "__main__":
    integrate_calibrated_parameters()