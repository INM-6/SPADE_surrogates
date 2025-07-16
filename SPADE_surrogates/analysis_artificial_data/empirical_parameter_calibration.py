#!/usr/bin/env python3
"""
Empirical Parameter Calibration for SPADE Analysis on Artificial Data

Optimizes min_occ parameters by running SPADE on original data (no surrogates)
to find values that yield approximately 2×winlen patterns per pattern size.

Usage: python empirical_parameter_calibration.py SESSION CONTEXT PROCESS
Example: python empirical_parameter_calibration.py l10121s001 start_PGHF poisson
"""

import numpy as np
import os
import sys
import argparse
import math
import signal
import quantities as pq

# Import utilities
from SPADE_surrogates.analyse_data_utils.spade_analysis_utils import (
    load_configuration, load_artificial_data, apply_firing_rate_threshold, extract_spade_parameters
)
from SPADE_surrogates.analyse_data_utils.estimate_number_occurrences import (
    extract_trial_count_from_concatenated_data, calculate_dynamic_abs_min_occ
)
from elephant.spade import spade


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Empirical SPADE parameter calibration for artificial data")
    parser.add_argument("session", help="Session identifier (e.g., l101210s001)")
    parser.add_argument("context", help="Context: epoch_trialtype (e.g., start_PGHF)")
    parser.add_argument("process", help="Point process to analyze (e.g., poisson)")
    return parser.parse_args()


def test_min_occ(spike_trains, params, min_occ_value, timeout=60):
    """
    Test a min_occ value and return number of patterns found.
    
    Returns
    -------
    int or None : Number of patterns found, None if timeout/error
    """
    if min_occ_value < 10:
        timeout = min(timeout, 30)  # Shorter timeout for risky values
    
    # Setup timeout protection
    class TimeoutError(Exception): pass
    def timeout_handler(signum, frame): raise TimeoutError()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        results = spade(
            spiketrains=spike_trains,
            binsize=params['binsize'],
            winlen=params['winlen'],
            min_occ=min_occ_value,
            min_spikes=params['min_spikes'],
            max_spikes=params['max_spikes'],
            min_neu=params['min_neu'],
            dither=params['dither'],
            n_surr=0,  # No surrogates for calibration
            alpha=params['alpha'],
            psr_param=params['psr_param'],
            spectrum=params['spectrum'],
            output_format='concepts'
        )
        signal.alarm(0)
        return len(results.get('patterns', []))
        
    except TimeoutError:
        signal.alarm(0)
        print(f"⏰ TIMEOUT ", end="")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"❌ ERROR: {e}")
        return None


def search_optimal_min_occ(spike_trains, params, start_min_occ, target, min_threshold, step_size):
    """
    Search for optimal min_occ using bidirectional search strategy.
    
    Strategy:
    - If patterns >= target: search upward to get closer to target
    - If patterns < target: search downward to reach target
    
    Returns
    -------
    int : Optimal min_occ value closest to target
    """
    print(f"🔍 Testing start value {start_min_occ}... ", end="")
    start_patterns = test_min_occ(spike_trains, params, start_min_occ)
    
    if start_patterns is None:
        print(f"Failed, returning original: {start_min_occ}")
        return start_min_occ
    
    print(f"{start_patterns} patterns")
    
    if start_patterns >= target:
        print(f"📈 Too many patterns ({start_patterns} >> {target}), searching upward to reduce...")
        return search_upward(spike_trains, params, start_min_occ, target, step_size)
    else:
        print(f"📉 Need more patterns ({start_patterns} < {target}), searching downward...")
        return search_downward(spike_trains, params, start_min_occ, target, min_threshold, step_size)


def search_upward(spike_trains, params, start_min_occ, target, step_size):
    """Search upward to reduce number of patterns and get closer to target."""
    current = start_min_occ
    best_min_occ = start_min_occ
    best_patterns = test_min_occ(spike_trains, params, start_min_occ) or 0
    
    max_iterations = 50
    max_search_limit = max(start_min_occ * 3, 500)  # Reasonable upper limit
    
    for i in range(1, max_iterations + 1):
        test_value = current + step_size
        
        if test_value > max_search_limit:
            print(f"🔝 Hit search limit ({max_search_limit})")
            break
            
        print(f"[{i:2d}] Testing {test_value}... ", end="")
        patterns = test_min_occ(spike_trains, params, test_value)
        
        if patterns is None:
            break
            
        print(f"{patterns} patterns")
        
        # Keep the value closest to target (but still >= target)
        if patterns >= target:
            if patterns < best_patterns:  # Closer to target
                best_min_occ = test_value
                best_patterns = patterns
        else:
            # Went below target, previous value was optimal
            print(f"📉 Below target, stopping")
            break
            
        current = test_value
        
        # If we're very close to target, we can stop
        if target <= patterns <= target * 1.2:  # Within 20% of target
            print(f"🎯 Close enough to target!")
            best_min_occ = test_value
            best_patterns = patterns
            break
    
    print(f"✅ Best found: {best_min_occ} ({best_patterns} patterns)")
    return best_min_occ


def search_downward(spike_trains, params, start_min_occ, target, min_threshold, step_size):
    """Search downward to increase number of patterns and reach target."""
    current = start_min_occ
    best_min_occ = start_min_occ
    best_patterns = test_min_occ(spike_trains, params, start_min_occ) or 0
    
    for i in range(1, 51):  # Max 50 iterations
        test_value = current - step_size
        
        if test_value < min_threshold:
            print(f"🛡️ Hit memory protection boundary ({min_threshold})")
            if min_threshold != best_min_occ:
                print(f"Testing boundary... ", end="")
                boundary_patterns = test_min_occ(spike_trains, params, min_threshold)
                if boundary_patterns and boundary_patterns >= target:
                    return min_threshold
            break
        
        print(f"[{i:2d}] Testing {test_value}... ", end="")
        patterns = test_min_occ(spike_trains, params, test_value)
        
        if patterns is None:
            break
        
        print(f"{patterns} patterns")
        
        if patterns > best_patterns:
            best_min_occ = test_value
            best_patterns = patterns
        
        if patterns >= target:
            print(f"🎯 TARGET REACHED!")
            return test_value
        
        current = test_value
        
        # Switch to finer steps for pattern size 2 after a few iterations
        if params['min_spikes'] == 2 and i >= 3 and step_size > 1:
            step_size = 1
            print(f"🔧 Switching to fine steps")
    
    print(f"✅ Best found: {best_min_occ} ({best_patterns} patterns)")
    return best_min_occ


def recalculate_memory_threshold(session, epoch, trialtype, process, spike_trains, config):
    """Recalculate dynamic memory protection threshold for artificial data."""
    try:
        n_trials = extract_trial_count_from_concatenated_data(spike_trains, config)
        if n_trials is None:
            return calculate_dynamic_abs_min_occ(session, epoch, trialtype, process)
        
        n_trials = max(3, min(n_trials, 1000))  # Reasonable bounds
        threshold = math.ceil(n_trials / 3.0)
        print(f"Memory threshold: {n_trials} trials → {threshold}")
        return threshold
        
    except Exception as e:
        print(f"Error calculating threshold: {e}")
        return calculate_dynamic_abs_min_occ(session, epoch, trialtype, process)


# =============================================================================
# MAIN CALIBRATION
# =============================================================================

def calibrate_session_context_process(session, context, process):
    """Main calibration function for artificial data."""
    print(f"🎯 EMPIRICAL CALIBRATION - ARTIFICIAL DATA")
    print(f"Session: {session}, Context: {context}, Process: {process}")
    
    # Load data and configuration
    config = load_configuration()
    param_dict = np.load('param_dict.npy', allow_pickle=True).item()
    
    if (session not in param_dict or 
        process not in param_dict[session] or 
        context not in param_dict[session][process]):
        raise ValueError(f"Session {session} process {process} context {context} not found")
    
    jobs = param_dict[session][process][context]
    winlen = config['winlen']
    target_patterns = 2 * winlen
    
    print(f"Target: ≥{target_patterns} patterns (2×winlen), Jobs: {len(jobs)}")
    
    # Load spike train data
    epoch, trialtype = context.split('_', 1)
    spike_trains, _ = load_artificial_data(session, epoch, trialtype, process)
    spike_trains = apply_firing_rate_threshold(spike_trains, session, config['firing_rate_threshold'])
    
    print(f"Data: {len(spike_trains)} spike trains")
    
    # Calculate memory protection threshold
    memory_threshold = recalculate_memory_threshold(session, epoch, trialtype, process, spike_trains, config)
    print(f"Memory protection: ≥{memory_threshold}")
    
    # Calibrate each job
    calibrated_params = {}
    
    for job_id in sorted(jobs.keys()):
        job_params = jobs[job_id]
        pattern_size = job_params.get('min_spikes', 2)
        original_min_occ = job_params['min_occ']
        
        print(f"\n🎯 JOB {job_id} (Pattern Size {pattern_size})")
        print(f"Original min_occ: {original_min_occ}")
        
        # Extract SPADE parameters
        spade_params = extract_spade_parameters(config, job_params)
        if spade_params.get('min_occ') != original_min_occ:
            print(f"⚠️ Using param_dict min_occ ({original_min_occ}) over extracted ({spade_params.get('min_occ')})")
            spade_params['min_occ'] = original_min_occ
        
        # Determine search strategy
        step_size = 5 if pattern_size == 2 else 1
        
        # Run calibration
        optimal_min_occ = search_optimal_min_occ(
            spike_trains, spade_params, original_min_occ, 
            target_patterns, memory_threshold, step_size
        )
        
        # Store results
        calibrated_job_params = job_params.copy()
        calibrated_job_params.update({
            'min_occ': optimal_min_occ,
            'calibration_applied': True,
            'original_min_occ': original_min_occ,
            'target_patterns': target_patterns,
            'dynamic_abs_min_occ': memory_threshold
        })
        calibrated_params[job_id] = calibrated_job_params
        
        # Summary
        change = optimal_min_occ - original_min_occ
        change_pct = (change / original_min_occ) * 100
        safety_ratio = optimal_min_occ / memory_threshold
        
        print(f"Result: {optimal_min_occ} ({change:+d}, {change_pct:+.1f}%)")
        print(f"Safety: {safety_ratio:.1f}× above threshold")
    
    return calibrated_params


def main():
    """Main execution."""
    args = parse_arguments()
    
    print(f"🚀 Starting calibration: {args.session} {args.context} {args.process}")
    
    try:
        # Load configuration to get surrogate method
        config = load_configuration()
        surr_method = config.get('surr_method', 'trial_shifting')
        
        print(f"Surrogate method from config: {surr_method}")
        
        # Run calibration
        calibrated_params = calibrate_session_context_process(args.session, args.context, args.process)
        
        # Save results with correct path structure including surrogate method and process
        output_dir = f'../../results/empirical_calibration/{surr_method}/{args.process}/{args.session}'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{args.context}_calibrated_params.npy')
        
        print(f"Saving calibration results to: {output_file}")
        np.save(output_file, calibrated_params)
        
        # Verify the file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ File successfully created: {output_file} ({file_size} bytes)")
        else:
            raise FileNotFoundError(f"Failed to create output file: {output_file}")
        
        # Summary statistics
        original_values = [p['original_min_occ'] for p in calibrated_params.values()]
        calibrated_values = [p['min_occ'] for p in calibrated_params.values()]
        
        n_reduced = sum(1 for o, c in zip(original_values, calibrated_values) if c < o)
        n_increased = sum(1 for o, c in zip(original_values, calibrated_values) if c > o)
        n_unchanged = len(calibrated_params) - n_reduced - n_increased
        
        avg_change_pct = ((np.mean(calibrated_values) - np.mean(original_values)) / np.mean(original_values)) * 100
        
        print(f"\n🎉 CALIBRATION COMPLETE")
        print(f"Jobs: {len(calibrated_params)} ({n_reduced} reduced, {n_increased} increased, {n_unchanged} unchanged)")
        print(f"Average change: {avg_change_pct:+.1f}%")
        print(f"Final output: {output_file}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()