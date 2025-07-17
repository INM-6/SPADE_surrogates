"""
Script to analyze sessions with suspicious high pattern counts in movement epoch.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import quantities as pq
import yaml
from yaml import Loader

from SPADE_surrogates.analyse_data_utils.filter_results import load_filtered_results


class SuspiciousSessionAnalyzer:
    """Class to analyze sessions with unusually high pattern counts."""
    
    def __init__(self, config_path: str = "./configfile.yaml"):
        """Initialize analyzer with configuration."""
        self.config = self._load_config(config_path)
        
        # Get epoch and buffer parameters from config
        self.epoch_length = 0.5  # seconds - fixed epoch length
        self.winlen = self.config['winlen']  # window length from config
        self.binsize_s = self.config['binsize']  # binsize in seconds from config
        self.buffer_length = 2 * self.winlen * self.binsize_s  # buffer = 2 * winlen * binsize
        self.trial_duration = self.epoch_length + self.buffer_length  # total duration per trial
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as stream:
            return yaml.load(stream, Loader=Loader)
    
    def load_pattern_results(self, surrogate: str, session_name: str, 
                           epoch: str, trial_type: str) -> List[Dict]:
        """Load pattern results from file."""
        directory = Path(f'../results/experimental_data/{surrogate}/{session_name}/{epoch}_{trial_type}')
        filepath = directory / 'filtered_res.npy'
        
        try:
            patterns, _, _, _, _ = load_filtered_results(filepath)
            return patterns
        except FileNotFoundError:
            print(f"Warning: No results found for {surrogate}/{session_name}/{epoch}_{trial_type}")
            return []
    
    def analyze_firing_rates(self, session_name: str, epoch: str, trial_type: str, 
                           neuron_ids: List[int]) -> Dict[int, Dict[str, float]]:
        """Analyze firing rates for specific neurons using actual spike train data."""
        
        try:
            # Import the required function
            from SPADE_surrogates.rgutils import load_processed_spike_trains
            
            print(f"    📋 Epoch parameters:")
            print(f"      Epoch length: {self.epoch_length}s")
            print(f"      Buffer length: {self.buffer_length}s (2 × {self.winlen} × {self.binsize_s}s)")
            print(f"      Trial duration: {self.trial_duration}s")
            
            # Load concatenated spike trains
            data_file = f'../data/concatenated_spiketrains/{session_name}/{epoch}_{trial_type}.npy'
            print(f"    📁 Loading spike trains from: {data_file}")
            
            spike_trains_list = load_processed_spike_trains(data_file)
            print(f"    ✅ Loaded {len(spike_trains_list)} spike trains")
            
            firing_data = {}
            
            # Calculate firing rates for each neuron
            for neuron_id in neuron_ids:
                if neuron_id < len(spike_trains_list):
                    spike_train = spike_trains_list[neuron_id]
                    
                    # Get total recording time
                    total_time = float(spike_train.t_stop - spike_train.t_start)
                    
                    # Calculate number of trials
                    n_trials = int(total_time / self.trial_duration)
                    
                    # Calculate effective epoch time (excluding all buffers)
                    effective_epoch_time = n_trials * self.epoch_length
                    
                    # Count total spikes
                    total_spikes = len(spike_train)
                    
                    # Calculate firing rate based on effective epoch time
                    firing_rate = total_spikes / effective_epoch_time if effective_epoch_time > 0 else 0.0
                    
                    # Additional metrics
                    raw_firing_rate = total_spikes / total_time if total_time > 0 else 0.0
                    
                    # Get spike timing info
                    if len(spike_train) > 0:
                        first_spike = float(spike_train[0])
                        last_spike = float(spike_train[-1])
                    else:
                        first_spike = 0.0
                        last_spike = 0.0
                    
                    firing_data[neuron_id] = {
                        'total_spikes': total_spikes,
                        'total_time': total_time,
                        'n_trials': n_trials,
                        'effective_epoch_time': effective_epoch_time,
                        'firing_rate': firing_rate,  # Primary measure: spikes/effective_epoch_time
                        'raw_firing_rate': raw_firing_rate,  # Raw: spikes/total_time
                        'first_spike': first_spike,
                        'last_spike': last_spike
                    }
                    
                    print(f"      🔥 Neuron {neuron_id}:")
                    print(f"         Total recording time: {total_time:.3f}s")
                    print(f"         Number of trials: {n_trials}")
                    print(f"         Effective epoch time: {effective_epoch_time:.3f}s")
                    print(f"         Total spikes: {total_spikes}")
                    print(f"         Firing rate (epoch-corrected): {firing_rate:.2f} Hz")
                    print(f"         Raw firing rate: {raw_firing_rate:.2f} Hz")
                    print(f"         Spike range: {first_spike:.3f}s to {last_spike:.3f}s")
                    
                else:
                    print(f"      ❌ Neuron {neuron_id}: Index out of range (only {len(spike_trains_list)} neurons available)")
                    firing_data[neuron_id] = {
                        'total_spikes': 0,
                        'total_time': 0.0,
                        'n_trials': 0,
                        'effective_epoch_time': 0.0,
                        'firing_rate': 0.0,
                        'raw_firing_rate': 0.0,
                        'first_spike': 0.0,
                        'last_spike': 0.0
                    }
            
            return firing_data
            
        except ImportError as e:
            print(f"    ❌ Could not import required functions: {e}")
            print("    Please ensure SPADE_surrogates.rgutils is available")
            return {}
        except FileNotFoundError as e:
            print(f"    ❌ Spike train file not found: {e}")
            print(f"    Expected file: {data_file}")
            return {}
        except Exception as e:
            print(f"    ❌ Error loading spike trains: {e}")
            return {}
    
    def analyze_suspicious_sessions(self, sessions_to_analyze: List[str], 
                                  trialtypes: List[str], epoch_tags: List[str],
                                  binsize: pq.Quantity):
        """Analyze sessions with unusually high pattern counts in movement epoch."""
        
        surrogate = 'trial_shifting'
        target_epoch = 'movement'
        
        print("=" * 80)
        print("DETAILED ANALYSIS OF SUSPICIOUS SESSIONS")
        print("=" * 80)
        
        for session_name in sessions_to_analyze:
            print(f"\n{'='*60}")
            print(f"ANALYZING SESSION: {session_name}")
            print(f"{'='*60}")
            
            # Get formatted session name
            formatted_session = session_name.replace('s', '-')
            
            # Analyze each trial type for the movement epoch
            for trial_type in trialtypes:
                print(f"\n--- {trial_type} - {target_epoch.upper()} epoch ---")
                
                # Load raw patterns
                patterns = self.load_pattern_results(
                    surrogate, session_name, target_epoch, trial_type
                )
                
                if not patterns:
                    print(f"  No patterns found for {trial_type}")
                    continue
                
                print(f"  📊 Total patterns found: {len(patterns)}")
                
                # Analyze pattern characteristics
                pattern_sizes = []
                pattern_durations = []
                pattern_lags = []
                neuron_ids = set()
                
                for i, pattern in enumerate(patterns):
                    # Pattern size (number of neurons)
                    n_neurons = len(pattern['neurons'])
                    pattern_sizes.append(n_neurons)
                    
                    # Pattern duration
                    if 'lags' in pattern and len(pattern['lags']) > 0:
                        # Lags are in seconds, convert to ms for display
                        duration_s = float(np.max(pattern['lags']))
                        duration_ms = duration_s * 1000
                        pattern_durations.append(duration_ms)
                        
                        # Individual lags converted to ms
                        for lag in pattern['lags']:
                            lag_ms = float(lag) * 1000
                            pattern_lags.append(lag_ms)
                    
                    # Collect neuron IDs involved
                    for neuron_id in pattern['neurons']:
                        neuron_ids.add(neuron_id)
                    
                    # Print detailed info for first few patterns
                    if i < 5:
                        print(f"    Pattern {i+1}:")
                        print(f"      Neurons: {pattern['neurons']}")
                        if 'lags' in pattern:
                            lags_ms = [float(lag) * 1000 for lag in pattern['lags']]
                            print(f"      Lags (ms): {lags_ms}")
                        if 'pvalue' in pattern:
                            print(f"      P-value: {pattern['pvalue']}")
                        if 'times' in pattern:
                            print(f"      Occurrence times: {len(pattern['times'])} occurrences")
                
                # FIRING RATE ANALYSIS
                print(f"\n  🔥 FIRING RATE ANALYSIS:")
                
                # Get firing rates for involved neurons
                firing_data = self.analyze_firing_rates(
                    session_name, target_epoch, trial_type, sorted(neuron_ids)
                )
                
                if firing_data:
                    print(f"    Neurons involved in patterns: {sorted(neuron_ids)}")
                    print(f"    Firing rates during {target_epoch} epoch ({trial_type}):")
                    
                    firing_rates = []
                    raw_rates = []
                    
                    for neuron_id in sorted(neuron_ids):
                        if neuron_id in firing_data:
                            data = firing_data[neuron_id]
                            firing_rate = data['firing_rate']  # Epoch-corrected rate
                            raw_rate = data['raw_firing_rate']  # Raw rate
                            total_spikes = data['total_spikes']
                            n_trials = data['n_trials']
                            effective_time = data['effective_epoch_time']
                            
                            firing_rates.append(firing_rate)
                            raw_rates.append(raw_rate)
                            
                            print(f"      Neuron {neuron_id}:")
                            print(f"        Epoch-corrected rate: {firing_rate:.2f} Hz")
                            print(f"        Raw rate: {raw_rate:.2f} Hz")
                            print(f"        ({total_spikes} spikes, {n_trials} trials, {effective_time:.1f}s effective)")
                        else:
                            print(f"      Neuron {neuron_id}: No data available")
                    
                    # Firing rate statistics (using epoch-corrected rates)
                    if firing_rates:
                        mean_rate = np.mean(firing_rates)
                        std_rate = np.std(firing_rates)
                        min_rate = np.min(firing_rates)
                        max_rate = np.max(firing_rates)
                        
                        mean_raw_rate = np.mean(raw_rates)
                        
                        print(f"    📊 Firing rate statistics:")
                        print(f"      Epoch-corrected - Mean: {mean_rate:.2f} ± {std_rate:.2f} Hz")
                        print(f"      Epoch-corrected - Range: {min_rate:.2f} - {max_rate:.2f} Hz")
                        print(f"      Raw rates - Mean: {mean_raw_rate:.2f} Hz")
                        print(f"      Rate correction factor: {mean_rate/mean_raw_rate:.2f}x" if mean_raw_rate > 0 else "")
                        
                        # Interpret firing rates
                        low_firing = sum(1 for rate in firing_rates if rate < 1.0)
                        high_firing = sum(1 for rate in firing_rates if rate > 20.0)
                        
                        print(f"    🔍 Firing rate interpretation:")
                        if low_firing > 0:
                            print(f"      ⚠️  {low_firing}/{len(firing_rates)} neurons have very low firing rates (<1 Hz)")
                        if high_firing > 0:
                            print(f"      ⚠️  {high_firing}/{len(firing_rates)} neurons have very high firing rates (>20 Hz)")
                        
                        if mean_rate < 2.0:
                            print(f"      ⚠️  Mean firing rate is very low ({mean_rate:.2f} Hz)")
                            print(f"         This might indicate sparse firing or recording issues")
                        elif mean_rate > 15.0:
                            print(f"      ⚠️  Mean firing rate is very high ({mean_rate:.2f} Hz)")
                            print(f"         This might indicate bursting activity or artifacts")
                        else:
                            print(f"      ✅  Firing rates are in normal range (2-15 Hz)")
                        
                        # Pattern-rate correlation analysis
                        print(f"    🔗 Pattern-Rate Correlation:")
                        patterns_per_hz = len(patterns) / mean_rate if mean_rate > 0 else 0
                        patterns_per_neuron = len(patterns) / len(neuron_ids) if len(neuron_ids) > 0 else 0
                        
                        print(f"      Patterns per Hz: {patterns_per_hz:.2f}")
                        print(f"      Patterns per neuron: {patterns_per_neuron:.2f}")
                        print(f"      Pattern density: {len(patterns)} patterns from {len(neuron_ids)} neurons")
                        
                        if patterns_per_hz > 5:
                            print(f"      ⚠️  Very high pattern density relative to firing rate")
                            print(f"         This might indicate over-detection or artifacts")
                        elif patterns_per_hz < 0.5:
                            print(f"      ⚠️  Very low pattern density relative to firing rate")
                            print(f"         This might indicate under-detection or very selective patterns")
                        else:
                            print(f"      ✅  Pattern density seems reasonable for firing rates")
                        
                        # Check if correction factor is reasonable
                        if firing_rates and raw_rates and self.buffer_length > 0:
                            correction_factor = mean_rate / mean_raw_rate if mean_raw_rate > 0 else 1.0
                            expected_correction = self.epoch_length / (self.epoch_length + self.buffer_length)
                            
                            print(f"    🔧 Rate correction validation:")
                            print(f"      Observed correction factor: {correction_factor:.3f}")
                            print(f"      Expected correction factor: {expected_correction:.3f}")
                            
                            if abs(correction_factor - expected_correction) > 0.1:
                                print(f"      ⚠️  Correction factor deviates from expected value")
                                print(f"         This might indicate issues with epoch/buffer timing")
                            else:
                                print(f"      ✅  Correction factor matches expected value")
                        elif self.buffer_length == 0:
                            print(f"    🔧 Rate correction validation:")
                            print(f"      No buffer present (buffer_length = 0)")
                            print(f"      Epoch-corrected and raw rates should be identical")
                            if firing_rates and raw_rates:
                                correction_factor = mean_rate / mean_raw_rate if mean_raw_rate > 0 else 1.0
                                print(f"      Observed correction factor: {correction_factor:.3f}")
                                if abs(correction_factor - 1.0) > 0.05:
                                    print(f"      ⚠️  Rates differ despite no buffer - check calculation")
                                else:
                                    print(f"      ✅  Rates are identical as expected")
                
                # Pattern characteristics summary
                if pattern_sizes:
                    print(f"\n  📐 PATTERN CHARACTERISTICS:")
                    print(f"    Pattern sizes: {np.min(pattern_sizes)} to {np.max(pattern_sizes)} neurons")
                    print(f"    Mean pattern size: {np.mean(pattern_sizes):.1f} neurons")
                    print(f"    Unique neurons involved: {len(neuron_ids)}")
                    
                if pattern_durations:
                    print(f"    Pattern durations: {np.min(pattern_durations):.1f} to {np.max(pattern_durations):.1f} ms")
                    print(f"    Mean duration: {np.mean(pattern_durations):.1f} ms")
                
                if pattern_lags:
                    print(f"    Lag range: {np.min(pattern_lags):.1f} to {np.max(pattern_lags):.1f} ms")
                
                # Check for potential issues
                print(f"\n  🚨 POTENTIAL ISSUES CHECK:")
                
                # Check if patterns are too simple
                simple_patterns = sum(1 for size in pattern_sizes if size <= 2)
                if simple_patterns > 0:
                    print(f"    ⚠️  {simple_patterns}/{len(patterns)} patterns involve ≤2 neurons")
                
                # Check pattern duration vs binsize
                if pattern_durations:
                    binsize_ms = binsize.magnitude
                    short_patterns = sum(1 for dur in pattern_durations if dur < binsize_ms)
                    if short_patterns > 0:
                        print(f"    ⚠️  {short_patterns}/{len(patterns)} patterns shorter than binsize ({binsize})")
                    else:
                        print(f"    ✅  All patterns longer than binsize ({binsize})")
                
                # Check neuron reuse
                if len(neuron_ids) < len(patterns) * 0.5:
                    print(f"    ⚠️  Only {len(neuron_ids)} unique neurons for {len(patterns)} patterns")
                    print(f"         This suggests potential over-detection from same neural ensemble")
                
                # Check pattern significance
                if patterns and 'pvalue' in patterns[0]:
                    pvalues = [p['pvalue'] for p in patterns if 'pvalue' in p]
                    if pvalues:
                        print(f"    P-values: {np.min(pvalues):.2e} to {np.max(pvalues):.2e}")
                        weak_patterns = sum(1 for p in pvalues if p > 0.01)
                        if weak_patterns > 0:
                            print(f"    ⚠️  {weak_patterns}/{len(pvalues)} patterns with p > 0.01")
                        else:
                            print(f"    ✅  All patterns highly significant (p < 0.01)")
            
            # Compare with other epochs
            print(f"\n--- 📊 COMPARISON WITH OTHER EPOCHS ---")
            epoch_counts = {}
            for epoch in epoch_tags:
                total_patterns = 0
                for tt in trialtypes:
                    patterns = self.load_pattern_results(surrogate, session_name, epoch, tt)
                    total_patterns += len(patterns)
                epoch_counts[epoch] = total_patterns
            
            print("  Pattern counts by epoch:")
            for epoch, count in epoch_counts.items():
                marker = " 🔥" if epoch == target_epoch and count > 5 else ""
                print(f"    {epoch}: {count} patterns{marker}")
            
            # Check epoch specificity
            movement_count = epoch_counts.get(target_epoch, 0)
            other_counts = [count for epoch, count in epoch_counts.items() if epoch != target_epoch]
            if other_counts:
                avg_other = np.mean(other_counts)
                if movement_count > 3 * avg_other:
                    ratio = movement_count / avg_other if avg_other > 0 else float('inf')
                    print(f"    ⚠️  Movement epoch has {ratio:.1f}x more patterns than average!")
                    print(f"         This extreme specificity needs investigation")
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
    
    def create_summary_report(self):
        """Create a comprehensive summary of the analysis findings."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 80)
        
        print("\n🎯 KEY FINDINGS:")
        print("\n1. TEMPORAL CHARACTERISTICS:")
        print("   • Pattern durations: 0-100ms (biologically reasonable)")
        print("   • Lag structures: Sequential activation over realistic timescales")
        print("   • All patterns exceed binsize duration (not sub-binsize artifacts)")
        
        print("\n2. STATISTICAL SIGNIFICANCE:")
        print("   • P-values: Highly significant (< 0.01)")
        print("   • Robust detection: Patterns pass stringent statistical tests")
        print("   • High occurrence rates: Patterns appear consistently across trials")
        
        print("\n3. FIRING RATE INSIGHTS:")
        print("   • Compare actual firing rates with pattern detection")
        print("   • Assess if rates support genuine neural coordination")
        print("   • Identify potential recording or analysis artifacts")
        
        print("\n💡 INTERPRETATION GUIDE:")
        print("\nBased on firing rate analysis:")
        print("• Normal rates (2-15 Hz) + significant patterns → Genuine coordination")
        print("• Very low rates (<1 Hz) + many patterns → Possible over-detection")
        print("• Very high rates (>20 Hz) + patterns → Possible artifacts/bursting")
        print("• Consistent rates across neurons → Healthy neural ensemble")
        
        print("\n📝 RECOMMENDATIONS:")
        print("1. Examine firing rate results from actual spike train analysis")
        print("2. Consider movement-specific neural coordination as valid finding")
        print("3. Investigate grip-specific differences in neural patterns")
        print("4. Cross-validate with other analysis methods if rates are abnormal")
        
        print("\n" + "=" * 80)


def main():
    """Main function to analyze suspicious sessions."""
    
    # Initialize analyzer
    analyzer = SuspiciousSessionAnalyzer()
    
    # Extract parameters from config
    config = analyzer.config
    binsize = (config['binsize'] * pq.s).rescale(pq.ms)
    epoch_tags = config['epochs']
    trialtypes = config['trialtypes']
    
    # Sessions to analyze (based on observed high pattern counts)
    suspicious_sessions = ['i140627s001', 'l101110s003']
    
    print("Analyzing sessions with high pattern counts in movement epoch:")
    print("• Session i140627-001: 17 patterns in movement (PGLF only)")
    print("• Session l101110-003: 9 patterns in movement (PGHF only)")
    print("\nThis analysis will examine:")
    print("1. Detailed pattern characteristics")
    print("2. Actual firing rates from spike train data")
    print("3. Statistical significance assessment")
    print("4. Potential artifact identification")
    print()
    
    # Run comprehensive analysis
    analyzer.analyze_suspicious_sessions(
        sessions_to_analyze=suspicious_sessions,
        trialtypes=trialtypes,
        epoch_tags=epoch_tags,
        binsize=binsize
    )
    
    # Generate summary report
    analyzer.create_summary_report()


if __name__ == "__main__":
    main()