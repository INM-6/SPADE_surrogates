"""
Improved script for creating artificial data that mimics experimental data features.

This script generates artificial spike trains using Poisson processes with refractory 
periods and Gamma processes, based on statistics estimated from real experimental data.

Updated to use load_processed_spike_trains function for loading concatenated data.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import sys
import yaml
from yaml import Loader

import numpy as np
import quantities as pq
import neo
from scipy.special import gamma as gamma_function
from scipy.optimize import brentq
from tqdm import tqdm

# Import elephant modules
import elephant.spike_train_generation as stg
import elephant.statistics as stat
import elephant.kernels as kernels


def save_processed_data(data: List, output_path: Path) -> bool:
    """
    Save processed spike train data using the same format as the main processing pipeline.
    
    This function uses the same save logic as the concatenated data processing script
    to ensure consistency across all data files.
    
    Parameters
    ----------
    data : List
        Processed spike train data (list of neo.SpikeTrain objects)
    output_path : Path
        Output file path
        
    Returns
    -------
    bool
        True if save was successful, False otherwise
    """
    try:
        if data:
            # Convert spike trains to standardized format for saving
            processed_data = []
            
            total_spikes_check = 0  # For verification
            
            for i, spike_train in enumerate(data):
                # Extract spike times - handle different input types
                if hasattr(spike_train, 'times'):
                    spike_times = spike_train.times.magnitude
                elif hasattr(spike_train, 'magnitude'):
                    spike_times = spike_train.magnitude
                else:
                    # Fallback: try to convert to array
                    spike_times = np.array(spike_train).flatten()
                
                total_spikes_check += len(spike_times)
                
                # Extract essential data from each spike train
                spike_data = {
                    'times': spike_times,  # Actual spike times array
                    'units': str(spike_train.units) if hasattr(spike_train, 'units') else 's',
                    't_start': float(spike_train.t_start.magnitude) if hasattr(spike_train, 't_start') else 0.0,
                    't_stop': float(spike_train.t_stop.magnitude) if hasattr(spike_train, 't_stop') else 0.0,
                    'annotations': {},
                    'spike_train_id': i
                }
                
                # Handle annotations carefully
                if hasattr(spike_train, 'annotations') and spike_train.annotations:
                    for key, value in spike_train.annotations.items():
                        try:
                            if isinstance(value, (np.ndarray, list)):
                                if isinstance(value, np.ndarray):
                                    spike_data['annotations'][key] = value.tolist()
                                else:
                                    spike_data['annotations'][key] = value
                            elif hasattr(value, 'magnitude'):  # quantities
                                spike_data['annotations'][key] = float(value.magnitude)
                            elif isinstance(value, (np.integer, np.floating)):
                                spike_data['annotations'][key] = float(value)
                            else:
                                spike_data['annotations'][key] = value
                        except Exception as e:
                            # Skip problematic annotations
                            print(f"      ⚠️ Skipping annotation '{key}': {e}")
                            continue
                
                processed_data.append(spike_data)
            
            # Verify data before saving
            saved_spike_counts = [len(st['times']) for st in processed_data]
            total_saved_spikes = sum(saved_spike_counts)
            
            # Save as numpy array with allow_pickle=True
            np.save(output_path, np.array(processed_data, dtype=object), allow_pickle=True)
            
            # Verify what was actually saved
            try:
                verification_data = np.load(output_path, allow_pickle=True)
                actual_total_spikes = sum(len(st['times']) for st in verification_data) if len(verification_data) > 0 else 0
                
                print(f"      💾 Saved {len(processed_data)} spike trains to: {output_path}")
                print(f"         File size: {output_path.stat().st_size / 1024:.1f} KB")
                print(f"         VERIFICATION: Total spikes saved: {actual_total_spikes:,}")
                
                if actual_total_spikes == total_spikes_check:
                    print(f"         ✅ Spike count verification PASSED")
                else:
                    print(f"         ⚠️ Spike count mismatch: Expected {total_spikes_check}, Saved {actual_total_spikes}")
                
                # Log statistics about the saved data
                if actual_total_spikes > 0:
                    spike_counts = [len(st['times']) for st in verification_data]
                    mean_spikes = np.mean(spike_counts)
                    print(f"         Mean spikes per unit: {mean_spikes:.1f}")
                    print(f"         Spike count range: {min(spike_counts)} - {max(spike_counts)}")
                    
            except Exception as e:
                print(f"         ⚠️ Could not verify saved data: {e}")
                
        else:
            # Save empty array
            np.save(output_path, np.array([]), allow_pickle=True)
            print(f"      💾 Saved empty array to: {output_path}")
            
        return True
        
    except Exception as e:
        print(f"      ❌ Failed to save data to {output_path}: {e}")
        import traceback
        print(f"         Full traceback:\n{traceback.format_exc()}")
        return False
    
def load_processed_spike_trains(file_path: str) -> List[neo.SpikeTrain]:
    """
    Load processed spike train data from a saved file and reconstruct neo.SpikeTrain objects.
    
    This function is designed to work with the data format saved by the improved
    spike train processing pipeline.
    
    Parameters
    ----------
    file_path : str
        Path to the saved spike train file (.npy format)
        
    Returns
    -------
    List[neo.SpikeTrain]
        List of reconstructed neo.SpikeTrain objects
        
    Example
    -------
    >>> spike_trains = load_processed_spike_trains("../data/concatenated_spiketrains/i140613s001/start_PGHF.npy")
    >>> print(f"Loaded {len(spike_trains)} spike trains")
    """
    try:
        print(f"   🔄 Loading spike train data from: {file_path}")
        
        # Load the data
        data = np.load(file_path, allow_pickle=True)
        
        if len(data) == 0:
            print(f"   ⚠️ Empty data file: {file_path}")
            return []
        
        print(f"   📊 Found {len(data)} spike train records")
        
        # Reconstruct spike trains
        spike_trains = []
        
        for i, spike_data in enumerate(data):
            try:
                # Handle different data formats that might be saved
                if isinstance(spike_data, dict):
                    # New format with structured data
                    times = spike_data['times']
                    units_str = spike_data.get('units', 's')
                    t_start = spike_data.get('t_start', 0.0)
                    t_stop = spike_data.get('t_stop', None)
                    annotations = spike_data.get('annotations', {})
                    
                elif hasattr(spike_data, 'times'):
                    # Already a neo.SpikeTrain object
                    spike_trains.append(spike_data)
                    continue
                    
                else:
                    print(f"   ⚠️ Unknown data format for spike train {i}: {type(spike_data)}")
                    continue
                
                # Parse units
                if units_str == 's' or units_str == '1.0 s':
                    units = pq.s
                elif units_str == 'ms' or units_str == '1.0 ms':
                    units = pq.ms
                else:
                    units = pq.s  # Default fallback
                
                # Create spike train
                if t_stop is None:
                    # Estimate t_stop from the maximum time if not provided
                    if len(times) > 0:
                        t_stop = max(times) + 0.1  # Add small buffer
                    else:
                        t_stop = 1.0  # Default for empty spike trains
                
                spike_train = neo.SpikeTrain(
                    times * units,
                    t_start=t_start * units,
                    t_stop=t_stop * units
                )
                
                # Restore annotations
                if annotations:
                    spike_train.annotations.update(annotations)
                
                spike_trains.append(spike_train)
                
            except Exception as e:
                print(f"   ⚠️ Error reconstructing spike train {i}: {e}")
                continue
        
        print(f"   ✅ Successfully reconstructed {len(spike_trains)} spike trains")
        
        # Log some statistics
        if spike_trains:
            spike_counts = [len(st) for st in spike_trains]
            total_spikes = sum(spike_counts)
            print(f"      Total spikes across all units: {total_spikes:,}")
            
            if total_spikes > 0:
                print(f"      Mean spikes per unit: {np.mean(spike_counts):.1f}")
                print(f"      Spike count range: {min(spike_counts)} - {max(spike_counts)}")
        
        return spike_trains
        
    except Exception as e:
        print(f"   ❌ Error loading spike trains from {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


class ArtificialDataGenerator:
    """
    Class for generating artificial spike train data based on experimental data statistics.
    """
    
    def __init__(self, max_refractory: pq.Quantity = 4 * pq.ms, 
                 sampling_period: pq.Quantity = 0.1 * pq.ms,
                 sigma: pq.Quantity = 100 * pq.ms,
                 trial_length: pq.Quantity = 500 * pq.ms):
        """
        Initialize the artificial data generator.
        
        Parameters
        ----------
        max_refractory : pq.Quantity, optional
            Maximum refractory period allowed, by default 4*pq.ms
        sampling_period : pq.Quantity, optional
            Sampling period for rate estimation, by default 0.1*pq.ms
        sigma : pq.Quantity, optional
            Standard deviation of Gaussian kernel, by default 100*pq.ms
        trial_length : pq.Quantity, optional
            Duration of each trial, by default 500*pq.ms
        """
        self.max_refractory = max_refractory
        self.sampling_period = sampling_period
        self.sigma = sigma
        self.trial_length = trial_length
    
    def estimate_rate_and_deadtime(self, spiketrain: neo.SpikeTrain, 
                                   sep: pq.Quantity) -> Tuple[neo.AnalogSignal, 
                                                              pq.Quantity, 
                                                              List[neo.AnalogSignal]]:
        """
        Estimate firing rate, refractory period, and CV from a spike train.
        
        Parameters
        ----------
        spiketrain : neo.SpikeTrain
            Input spike train for analysis
        sep : pq.Quantity
            Buffer between trials in concatenated data
            
        Returns
        -------
        rate : neo.AnalogSignal
            Estimated firing rate
        refractory_period : pq.Quantity
            Estimated refractory period
        rate_list : List[neo.AnalogSignal]
            List of rate profiles for all trials
        """
        # Deconcatenate data into individual trials
        trial_list = self._create_trial_list(spiketrain, sep)
        
        # Estimate instantaneous firing rate for each trial
        rate_list = self._estimate_rates_per_trial(trial_list)
        
        # Concatenate rates back together
        rate = self._concatenate_rates(rate_list, sep)
        
        # Calculate refractory period
        refractory_period = self._estimate_deadtime(spiketrain)
        
        return rate, refractory_period, rate_list
    
    def _create_trial_list(self, spiketrain: neo.SpikeTrain, 
                          sep: pq.Quantity) -> List[neo.SpikeTrain]:
        """
        Split concatenated spike train into individual trials.
        
        Parameters
        ----------
        spiketrain : neo.SpikeTrain
            Concatenated spike train
        sep : pq.Quantity
            Buffer between trials
            
        Returns
        -------
        List[neo.SpikeTrain]
            List of individual trial spike trains
        """
        spiketrain_list = []
        t_max = spiketrain.t_stop
        trial = 0
        
        sep = sep.rescale(pq.s)
        epoch_length = self.trial_length.rescale(pq.s)
        
        while True:
            t_start = trial * (epoch_length + sep)
            t_stop = t_start + epoch_length
            
            if t_start >= t_max - sep:
                break
                
            # Extract spikes for this trial
            trial_mask = (spiketrain >= t_start) & (spiketrain < t_stop)
            cutted_st = spiketrain[trial_mask] - t_start
            cutted_st.t_start = 0.0 * pq.s
            cutted_st.t_stop = epoch_length
            
            spiketrain_list.append(cutted_st)
            trial += 1
            
        return spiketrain_list
    
    def _estimate_rates_per_trial(self, trial_list: List[neo.SpikeTrain]) -> List[neo.AnalogSignal]:
        """
        Estimate instantaneous firing rates for each trial.
        
        Parameters
        ----------
        trial_list : List[neo.SpikeTrain]
            List of trial spike trains
            
        Returns
        -------
        List[neo.AnalogSignal]
            List of rate estimates for each trial
        """
        rate_list = []
        
        for trial in trial_list:
            if len(trial) > 10:
                # Use automatic kernel selection for sufficient spikes
                rate = self._compute_instantaneous_rate(
                    trial, kernel='auto', border_correction=True
                )
            else:
                # Use Gaussian kernel for sparse spike trains
                gaussian_kernel = kernels.GaussianKernel(sigma=self.sigma)
                rate = self._compute_instantaneous_rate(
                    trial, kernel=gaussian_kernel, border_correction=True
                )
            rate_list.append(rate)
            
        return rate_list
    
    def _compute_instantaneous_rate(self, spiketrain: neo.SpikeTrain,
                                   kernel: str = 'auto',
                                   border_correction: bool = True) -> neo.AnalogSignal:
        """
        Compute instantaneous firing rate using kernel convolution.
        
        This is a simplified wrapper around the main instantaneous rate function
        with commonly used parameters.
        """
        return stat.instantaneous_rate(
            spiketrain,
            sampling_period=self.sampling_period,
            kernel=kernel,
            border_correction=border_correction
        )
    
    def _concatenate_rates(self, rate_list: List[neo.AnalogSignal], 
                          sep: pq.Quantity) -> neo.AnalogSignal:
        """
        Concatenate individual trial rates back into a single rate signal.
        
        Parameters
        ----------
        rate_list : List[neo.AnalogSignal]
            List of rate signals for each trial
        sep : pq.Quantity
            Buffer between trials
            
        Returns
        -------
        neo.AnalogSignal
            Concatenated rate signal
        """
        trial_length_bins = int(
            (self.trial_length / self.sampling_period).simplified.magnitude
        )
        sep_bins = int((sep / self.sampling_period).simplified.magnitude)
        total_bins = len(rate_list) * (trial_length_bins + sep_bins)
        
        rate_array = np.zeros(total_bins)
        
        for trial_id, rate_trial in enumerate(rate_list):
            start_idx = trial_id * (trial_length_bins + sep_bins)
            end_idx = start_idx + trial_length_bins
            rate_array[start_idx:end_idx] = rate_trial.flatten().magnitude
        
        return neo.AnalogSignal(
            signal=rate_array,
            units=rate_list[0].units,
            sampling_period=self.sampling_period
        )
    
    def _estimate_deadtime(self, spiketrain: neo.SpikeTrain) -> pq.Quantity:
        """
        Estimate the dead time (refractory period) of a spike train.
        
        Parameters
        ----------
        spiketrain : neo.SpikeTrain
            Input spike train
            
        Returns
        -------
        pq.Quantity
            Estimated dead time
        """
        if len(spiketrain) > 1:
            isi = np.diff(spiketrain.simplified.magnitude)
            dead_time = np.min(isi) * pq.s
            return min(dead_time, self.max_refractory)
        else:
            return self.max_refractory
    
    def calculate_cv2(self, spiketrain: neo.SpikeTrain, sep: pq.Quantity) -> float:
        """
        Calculate the CV2 (coefficient of variation of ISIs) for a spike train.
        
        Parameters
        ----------
        spiketrain : neo.SpikeTrain
            Input spike train
        sep : pq.Quantity
            Buffer between trials
            
        Returns
        -------
        float
            CV2 value, or NaN if insufficient spikes
        """
        if len(spiketrain) <= 5:
            return np.nan
            
        trial_list = self._create_trial_list(spiketrain, sep)
        isis_list = [
            np.diff(st.magnitude) for st in trial_list if len(st) > 2
        ]
        
        if not isis_list:
            return np.nan
            
        cv2_values = [stat.cv2(isi) for isi in isis_list]
        return np.nanmean(cv2_values)
    
    def calculate_cv_operational_time(self, spiketrain: neo.SpikeTrain,
                                     rate_list: List[neo.AnalogSignal],
                                     sep: pq.Quantity) -> float:
        """
        Calculate CV in operational time (time scaled by instantaneous rate).
        
        Parameters
        ----------
        spiketrain : neo.SpikeTrain
            Input spike train
        rate_list : List[neo.AnalogSignal]
            List of rate profiles for each trial
        sep : pq.Quantity
            Buffer between trials
            
        Returns
        -------
        float
            CV in operational time
        """
        trial_list = self._create_trial_list(spiketrain, sep)
        
        # Check if we have enough spikes
        total_spikes = sum(len(trial) for trial in trial_list)
        if total_spikes < len(trial_list):
            return 1.0
        
        isis_operational_time = []
        
        for rate, trial in zip(rate_list, trial_list):
            if len(trial) <= 1:
                continue
                
            # Convert to operational time
            operational_time = self._convert_to_operational_time(rate, trial)
            isis_operational_time.append(np.diff(operational_time))
        
        return self._calculate_cv_from_isis(isis_operational_time)
    
    def _convert_to_operational_time(self, rate: neo.AnalogSignal,
                                    trial: neo.SpikeTrain) -> np.ndarray:
        """
        Convert spike times to operational time based on instantaneous rate.
        
        Parameters
        ----------
        rate : neo.AnalogSignal
            Instantaneous firing rate
        trial : neo.SpikeTrain
            Spike train for one trial
            
        Returns
        -------
        np.ndarray
            Spike times in operational time
        """
        # Time points at which firing rates are given
        real_time = np.concatenate([
            rate.times.simplified.magnitude,
            [rate.t_stop.simplified.magnitude]
        ])
        
        # Find indices where spikes fall
        indices = np.searchsorted(real_time, trial.magnitude)
        
        # Calculate cumulative operational time
        rate_values = rate.magnitude.flatten()
        dt = rate.sampling_period.simplified.magnitude
        operational_time = np.concatenate([[0], np.cumsum(rate_values * dt)])
        
        # Get operational times for spike positions
        trial_operational_time = operational_time[indices - 1]
        
        # Interpolate within bins
        positions_in_bins = (
            (trial.magnitude - real_time[indices - 1]) / dt
        )
        
        bin_contributions = (
            (operational_time[indices] - operational_time[indices - 1]) 
            * positions_in_bins
        )
        
        return trial_operational_time + bin_contributions
    
    def _calculate_cv_from_isis(self, isis_operational_time: List[np.ndarray]) -> float:
        """
        Calculate CV from a list of ISI arrays.
        
        Parameters
        ----------
        isis_operational_time : List[np.ndarray]
            List of ISI arrays from different trials
            
        Returns
        -------
        float
            Coefficient of variation
        """
        if not isis_operational_time:
            return 1.0
            
        # Combine all ISIs
        all_isis = np.concatenate(isis_operational_time)
        
        if len(all_isis) < 4:
            return 1.0
            
        mean_isi = np.mean(all_isis)
        std_isi = np.std(all_isis)
        
        return std_isi / mean_isi if mean_isi > 0 else 1.0
    
    def generate_artificial_data(self, data: List[neo.SpikeTrain],
                               processes: List[str],
                               sep: pq.Quantity,
                               seed: int = None) -> Tuple[List[neo.SpikeTrain], 
                                                         List[neo.SpikeTrain], 
                                                         List[float]]:
        """
        Generate artificial spike trains based on experimental data statistics.
        
        Parameters
        ----------
        data : List[neo.SpikeTrain]
            List of experimental spike trains
        processes : List[str]
            List of processes to generate ('ppd' for Poisson with dead time,
            'gamma' for Gamma process)
        sep : pq.Quantity
            Buffer between trials
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        ppd_spiketrains : List[neo.SpikeTrain]
            List of Poisson spike trains with dead time
        gamma_spiketrains : List[neo.SpikeTrain]
            List of Gamma process spike trains
        cv2_list : List[float]
            List of CV2 values from original data
        """
        if seed is not None:
            np.random.seed(seed)
        
        ppd_spiketrains = []
        gamma_spiketrains = []
        cv2_list = []
        
        for spiketrain in tqdm(data, desc="Generating artificial data"):
            # Estimate statistics from experimental data
            rate, refractory_period, rate_list = self.estimate_rate_and_deadtime(
                spiketrain, sep
            )
            cv2 = self.calculate_cv2(spiketrain, sep)
            cv2_list.append(cv2)
            
            # Generate Poisson process with dead time
            if 'ppd' in processes:
                ppd_spiketrain = self._generate_poisson_with_deadtime(
                    rate, refractory_period, spiketrain
                )
                ppd_spiketrains.append(ppd_spiketrain)
            
            # Generate Gamma process
            if 'gamma' in processes:
                gamma_spiketrain = self._generate_gamma_process(
                    rate, spiketrain, rate_list, sep
                )
                gamma_spiketrains.append(gamma_spiketrain)
        
        return ppd_spiketrains, gamma_spiketrains, cv2_list
    
    def _generate_poisson_with_deadtime(self, rate: neo.AnalogSignal,
                                       refractory_period: pq.Quantity,
                                       original_spiketrain: neo.SpikeTrain) -> neo.SpikeTrain:
        """
        Generate a Poisson process with dead time.
        
        Parameters
        ----------
        rate : neo.AnalogSignal
            Instantaneous firing rate
        refractory_period : pq.Quantity
            Refractory period
        original_spiketrain : neo.SpikeTrain
            Original spike train (for copying annotations)
            
        Returns
        -------
        neo.SpikeTrain
            Generated spike train
        """
        ppd_spiketrain = stg.inhomogeneous_poisson_process(
            rate=rate,
            as_array=False,
            refractory_period=refractory_period
        )
        
        # Copy annotations from original
        ppd_spiketrain.annotate(**original_spiketrain.annotations)
        return ppd_spiketrain.rescale(pq.s)
    
    def _generate_gamma_process(self, rate: neo.AnalogSignal,
                               original_spiketrain: neo.SpikeTrain,
                               rate_list: List[neo.AnalogSignal],
                               sep: pq.Quantity) -> neo.SpikeTrain:
        """
        Generate a Gamma process based on operational time CV.
        
        Parameters
        ----------
        rate : neo.AnalogSignal
            Instantaneous firing rate
        original_spiketrain : neo.SpikeTrain
            Original spike train
        rate_list : List[neo.AnalogSignal]
            List of rate profiles for each trial
        sep : pq.Quantity
            Buffer between trials
            
        Returns
        -------
        neo.SpikeTrain
            Generated Gamma process spike train
        """
        cv = self.calculate_cv_operational_time(original_spiketrain, rate_list, sep)
        shape_factor = 1 / (cv ** 2) if cv > 0 else 1.0
        
        print(f'CV = {cv:.3f}, shape_factor = {shape_factor:.3f}')
        
        if np.any(rate.magnitude > 0):
            gamma_spiketrain = stg.inhomogeneous_gamma_process(
                rate=rate, 
                shape_factor=shape_factor
            )
        else:
            # Create empty spike train if no firing
            gamma_spiketrain = neo.SpikeTrain(
                [] * pq.s,
                t_start=original_spiketrain.t_start,
                t_stop=original_spiketrain.t_stop
            )
        
        gamma_spiketrain.annotate(**original_spiketrain.annotations)
        return gamma_spiketrain


# Utility functions for CV calculations
def cv2_from_shape_factor(shape_factor: float) -> float:
    """
    Calculate CV2 from shape factor using van Vreeswijk's formula.
    
    Parameters
    ----------
    shape_factor : float
        Shape factor of the Gamma distribution
        
    Returns
    -------
    float
        CV2 value
    """
    numerator = gamma_function(2 * shape_factor)
    denominator = (shape_factor * (2 ** (shape_factor - 1) * gamma_function(shape_factor))) ** 2
    return numerator / denominator


def shape_factor_from_cv2(cv2: float) -> float:
    """
    Calculate shape factor from CV2 using van Vreeswijk's formula.
    
    Parameters
    ----------
    cv2 : float
        CV2 value
        
    Returns
    -------
    float
        Shape factor
    """
    return brentq(lambda x: cv2_from_shape_factor(x) - cv2, 0.05, 50.0)


# Module-level functions for backward compatibility and external access
def estimate_rate_deadtime(spiketrain, max_refractory, sep, 
                          sampling_period=0.1 * pq.ms,
                          sigma=100 * pq.ms,
                          trial_length=500 * pq.ms):
    """
    Legacy function wrapper for backward compatibility.
    
    Function estimating rate, refractory period and cv, given one spiketrain
    
    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        spiketrain from which rate, refractory period and cv are estimated
    max_refractory : pq.Quantity
        maximal refractory period allowed
    sep : pq.Quantity
        buffer in between trials in the concatenated data (typically is equal
        to 2 * binsize * winlen)
    sampling_period : pq.Quantity
        sampling period of the firing rate (optional)
    sigma : pq.Quantity
        sd of the gaussian kernel for rate estimation (optional)
    trial_length : pq.Quantity
        duration of each trial
        
    Returns
    -------
    rate: neo.AnalogSignal
        rate of the spiketrain
    refractory_period: pq.Quantity
        refractory period of the spiketrain (minimal isi)
    rate_list: list
        list of rate profiles for all trials
    """
    generator = ArtificialDataGenerator(
        max_refractory=max_refractory,
        sampling_period=sampling_period,
        sigma=sigma,
        trial_length=trial_length
    )
    return generator.estimate_rate_and_deadtime(spiketrain, sep)


def create_st_list(spiketrain, sep, epoch_length=0.5*pq.s):
    """
    Legacy function wrapper for backward compatibility.
    
    The function generates a list of spiketrains from the concatenated data,
    where each list corresponds to a trial.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        spiketrains which are concatenated over trials for a
        certain epoch.
    sep: pq.Quantity
        buffer in between concatenated trials
    epoch_length: pq.Quantity
        length of each trial
        
    Returns
    -------
    spiketrain_list : list of neo.SpikeTrain
        List of spiketrains, where each spiketrain corresponds
        to one trial of a certain epoch.
    """
    generator = ArtificialDataGenerator(trial_length=epoch_length)
    return generator._create_trial_list(spiketrain, sep)


def estimate_deadtime(spiketrain, max_dead_time):
    """
    Legacy function wrapper for backward compatibility.
    
    Function to calculate the dead time of one spike train.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        spiketrain from which rate, refractory period and cv are estimated
    max_dead_time : pq.Quantity
        maximal refractory period allowed

    Returns
    -------
    dead_time: pq.Quantity
        refractory period of the spiketrain (minimal isi)
    """
    generator = ArtificialDataGenerator(max_refractory=max_dead_time)
    return generator._estimate_deadtime(spiketrain)


def get_cv2(spiketrain, sep):
    """
    Legacy function wrapper for backward compatibility.
    
    calculates the cv2 of a spiketrain

    Parameters
    ----------
    spiketrain: neo.SpikeTrain
        single neuron concatenate spike train
    sep: pq.Quantity

    Returns
    -------
    cv2 : float
        The CV2 or 1. if not enough spikes are in the spiketrain.
    """
    generator = ArtificialDataGenerator()
    return generator.calculate_cv2(spiketrain, sep)


# Additional utility functions that might be useful
def get_cv_operational_time(spiketrain, rate_list, sep):
    """
    Legacy function wrapper for backward compatibility.
    
    calculates cv of spike train in operational time

    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        Input spike train
    rate_list : List[neo.AnalogSignal]
        List of rate profiles for each trial
    sep : pq.Quantity
        Buffer between trials

    Returns
    -------
    cv : float
        CV in operational time
    """
    generator = ArtificialDataGenerator()
    return generator.calculate_cv_operational_time(spiketrain, rate_list, sep)


def generate_artificial_data(data, seed, max_refractory, processes, sep):
    """
    Legacy function wrapper for backward compatibility.
    
    Generate data as Poisson with refractory period and Gamma
    
    Parameters
    ----------
    data: list
        list of spiketrains
    seed: int
        seed for the data generation
    max_refractory: quantity
        maximal refractory period
    processes: list
        processes to be generated
    sep: pq.Quantity
        buffering between two trials

    Returns
    -------
    ppd_spiketrains: list
        list of poisson processes (neo.SpikeTrain) with rate profile and
        refractory period estimated from data
    gamma_spiketrains: list
        list of gamma processes (neo.SpikeTrain) with rate profile and
        shape (cv) estimated from data
    cv_list : list
        list of cvs, estimated from data, one for each neuron
    """
    generator = ArtificialDataGenerator(max_refractory=max_refractory)
    return generator.generate_artificial_data(data, processes, sep, seed)

def main_all_combinations():
    """
    Main function to run the artificial data generation pipeline.
    """
    import yaml
    from yaml import Loader
    
    # Load configuration
    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    
    # Extract configuration parameters
    sessions = config['sessions']
    epochs = config['epochs']
    trialtypes = config['trialtypes']
    winlen = config['winlen']
    unit = config['unit']
    binsize = (config['binsize'] * pq.s).rescale(unit)
    seed = config['seed']
    processes = config['processes']
    SNR_thresh = config['SNR_thresh']
    synchsize = config['synchsize']
    
    # Calculate derived parameters
    sep = 2 * winlen * binsize
    max_refractory = 4 * pq.ms
    
    # Initialize data generator
    generator = ArtificialDataGenerator(max_refractory=max_refractory)
    
    print("="*60)
    print("ARTIFICIAL DATA GENERATION")
    print("="*60)
    print(f"Sessions: {sessions}")
    print(f"Epochs: {epochs}")
    print(f"Trial types: {trialtypes}")
    print(f"Processes to generate: {processes}")
    print(f"Random seed: {seed}")
    print(f"Separation time: {sep}")
    print("="*60)
    
    # Process each session, epoch, and trial type
    for session in sessions:
        print(f"\n🎯 Processing session: {session}")
        
        # Create output directories
        output_dirs = {
            'ppd': Path(f'../data/artificial_data/ppd/{session}'),
            'gamma': Path(f'../data/artificial_data/gamma/{session}')
        }
        
        for process_type, output_dir in output_dirs.items():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"   📁 Created directory: {output_dir}")
        
        for epoch in epochs:
            for trialtype in trialtypes:
                print(f'\n📋 Processing: {session} | {epoch} | {trialtype}')
                
                try:
                    # Load pre-processed concatenated spike trains
                    data_path = (Path('../data/concatenated_spiketrains') / 
                               session / f'{epoch}_{trialtype}.npy')
                    
                    print(f'   📂 Loading from: {data_path}')
                    
                    if not data_path.exists():
                        print(f'   ❌ File not found: {data_path}')
                        print(f'      Skipping {session}/{epoch}/{trialtype}')
                        continue
                    
                    # Load the processed spike trains using our improved loader
                    spike_trains = load_processed_spike_trains(str(data_path))
                    
                    if not spike_trains:
                        print(f'   ⚠️ No spike trains loaded from {data_path}')
                        print(f'      Skipping {session}/{epoch}/{trialtype}')
                        continue
                    
                    print(f'   ✅ Loaded {len(spike_trains)} spike trains')
                    
                    # Log some statistics about the loaded data
                    spike_counts = [len(st) for st in spike_trains]
                    total_spikes = sum(spike_counts)
                    mean_spikes = np.mean(spike_counts) if spike_counts else 0
                    
                    print(f'      Total spikes: {total_spikes:,}')
                    print(f'      Mean spikes per unit: {mean_spikes:.1f}')
                    print(f'      Spike count range: {min(spike_counts)} - {max(spike_counts)}')
                    
                    # Generate artificial data
                    print("   🔬 Generating artificial data...")
                    ppd_trains, gamma_trains, cv2_values = generator.generate_artificial_data(
                        data=spike_trains, processes=processes, sep=sep, seed=seed
                    )
                    
                    print('   ✅ Data generation complete!')
                    
                    # Save generated data using the improved save function
                    print('   💾 Saving artificial data...')
                    
                    if 'ppd' in processes and ppd_trains:
                        ppd_path = output_dirs['ppd'] / f'ppd_{epoch}_{trialtype}.npy'
                        print(f'      📁 Saving PPD data to: {ppd_path}')
                        
                        if save_processed_data(ppd_trains, ppd_path):
                            print(f'      ✅ PPD data saved successfully')
                            print(f'         {len(ppd_trains)} spike trains processed')
                        else:
                            print(f'      ❌ Failed to save PPD data')
                    
                    if 'gamma' in processes and gamma_trains:
                        gamma_path = output_dirs['gamma'] / f'gamma_{epoch}_{trialtype}.npy'
                        cv2_path = output_dirs['gamma'] / f'cv2s_{epoch}_{trialtype}.npy'
                        
                        print(f'      📁 Saving Gamma data to: {gamma_path}')
                        
                        if save_processed_data(gamma_trains, gamma_path):
                            print(f'      ✅ Gamma spike trains saved successfully')
                            print(f'         {len(gamma_trains)} spike trains processed')
                        else:
                            print(f'      ❌ Failed to save Gamma spike trains')
                        
                        # Save CV2 values using numpy (these are just numbers, not spike trains)
                        try:
                            np.save(cv2_path, cv2_values, allow_pickle=True)
                            print(f'      ✅ CV2 values saved to: {cv2_path}')
                            print(f'         {len(cv2_values)} CV2 values saved')
                        except Exception as e:
                            print(f'      ❌ Failed to save CV2 values: {e}')
                    
                    print(f'   🎉 Completed: {session}/{epoch}/{trialtype}')
                    
                except Exception as e:
                    print(f'   ❌ Error processing {session}/{epoch}/{trialtype}: {e}')
                    import traceback
                    print(f'      Full traceback:')
                    traceback.print_exc()
                    continue
    
    print(f"\n{'='*60}")
    print("🎉 ARTIFICIAL DATA GENERATION COMPLETE!")
    print(f"{'='*60}")


def main():
    """
    Main function to run artificial data generation for a single combination.
    Expects command line arguments: session epoch trialtype
    """
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python generate_artificial_data_single.py <session> <epoch> <trialtype>")
        sys.exit(1)
    
    session = sys.argv[1]
    epoch = sys.argv[2]
    trialtype = sys.argv[3]
    
    # Load configuration
    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    
    # Extract configuration parameters
    epochs = config['epochs']
    trialtypes = config['trialtypes']
    winlen = config['winlen']
    unit = config['unit']
    binsize = (config['binsize'] * pq.s).rescale(unit)
    seed = config['seed']
    processes = config['processes']
    SNR_thresh = config['SNR_thresh']
    synchsize = config['synchsize']
    
    # Calculate derived parameters
    sep = 2 * winlen * binsize
    max_refractory = 4 * pq.ms
    
    # Initialize data generator
    generator = ArtificialDataGenerator(max_refractory=max_refractory)
    
    print("="*60)
    print("ARTIFICIAL DATA GENERATION - SINGLE JOB")
    print("="*60)
    print(f"Session: {session}")
    print(f"Epoch: {epoch}")
    print(f"Trial type: {trialtype}")
    print(f"Processes to generate: {processes}")
    print(f"Random seed: {seed}")
    print(f"Separation time: {sep}")
    print("="*60)
    
    # Create output directories
    output_dirs = {
        'ppd': Path(f'/users/bouss/spade/SPADE_surrogates/data/artificial_data/ppd/{session}'),
        'gamma': Path(f'/users/bouss/spade/SPADE_surrogates/data/artificial_data/gamma/{session}')
    }
    
    for process_type, output_dir in output_dirs.items():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {output_dir}")
    
    print(f'\n📋 Processing: {session} | {epoch} | {trialtype}')
    
    try:
        # Load pre-processed concatenated spike trains
        data_path = Path(f'/users/bouss/spade/SPADE_surrogates/data/concatenated_spiketrains/{session}/{epoch}_{trialtype}.npy')
        
        print(f'📂 Loading from: {data_path}')
        
        if not data_path.exists():
            print(f'❌ File not found: {data_path}')
            print(f'   Skipping {session}/{epoch}/{trialtype}')
            sys.exit(1)
        
        # Load the processed spike trains using our improved loader
        spike_trains = load_processed_spike_trains(str(data_path))
        
        if not spike_trains:
            print(f'⚠️ No spike trains loaded from {data_path}')
            print(f'   Skipping {session}/{epoch}/{trialtype}')
            sys.exit(1)
        
        print(f'✅ Loaded {len(spike_trains)} spike trains')
        
        # Log some statistics about the loaded data
        spike_counts = [len(st) for st in spike_trains]
        total_spikes = sum(spike_counts)
        mean_spikes = np.mean(spike_counts) if spike_counts else 0
        
        print(f'   Total spikes: {total_spikes:,}')
        print(f'   Mean spikes per unit: {mean_spikes:.1f}')
        print(f'   Spike count range: {min(spike_counts)} - {max(spike_counts)}')
        
        # Generate artificial data
        print("🔬 Generating artificial data...")
        ppd_trains, gamma_trains, cv2_values = generator.generate_artificial_data(
            data=spike_trains, processes=processes, sep=sep, seed=seed
        )
        
        print('✅ Data generation complete!')
        
        # Save generated data using the improved save function
        print('💾 Saving artificial data...')
        
        if 'ppd' in processes and ppd_trains:
            ppd_path = output_dirs['ppd'] / f'ppd_{epoch}_{trialtype}.npy'
            print(f'   📁 Saving PPD data to: {ppd_path}')
            
            if save_processed_data(ppd_trains, ppd_path):
                print(f'   ✅ PPD data saved successfully')
                print(f'      {len(ppd_trains)} spike trains processed')
            else:
                print(f'   ❌ Failed to save PPD data')
                sys.exit(1)
        
        if 'gamma' in processes and gamma_trains:
            gamma_path = output_dirs['gamma'] / f'gamma_{epoch}_{trialtype}.npy'
            cv2_path = output_dirs['gamma'] / f'cv2s_{epoch}_{trialtype}.npy'
            
            print(f'   📁 Saving Gamma data to: {gamma_path}')
            
            if save_processed_data(gamma_trains, gamma_path):
                print(f'   ✅ Gamma spike trains saved successfully')
                print(f'      {len(gamma_trains)} spike trains processed')
            else:
                print(f'   ❌ Failed to save Gamma spike trains')
                sys.exit(1)
            
            # Save CV2 values using numpy (these are just numbers, not spike trains)
            try:
                np.save(cv2_path, cv2_values, allow_pickle=True)
                print(f'   ✅ CV2 values saved to: {cv2_path}')
                print(f'      {len(cv2_values)} CV2 values saved')
            except Exception as e:
                print(f'   ❌ Failed to save CV2 values: {e}')
                sys.exit(1)
        
        print(f'🎉 Completed: {session}/{epoch}/{trialtype}')
        
    except Exception as e:
        print(f'❌ Error processing {session}/{epoch}/{trialtype}: {e}')
        import traceback
        print(f'   Full traceback:')
        traceback.print_exc()
        sys.exit(1)

    
if __name__ == '__main__':
    main()