"""
Script for creating artificial data that mimics experimental data features.

This script generates artificial spike trains using Poisson processes with refractory 
periods and Gamma processes, based on statistics estimated from real experimental data.
"""

from pathlib import Path
from typing import List, Tuple

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


def main():
    """
    Main function to run the artificial data generation pipeline.
    """
    import rgutils
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
    load_original_data = False
    
    # Initialize data generator
    generator = ArtificialDataGenerator(max_refractory=max_refractory)
    
    # Process each session, epoch, and trial type
    for session in sessions:
        # Create output directories
        output_dirs = {
            'ppd': Path(f'../data/artificial_data/ppd/{session}'),
            'gamma': Path(f'../data/artificial_data/gamma/{session}')
        }
        
        for process_type, output_dir in output_dirs.items():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in epochs:
            for trialtype in trialtypes:
                print(f'Processing: {session} {epoch} {trialtype}')
                
                # Load data
                if load_original_data:
                    print('Loading experimental data...')
                    sts = rgutils.load_epoch_concatenated_trials(
                        session, epoch, trialtypes=trialtype,
                        SNRthresh=SNR_thresh, synchsize=synchsize, sep=sep
                    )
                else:
                    print('Loading pre-concatenated spike trains...')
                    data_path = (Path('../data/concatenated_spiketrains') / 
                               session / f'{epoch}_{trialtype}.npy')
                    sts = np.load(data_path, allow_pickle=True)
                
                # Generate artificial data
                print("Generating artificial data...")
                ppd_trains, gamma_trains, cv2_values = generator.generate_artificial_data(
                    data=sts, processes=processes, sep=sep, seed=seed
                )
                
                print('Data generation complete!')
                
                # Save generated data
                print('Saving data...')
                if 'ppd' in processes:
                    ppd_path = output_dirs['ppd'] / f'ppd_{epoch}_{trialtype}.npy'
                    np.save(ppd_path, ppd_trains)
                
                if 'gamma' in processes:
                    gamma_path = output_dirs['gamma'] / f'gamma_{epoch}_{trialtype}.npy'
                    cv2_path = output_dirs['gamma'] / f'cv2s_{epoch}_{trialtype}.npy'
                    np.save(gamma_path, gamma_trains)
                    np.save(cv2_path, cv2_values)
                
                print(f'Completed: {session} {epoch} {trialtype}')


if __name__ == '__main__':
    main()