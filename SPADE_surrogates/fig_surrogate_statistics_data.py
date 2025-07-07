"""
Script for generating data for surrogate method statistical analysis.

This script generates comprehensive datasets comparing original spike train data
with various surrogate methods, including ISI distributions, correlations,
firing rate profiles, and coefficient of variation analysis.

The generated data is used by the plotting script to create statistical
comparison visualizations.
"""

import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import warnings

import numpy as np
import quantities as pq
import neo

from elephant import conversion as conv
from elephant import spike_train_correlation as corr
from elephant import statistics as stat
from elephant import spike_train_generation as stg
from elephant import spike_train_surrogates as surr

import fig_surrogate_statistics_config as cf


class SpikeTrainGenerator:
    """
    Class for generating different types of spike trains for statistical analysis.
    
    This class provides methods to create Poisson, Poisson with dead time (PPD),
    and Gamma process spike trains, both stationary and non-stationary.
    """
    
    def __init__(self):
        """Initialize the spike train generator with configuration parameters."""
        self.dead_time = cf.DEAD_TIME
        self.shape_factor = cf.SHAPE_FACTOR
    
    def create_stationary_spiketrain(self, data_type: str, rate: pq.Quantity,
                                   t_start: pq.Quantity, t_stop: pq.Quantity) -> neo.SpikeTrain:
        """
        Create a stationary spike train of specified type.
        
        Parameters
        ----------
        data_type : str
            Type of spike train ('Poisson', 'PPD', or 'Gamma')
        rate : pq.Quantity
            Firing rate
        t_start : pq.Quantity
            Start time
        t_stop : pq.Quantity
            Stop time
            
        Returns
        -------
        neo.SpikeTrain
            Generated spike train
            
        Raises
        ------
        ValueError
            If data_type is not recognized
        """
        if data_type == 'Poisson':
            return stg.homogeneous_poisson_process(
                rate=rate, t_start=t_start, t_stop=t_stop
            )
        elif data_type == 'PPD':
            return stg.homogeneous_poisson_process(
                rate=rate, t_start=t_start, t_stop=t_stop,
                refractory_period=self.dead_time
            )
        elif data_type == 'Gamma':
            return stg.homogeneous_gamma_process(
                a=self.shape_factor, 
                b=self.shape_factor * rate,
                t_start=t_start, t_stop=t_stop
            )
        else:
            raise ValueError(f"Unknown data_type '{data_type}'. "
                           "Must be one of: 'Poisson', 'PPD', 'Gamma'")
    
    def create_nonstationary_spiketrain(self, data_type: str, base_rate: pq.Quantity,
                                      t_start: pq.Quantity, t_stop: pq.Quantity) -> neo.SpikeTrain:
        """
        Create a non-stationary spike train with sinusoidal rate modulation.
        
        Parameters
        ----------
        data_type : str
            Type of spike train ('Poisson', 'PPD', or 'Gamma')
        base_rate : pq.Quantity
            Base firing rate
        t_start : pq.Quantity
            Start time
        t_stop : pq.Quantity
            Stop time
            
        Returns
        -------
        neo.SpikeTrain
            Generated non-stationary spike train
        """
        sampling_period = 1.0 * pq.ms
        wavelength = 100.0 * pq.ms
        
        # Create time vector
        time_points = np.arange(
            0.0, t_stop.rescale(pq.ms).magnitude, sampling_period.magnitude
        )
        
        # Create sinusoidal rate modulation
        rate_signal = (base_rate.magnitude + 
                      base_rate.magnitude * np.sin(2.0 * np.pi * time_points / wavelength.magnitude))
        
        rate = neo.AnalogSignal(
            signal=rate_signal * pq.Hz,
            t_start=t_start, t_stop=t_stop,
            sampling_period=sampling_period
        )
        
        if data_type == 'Poisson':
            return stg.inhomogeneous_poisson_process(rate=rate)
        elif data_type == 'PPD':
            return stg.inhomogeneous_poisson_process(
                rate=rate, refractory_period=self.dead_time
            )
        elif data_type == 'Gamma':
            return stg.inhomogeneous_gamma_process(
                rate=rate, shape_factor=self.shape_factor
            )
        else:
            raise ValueError(f"Unknown data_type '{data_type}'")


class SurrogateMethodApplier:
    """
    Class for applying various surrogate methods to spike trains.
    
    This class provides a unified interface for applying different surrogate
    methods including dithering, ISI methods, and shuffling approaches.
    """
    
    def __init__(self):
        """Initialize with configuration parameters."""
        self.dither = cf.DITHER
        self.trial_length = cf.TRIAL_LENGTH
        self.trial_separation = cf.TRIAL_SEPARATION
        self.spade_bin_size = cf.SPADE_BIN_SIZE
    
    def apply_surrogate_method(self, spiketrains: List[neo.SpikeTrain], 
                             method: str) -> List[neo.SpikeTrain]:
        """
        Apply a surrogate method to a list of spike trains.
        
        Parameters
        ----------
        spiketrains : List[neo.SpikeTrain]
            Input spike trains
        method : str
            Surrogate method to apply
            
        Returns
        -------
        List[neo.SpikeTrain]
            Surrogate spike trains
            
        Raises
        ------
        ValueError
            If method is not recognized
        """
        method_map = {
            'UD': self._uniform_dithering,
            'UDD': self._uniform_dithering_with_deadtime,
            'JISI-D': self._joint_isi_dithering,
            'ISI-D': self._isi_dithering,
            'SHIFT-ST': self._trial_shifting,
            'BIN-SHUFF': self._bin_shuffling
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown surrogate method '{method}'. "
                           f"Available methods: {list(method_map.keys())}")
        
        return method_map[method](spiketrains)
    
    def _uniform_dithering(self, spiketrains: List[neo.SpikeTrain]) -> List[neo.SpikeTrain]:
        """Apply uniform dithering to spike trains."""
        return [surr.dither_spikes(spiketrain=st, dither=self.dither)[0] 
                for st in spiketrains]
    
    def _uniform_dithering_with_deadtime(self, spiketrains: List[neo.SpikeTrain]) -> List[neo.SpikeTrain]:
        """Apply uniform dithering with dead time to spike trains."""
        return [surr.dither_spikes(
            spiketrain=st, dither=self.dither, refractory_period=10 * pq.ms
        )[0] for st in spiketrains]
    
    def _joint_isi_dithering(self, spiketrains: List[neo.SpikeTrain]) -> List[neo.SpikeTrain]:
        """Apply joint ISI dithering to spike trains."""
        return [surr.JointISI(
            spiketrain=st, dither=self.dither, method='window',
            cutoff=False, refractory_period=0.0, sigma=0.0
        ).dithering()[0] for st in spiketrains]
    
    def _isi_dithering(self, spiketrains: List[neo.SpikeTrain]) -> List[neo.SpikeTrain]:
        """Apply ISI dithering to spike trains."""
        return [surr.JointISI(
            spiketrain=st, dither=self.dither, method='window',
            isi_dithering=True, cutoff=False, refractory_period=0.0, sigma=0.0
        ).dithering()[0] for st in spiketrains]
    
    def _trial_shifting(self, spiketrains: List[neo.SpikeTrain]) -> List[neo.SpikeTrain]:
        """Apply trial shifting to spike trains."""
        return [surr.surrogates(
            method='trial_shifting', spiketrain=st, dt=self.dither,
            trial_length=self.trial_length, trial_separation=self.trial_separation
        )[0] for st in spiketrains]
    
    def _bin_shuffling(self, spiketrains: List[neo.SpikeTrain]) -> List[neo.SpikeTrain]:
        """Apply bin shuffling to spike trains."""
        return [surr.surrogates(
            method='bin_shuffling', spiketrain=st, dt=self.dither,
            bin_size=self.spade_bin_size
        )[0] for st in spiketrains]


class StatisticalAnalyzer:
    """
    Class for computing various statistical measures on spike trains.
    
    This class provides methods for calculating ISI distributions, correlations,
    displacement distributions, and other statistical measures.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with output path.
        
        Parameters
        ----------
        data_path : str
            Directory to save analysis results
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration parameters
        self.bin_size = cf.BIN_SIZE
        self.num_bins = cf.NUM_BINS
        self.isi_limit = cf.ISI_LIM
        self.firing_rate = cf.FIRING_RATE
    
    def compute_and_save_isi_distribution(self, spiketrain: neo.SpikeTrain,
                                        data_type: str, method: str) -> None:
        """
        Compute and save Inter-Spike Interval distribution.
        
        Parameters
        ----------
        spiketrain : neo.SpikeTrain
            Input spike train
        data_type : str
            Type of data (e.g., 'Poisson', 'PPD', 'Gamma')
        method : str
            Method used to generate the data (e.g., 'original', 'UD')
        """
        try:
            # Calculate ISIs
            isi = np.diff(spiketrain.rescale(pq.ms).magnitude)
            
            if len(isi) == 0:
                warnings.warn(f"No ISIs found for {data_type}_{method}")
                return
            
            # Create histogram
            max_isi = self.isi_limit * (1.0 / self.firing_rate).rescale(pq.ms).magnitude
            bin_edges = np.arange(0.0, max_isi, self.bin_size.rescale(pq.ms).magnitude)
            
            hist, bin_edges = np.histogram(isi, bins=bin_edges)
            
            # Normalize to probability density (1/s)
            hist = hist * 1000.0 / len(isi)
            
            # Save results
            results = {'hist': hist, 'bin_edges': bin_edges}
            output_file = self.data_path / f'isi_{data_type}_{method}.npy'
            np.save(output_file, results)
            
        except Exception as e:
            warnings.warn(f"Failed to compute ISI for {data_type}_{method}: {e}")
    
    def compute_and_save_displacement_distribution(self, original_spiketrain: neo.SpikeTrain,
                                                 surrogate_spiketrain: neo.SpikeTrain,
                                                 data_type: str, method: str) -> None:
        """
        Compute and save spike displacement distribution.
        
        Parameters
        ----------
        original_spiketrain : neo.SpikeTrain
            Original spike train
        surrogate_spiketrain : neo.SpikeTrain
            Surrogate spike train
        data_type : str
            Type of data
        method : str
            Surrogate method used
        """
        try:
            original_times = original_spiketrain.rescale(pq.ms).magnitude
            surrogate_times = np.sort(surrogate_spiketrain.rescale(pq.ms).magnitude)
            
            # Calculate displacements
            if len(original_times) == len(surrogate_times):
                displacement = surrogate_times - original_times
            else:
                min_length = min(len(original_times), len(surrogate_times))
                displacement = surrogate_times[:min_length] - original_times[:min_length]
            
            if len(displacement) == 0:
                warnings.warn(f"No displacements calculated for {data_type}_{method}")
                return
            
            # Create histogram
            dither_magnitude = cf.DITHER.rescale(pq.ms).magnitude
            bin_edges = np.arange(
                -dither_magnitude, dither_magnitude,
                self.bin_size.rescale(pq.ms).magnitude
            )
            
            hist, bin_edges = np.histogram(displacement, bins=bin_edges)
            
            # Normalize
            hist = hist * 1000.0 / len(displacement)
            
            # Save results
            results = {'hist': hist, 'bin_edges': bin_edges}
            output_file = self.data_path / f'displacement_{data_type}_{method}.npy'
            np.save(output_file, results)
            
        except Exception as e:
            warnings.warn(f"Failed to compute displacement for {data_type}_{method}: {e}")
    
    def compute_and_save_correlations(self, spiketrain1: neo.SpikeTrain,
                                    spiketrain2: neo.SpikeTrain,
                                    t_stop: pq.Quantity,
                                    data_type: str, method: str) -> None:
        """
        Compute and save auto-correlation and cross-correlation.
        
        Parameters
        ----------
        spiketrain1 : neo.SpikeTrain
            First spike train
        spiketrain2 : neo.SpikeTrain
            Second spike train
        t_stop : pq.Quantity
            Duration of recording
        data_type : str
            Type of data
        method : str
            Method used
        """
        try:
            # Create binned spike trains
            binned_st1 = conv.BinnedSpikeTrain(spiketrain1, bin_size=self.bin_size)
            binned_st2 = conv.BinnedSpikeTrain(spiketrain2, bin_size=self.bin_size)
            
            # Compute auto-correlation
            ac_hist = corr.cross_correlation_histogram(
                binned_spiketrain_i=binned_st1,
                binned_spiketrain_j=binned_st1,
                window=[-self.num_bins, self.num_bins],
                border_correction=False,
                binary=False,
                kernel=None,
                method='speed'
            )[0]
            
            # Process auto-correlation results
            ac_times = (ac_hist.times.rescale(pq.ms).magnitude + 
                       ac_hist.sampling_period.rescale(pq.ms).magnitude / 2)
            ac_values = (ac_hist[:, 0].magnitude / 
                        (self.firing_rate * self.bin_size * t_stop).simplified.magnitude)
            
            # Save auto-correlation
            ac_results = {'hist_times': ac_times, 'hist': ac_values}
            ac_file = self.data_path / f'ac_{data_type}_{method}.npy'
            np.save(ac_file, ac_results)
            
            # Compute cross-correlation
            cc_hist = corr.cross_correlation_histogram(
                binned_spiketrain_i=binned_st1,
                binned_spiketrain_j=binned_st2,
                window=[-self.num_bins, self.num_bins],
                border_correction=False,
                binary=False,
                kernel=None,
                method='speed'
            )[0]
            
            # Process cross-correlation results
            cc_times = (cc_hist.times.rescale(pq.ms).magnitude + 
                       cc_hist.sampling_period.rescale(pq.ms).magnitude / 2)
            cc_values = (cc_hist[:, 0].magnitude / 
                        (self.firing_rate * self.bin_size * t_stop).simplified.magnitude)
            
            # Save cross-correlation
            cc_results = {'hist_times': cc_times, 'hist': cc_values}
            cc_file = self.data_path / f'cc_{data_type}_{method}.npy'
            np.save(cc_file, cc_results)
            
        except Exception as e:
            warnings.warn(f"Failed to compute correlations for {data_type}_{method}: {e}")


class ComprehensiveDataGenerator:
    """
    Main class orchestrating the generation of all statistical analysis data.
    
    This class coordinates spike train generation, surrogate method application,
    and statistical analysis to create comprehensive datasets for visualization.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the comprehensive data generator.
        
        Parameters
        ----------
        data_path : str, optional
            Output directory path, defaults to config value
        """
        self.data_path = data_path or cf.DATA_PATH
        
        # Initialize component classes
        self.spike_generator = SpikeTrainGenerator()
        self.surrogate_applier = SurrogateMethodApplier()
        self.analyzer = StatisticalAnalyzer(self.data_path)
        
        # Configuration parameters
        self.data_types = cf.DATA_TYPES
        self.surr_methods = cf.SURR_METHODS
        self.firing_rate = cf.FIRING_RATE
        self.rates = cf.RATES
        self.cvs = cf.CVS
        self.high_spike_count = cf.HIGH_NUMBER_SPIKES
        self.low_spike_count = cf.LOW_NUMBER_SPIKES
        self.step_data_type = cf.STEP_DATA_TYPE
        self.firing_rates_step = cf.FIRING_RATES_STEP
        self.duration_rates_step = cf.DURATION_RATES_STEP
        self.number_spiketrains = cf.NUMBER_SPIKETRAINS
    
    def generate_single_rate_statistical_analysis(self) -> None:
        """
        Generate ISI distributions, correlations, and displacements for single firing rate.
        
        This method creates comprehensive statistical analysis data comparing
        original spike trains with their surrogates at a fixed firing rate.
        """
        print("Generating single rate statistical analysis...")
        
        t_start = 0.0 * pq.s
        t_stop = (self.high_spike_count / self.firing_rate).rescale(pq.s)
        
        for data_type in self.data_types:
            print(f"Processing {data_type}...")
            
            # Set random seeds for reproducibility
            np.random.seed(1)
            random.seed(1)
            
            start_time = time.time()
            
            # Generate original spike trains
            spiketrain = self.spike_generator.create_stationary_spiketrain(
                data_type, self.firing_rate, t_start, t_stop
            )
            spiketrain2 = self.spike_generator.create_stationary_spiketrain(
                data_type, self.firing_rate, t_start, t_stop
            )
            
            # Analyze original data
            self.analyzer.compute_and_save_isi_distribution(spiketrain, data_type, 'original')
            self.analyzer.compute_and_save_correlations(
                spiketrain, spiketrain, t_stop, data_type, 'original'
            )
            
            print(f"Original {data_type}: {time.time() - start_time:.2f}s")
            
            # Process surrogate methods
            for surr_method in self.surr_methods:
                start_time = time.time()
                
                # Set random seeds
                np.random.seed(0)
                random.seed(0)
                
                # Generate surrogates
                surrogate_trains = self.surrogate_applier.apply_surrogate_method(
                    [spiketrain, spiketrain2], surr_method
                )
                dithered_spiketrain, dithered_spiketrain2 = surrogate_trains
                
                # Analyze surrogate data
                self.analyzer.compute_and_save_isi_distribution(
                    dithered_spiketrain, data_type, surr_method
                )
                self.analyzer.compute_and_save_correlations(
                    dithered_spiketrain, spiketrain, t_stop, data_type, surr_method
                )
                self.analyzer.compute_and_save_displacement_distribution(
                    spiketrain, dithered_spiketrain, data_type, surr_method
                )
                
                print(f"{surr_method} {data_type}: {time.time() - start_time:.2f}s")
    
    def _analyze_single_rate_clipping_and_movement(self, rate: pq.Quantity) -> Tuple[Dict, Dict]:
        """
        Analyze clipping and spike movement for a single firing rate.
        
        Parameters
        ----------
        rate : pq.Quantity
            Firing rate to analyze
            
        Returns
        -------
        Tuple[Dict, Dict]
            Clipped rates and movement efficiency dictionaries
        """
        clipped_rates = {}
        movement_efficiency = {}
        
        t_start = 0.0 * pq.s
        t_stop = (self.low_spike_count / rate).rescale(pq.s)
        
        for data_type in self.data_types:
            # Set random seeds
            np.random.seed(1)
            random.seed(1)
            
            # Generate spike trains
            spiketrain = self.spike_generator.create_stationary_spiketrain(
                data_type, rate, t_start, t_stop
            )
            spiketrain2 = self.spike_generator.create_stationary_spiketrain(
                data_type, rate, t_start, t_stop
            )
            
            # Analyze original data
            clipped_rates[data_type] = {'rate': (len(spiketrain) / t_stop).rescale(pq.Hz)}
            
            # Convert to binned representations
            binned_bool = conv.BinnedSpikeTrain(
                spiketrain, bin_size=cf.SPADE_BIN_SIZE
            ).to_bool_array()
            binned_array = conv.BinnedSpikeTrain(
                spiketrain, bin_size=cf.SPADE_BIN_SIZE
            ).to_array()
            binned_array2 = conv.BinnedSpikeTrain(
                spiketrain2, bin_size=cf.SPADE_BIN_SIZE
            ).to_array()
            
            # Calculate clipping ratio
            clipped_rates[data_type]['binned'] = np.sum(binned_bool) / len(spiketrain)
            
            # Calculate independent movement baseline
            movement_efficiency[data_type] = {}
            movement_efficiency[data_type]['indep.'] = (
                np.sum(np.abs(binned_array - binned_array2)) / 2 /
                np.sqrt(len(spiketrain) * len(spiketrain2))
            )
            
            # Analyze surrogate methods
            for surr_method in self.surr_methods:
                # Set random seeds
                np.random.seed(0)
                random.seed(0)
                
                # Generate surrogate
                surrogate_train = self.surrogate_applier.apply_surrogate_method(
                    [spiketrain], surr_method
                )[0]
                
                # Analyze surrogate
                surrogate_binned_bool = conv.BinnedSpikeTrain(
                    surrogate_train, bin_size=cf.SPADE_BIN_SIZE
                ).to_bool_array()
                
                clipped_rates[data_type][surr_method] = (
                    np.sum(surrogate_binned_bool) / len(spiketrain)
                )
                
                movement_efficiency[data_type][surr_method] = (
                    np.sum(np.abs(binned_array - surrogate_binned_bool)) / 2 /
                    np.sqrt(len(spiketrain) * len(surrogate_train))
                )
        
        return clipped_rates, movement_efficiency
    
    def generate_clipping_and_movement_analysis(self) -> None:
        """
        Generate analysis of spike clipping and movement across firing rates.
        
        This method analyzes how surrogate methods affect spike train statistics
        as a function of firing rate, including clipping effects and spike movement.
        """
        print("Generating clipping and movement analysis...")
        
        # Initialize storage
        rates_dict = defaultdict(list)
        ratio_clipped = defaultdict(list)
        ratio_clipped_surr = defaultdict(lambda: defaultdict(list))
        ratio_indep_moved = defaultdict(list)
        ratio_moved = defaultdict(lambda: defaultdict(list))
        
        # Analyze each firing rate
        for rate in self.rates:
            print(f"Processing rate: {rate}")
            
            clipped_rates, movement_eff = self._analyze_single_rate_clipping_and_movement(rate)
            
            # Store results
            for data_type in self.data_types:
                rates_dict[data_type].append(clipped_rates[data_type]['rate'])
                ratio_clipped[data_type].append(clipped_rates[data_type]['binned'])
                ratio_indep_moved[data_type].append(movement_eff[data_type]['indep.'])
                
                for surr_method in self.surr_methods:
                    ratio_clipped_surr[data_type][surr_method].append(
                        clipped_rates[data_type][surr_method]
                    )
                    ratio_moved[data_type][surr_method].append(
                        movement_eff[data_type][surr_method]
                    )
        
        # Convert nested defaultdicts to regular dicts for saving
        def convert_nested_defaultdict(dd):
            return {k: dict(v) if isinstance(v, defaultdict) else v for k, v in dd.items()}
        
        # Save results
        results = {
            'rates': dict(rates_dict),
            'ratio_clipped': dict(ratio_clipped),
            'ratio_clipped_surr': convert_nested_defaultdict(ratio_clipped_surr),
            'ratio_indep_moved': dict(ratio_indep_moved),
            'ratio_moved': convert_nested_defaultdict(ratio_moved)
        }
        
        output_file = Path(self.data_path) / 'clipped_rates.npy'
        np.save(output_file, results)
        print(f"Saved clipping analysis to {output_file}")
    
    def generate_firing_rate_change_analysis(self) -> None:
        """
        Generate analysis of firing rate profile changes after surrogate application.
        
        This method analyzes how surrogate methods affect step changes in firing rate,
        similar to the analysis in Louis et al. (2010).
        """
        print("Generating firing rate change analysis...")
        
        rate_1, rate_2 = self.firing_rates_step
        t_start = 0.0 * pq.ms
        t_stop = self.duration_rates_step
        units = t_stop.units
        t_stop_magnitude = t_stop.magnitude
        dither_magnitude = cf.DITHER.magnitude
        
        results = {}
        
        # Generate original data
        start_time = time.time()
        np.random.seed(1)
        
        spiketrains = []
        for _ in range(self.number_spiketrains):
            # Generate two segments with different rates
            if self.step_data_type == 'Poisson':
                st1 = stg.homogeneous_poisson_process(
                    rate=rate_1, t_start=t_start - cf.DITHER, t_stop=t_stop + cf.DITHER
                )
                st2 = stg.homogeneous_poisson_process(
                    rate=rate_2, t_start=t_start - cf.DITHER, t_stop=t_stop + cf.DITHER
                )
            elif self.step_data_type == 'PPD':
                st1 = stg.homogeneous_poisson_process(
                    rate=rate_1, t_start=t_start - cf.DITHER, t_stop=t_stop + cf.DITHER,
                    refractory_period=cf.DEAD_TIME
                )
                st2 = stg.homogeneous_poisson_process(
                    rate=rate_2, t_start=t_start - cf.DITHER, t_stop=t_stop + cf.DITHER,
                    refractory_period=cf.DEAD_TIME
                )
            elif self.step_data_type == 'Gamma':
                st1 = stg.homogeneous_gamma_process(
                    a=cf.SHAPE_FACTOR, b=cf.SHAPE_FACTOR * rate_1,
                    t_start=t_start - cf.DITHER, t_stop=t_stop + cf.DITHER
                )
                st2 = stg.homogeneous_gamma_process(
                    a=cf.SHAPE_FACTOR, b=cf.SHAPE_FACTOR * rate_2,
                    t_start=t_start - cf.DITHER, t_stop=t_stop + cf.DITHER
                )
            else:
                raise ValueError(f"Unknown step data type: {self.step_data_type}")
            
            # Combine segments to create step function
            st1_times = st1.magnitude
            st2_times = st2.magnitude
            
            combined_times = np.concatenate([
                st1_times[st1_times <= t_stop_magnitude/2] - dither_magnitude,
                st2_times[st2_times > t_stop_magnitude/2]
            ])
            
            spiketrain = neo.SpikeTrain(
                times=combined_times,
                t_start=t_start - cf.DITHER,
                t_stop=t_stop + cf.DITHER,
                units=units
            )
            spiketrains.append(spiketrain)
        
        # Calculate original rate profile
        rate_original = stat.time_histogram(
            spiketrains, bin_size=cf.BIN_SIZE, t_start=t_start, t_stop=t_stop,
            output='rate'
        )
        results['original'] = rate_original
        print(f"Original processing: {time.time() - start_time:.2f}s")
        
        # Process surrogate methods
        for surr_method in self.surr_methods:
            start_time = time.time()
            np.random.seed(0)
            
            # Apply surrogate method with special handling for concatenated data
            if surr_method in ('UD', 'UDD'):
                if surr_method == 'UD':
                    dithered_spiketrains = [
                        surr.dither_spikes(spiketrain=st, edges=False, dither=cf.DITHER)[0]
                        for st in spiketrains
                    ]
                else:  # UDD
                    dithered_spiketrains = [
                        surr.dither_spikes(
                            spiketrain=st, dither=cf.DITHER, refractory_period=10 * pq.ms
                        )[0] for st in spiketrains
                    ]
            
            elif surr_method in ('ISI-D', 'JISI-D'):
                # Special handling for ISI methods with concatenated data
                concatenated_spiketrain = neo.SpikeTrain(
                    times=np.concatenate([
                        st.magnitude + dither_magnitude + 
                        st_id * (t_stop_magnitude + 2 * dither_magnitude)
                        for st_id, st in enumerate(spiketrains)
                    ]),
                    t_start=t_start,
                    t_stop=len(spiketrains) * (t_stop + 2 * cf.DITHER),
                    units=units
                )
                
                if surr_method == 'JISI-D':
                    dithered_concatenated = surr.JointISI(
                        spiketrain=concatenated_spiketrain, dither=cf.DITHER,
                        method='window', cutoff=False, refractory_period=0.0, sigma=0.0
                    ).dithering()[0]
                else:  # ISI-D
                    dithered_concatenated = surr.JointISI(
                        spiketrain=concatenated_spiketrain, dither=cf.DITHER,
                        method='window', isi_dithering=True, cutoff=False,
                        refractory_period=0.0, sigma=0.0
                    ).dithering()[0]
                
                dithered_times = dithered_concatenated.magnitude
                
                # Extract individual spike trains
                dithered_spiketrains = []
                for st_id in range(len(spiketrains)):
                    segment_start = st_id * (t_stop_magnitude + 2 * dither_magnitude)
                    segment_end = (st_id + 1) * (t_stop_magnitude + 2 * dither_magnitude)
                    
                    segment_mask = (dithered_times >= segment_start) & (dithered_times < segment_end)
                    segment_times = (dithered_times[segment_mask] - 
                                   segment_start - dither_magnitude)
                    
                    dithered_st = neo.SpikeTrain(
                        times=segment_times,
                        t_start=-cf.DITHER,
                        t_stop=t_stop + cf.DITHER,
                        units=units
                    )
                    dithered_spiketrains.append(dithered_st)
            
            elif surr_method == 'SHIFT-ST':
                trial_length = spiketrains[0].t_stop - spiketrains[0].t_start
                dithered_spiketrains = [
                    surr.surrogates(
                        method='trial_shifting', spiketrain=st, dt=cf.DITHER,
                        trial_length=trial_length, trial_separation=cf.TRIAL_SEPARATION
                    )[0] for st in spiketrains
                ]
            
            elif surr_method == 'BIN-SHUFF':
                dithered_spiketrains = []
                for st in spiketrains:
                    # Extract spikes within the analysis window
                    st_times = st.magnitude
                    valid_times = st_times[(st_times >= 0.0) & (st_times < t_stop_magnitude)]
                    
                    if len(valid_times) > 0:
                        trimmed_st = neo.SpikeTrain(
                            valid_times, t_start=t_start, t_stop=t_stop, units=units
                        )
                        dithered_st = surr.surrogates(
                            method='bin_shuffling', spiketrain=trimmed_st,
                            dt=cf.DITHER, bin_size=cf.SPADE_BIN_SIZE
                        )[0]
                    else:
                        dithered_st = neo.SpikeTrain(
                            [], t_start=t_start, t_stop=t_stop, units=units
                        )
                    
                    dithered_spiketrains.append(dithered_st)
            
            else:
                raise ValueError(f"Unknown surrogate method: {surr_method}")
            
            # Calculate surrogate rate profile
            rate_dithered = stat.time_histogram(
                dithered_spiketrains, bin_size=cf.BIN_SIZE,
                t_start=t_start, t_stop=t_stop, output='rate'
            )
            
            results[surr_method] = rate_dithered
            print(f"{surr_method} processing: {time.time() - start_time:.2f}s")
        
        # Save results
        output_file = Path(self.data_path) / 'rate_step.npy'
        np.save(output_file, results)
        print(f"Saved rate change analysis to {output_file}")
    
    def generate_cv_change_analysis(self) -> None:
        """
        Generate analysis of coefficient of variation changes in surrogate data.
        
        This method analyzes how surrogate methods affect the coefficient of
        variation of inter-spike intervals across different CV values.
        """
        print("Generating CV change analysis...")
        
        results = {}
        t_start = 0.0 * pq.s
        t_stop = (self.low_spike_count / self.firing_rate).rescale(pq.s)
        
        # Generate spike trains with different CVs using Gamma processes
        shape_factors = 1.0 / (np.array(self.cvs) ** 2)
        
        np.random.seed(0)
        spiketrains = [
            stg.homogeneous_gamma_process(
                a=shape_factor,
                b=self.firing_rate * shape_factor,
                t_start=t_start,
                t_stop=t_stop
            ) for shape_factor in shape_factors
        ]
        
        # Calculate original CVs
        cvs_original = [
            stat.cv(np.diff(st.magnitude)) for st in spiketrains
            if len(st) > 1
        ]
        results['cvs_real'] = cvs_original
        
        # Process surrogate methods
        random.seed(1)
        np.random.seed(1)
        
        for surr_method in self.surr_methods:
            print(f"Processing {surr_method}...")
            
            try:
                # Apply surrogate method
                dithered_spiketrains = [
                    self.surrogate_applier.apply_surrogate_method([st], surr_method)[0]
                    for st in spiketrains
                ]
                
                # Calculate surrogate CVs
                cvs_surrogate = [
                    stat.cv(np.diff(st.magnitude)) for st in dithered_spiketrains
                    if len(st) > 1
                ]
                
                # Ensure same length as original
                if len(cvs_surrogate) == len(cvs_original):
                    results[surr_method] = cvs_surrogate
                else:
                    warnings.warn(f"CV length mismatch for {surr_method}")
                    
            except Exception as e:
                warnings.warn(f"Failed to process {surr_method}: {e}")
        
        # Save results
        output_file = Path(self.data_path) / 'cv_change.npy'
        np.save(output_file, results)
        print(f"Saved CV change analysis to {output_file}")
    
    def generate_all_data(self) -> None:
        """
        Generate all statistical analysis datasets.
        
        This method orchestrates the generation of all required datasets
        for comprehensive surrogate method statistical analysis.
        """
        print("=" * 60)
        print("GENERATING COMPREHENSIVE SURROGATE STATISTICS DATA")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Generate all analysis types
            self.generate_clipping_and_movement_analysis()
            print()
            
            self.generate_single_rate_statistical_analysis()
            print()
            
            self.generate_firing_rate_change_analysis()
            print()
            
            self.generate_cv_change_analysis()
            print()
            
            total_time = time.time() - start_time
            print("=" * 60)
            print(f"DATA GENERATION COMPLETE")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Output directory: {self.data_path}")
            print("=" * 60)
            
        except Exception as e:
            print(f"ERROR: Data generation failed: {e}")
            raise


def main():
    """
    Main function to generate all surrogate statistics data.
    """
    try:
        # Create data generator
        generator = ComprehensiveDataGenerator()
        
        # Generate all data
        generator.generate_all_data()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        raise


if __name__ == '__main__':
    main()