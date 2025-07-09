"""
Updated script for loading experimental spike train data and saving as concatenated trials.

This script loads spike train data from the 'multielectrode grasp' dataset,
concatenates trials for each session/epoch/trialtype combination, and saves
the processed data for efficient access in downstream analyses.

Updated to use the new improved CachedDataLoader for better performance and reliability.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import time

import numpy as np
import quantities as pq
import yaml
from yaml import Loader
import neo

# Import the improved utilities
from rgutils import CachedDataLoader, calc_spiketrains_SNR, remove_synchrofact_spikes


class SpikeTrainDataProcessor:
    """
    A class for processing and concatenating experimental spike train data.
    
    This class handles loading spike train data from experimental sessions,
    concatenating trials, and saving the processed data in a standardized format.
    
    Updated to use the improved CachedDataLoader for better performance and reliability.
    """
    
    def __init__(self, config_path: str = "configfile.yaml", 
                 output_base_dir: str = "../data/concatenated_spiketrains",
                 use_cache: bool = True):
        """
        Initialize the spike train data processor.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the YAML configuration file, by default "configfile.yaml"
        output_base_dir : str, optional
            Base directory for saving processed data, by default "../data/concatenated_spiketrains"
        use_cache : bool, optional
            Whether to use caching for improved performance, by default True
        """
        self.config_path = Path(config_path)
        self.output_base_dir = Path(output_base_dir)
        self.use_cache = use_cache
        
        self.config = self._load_configuration()
        self._setup_logging()
        
        # Initialize the improved cached data loader
        if self.use_cache:
            self.cached_loader = CachedDataLoader(verbose=True)
            self.logger.info("Initialized improved cached data loader with verbose logging")
        else:
            self.cached_loader = None
            self.logger.info("Caching disabled - using direct loading")
        
    def _load_configuration(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
            
        Raises
        ------
        FileNotFoundError
            If configuration file doesn't exist
        yaml.YAMLError
            If configuration file is malformed
        """
        try:
            with open(self.config_path, 'r') as stream:
                config = yaml.load(stream, Loader=Loader)
            
            self._validate_configuration(config)
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")
    
    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """
        Validate that the configuration contains required fields.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to validate
            
        Raises
        ------
        KeyError
            If required configuration keys are missing
        """
        required_keys = [
            'sessions', 'epochs', 'trialtypes', 'winlen', 
            'unit', 'binsize', 'SNR_thresh', 'synchsize'
        ]
        
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(f"Missing required configuration keys: {missing_keys}")
    
    def _setup_logging(self) -> None:
        """Set up logging configuration for better debugging and monitoring."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('spike_train_processing.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    @property
    def sessions(self) -> List[str]:
        """Get list of sessions from configuration."""
        return self.config['sessions']
    
    @property
    def epochs(self) -> List[str]:
        """Get list of epochs from configuration."""
        return self.config['epochs']
    
    @property
    def trialtypes(self) -> List[str]:
        """Get list of trial types from configuration."""
        return self.config['trialtypes']
    
    @property
    def separation_time(self) -> pq.Quantity:
        """
        Calculate separation time between trials.
        
        Returns
        -------
        pq.Quantity
            Separation time between concatenated trials
        """
        winlen = self.config['winlen']
        unit = self.config['unit']
        binsize = (self.config['binsize'] * pq.s).rescale(unit)
        return 2 * winlen * binsize
    
    def _create_output_directory(self, session: str) -> Path:
        """
        Create output directory for a session if it doesn't exist.
        
        Parameters
        ----------
        session : str
            Session identifier
            
        Returns
        -------
        Path
            Path to the session output directory
        """
        session_dir = self.output_base_dir / session
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    def _concatenate_epoch_data(self, epoch_data: Dict[int, List], sep: pq.Quantity) -> List:
        """
        Concatenate trials from epoch data using the correct original logic.
        
        This now follows the exact logic from load_epoch_concatenated_trials.
        
        Parameters
        ----------
        epoch_data : Dict[int, List]
            Dictionary mapping SUA IDs to lists of spike trains
        sep : pq.Quantity
            Separation time between trials
            
        Returns
        -------
        List
            List of concatenated spike trains
        """
        if not epoch_data:
            return []
        
        # Get timing information from the first spike train (following original logic)
        sample_st_list = list(epoch_data.values())[0]
        if not sample_st_list:
            return []
        
        sample_st = sample_st_list[0]
        t_pre = abs(sample_st.t_start)
        t_post = abs(sample_st.t_stop) 
        time_unit = sample_st.units
        
        self.logger.info(f"Concatenation parameters:")
        self.logger.info(f"  t_pre: {t_pre}, t_post: {t_post}")
        self.logger.info(f"  separation: {sep}")
        self.logger.info(f"  time_unit: {time_unit}")
        
        # Calculate trial duration (following original logic)
        trial_duration = (t_post + t_pre + sep).rescale(time_unit)
        self.logger.info(f"  trial_duration: {trial_duration}")
        
        # Get unique trial IDs from all spike trains (following original logic)
        trial_ids_of_chosen_types = []
        for st_list in epoch_data.values():
            for st in st_list:
                if (hasattr(st, 'annotations') and 
                    isinstance(st.annotations, dict) and
                    'trial_id' in st.annotations):
                    trial_ids_of_chosen_types.append(st.annotations['trial_id'])
        
        trial_ids_of_chosen_types = np.unique(trial_ids_of_chosen_types)
        self.logger.info(f"  trial_ids: {trial_ids_of_chosen_types}")
        
        # Concatenate trials for each SUA (following original logic exactly)
        concatenated_trains = []
        
        for sua_id in sorted(epoch_data.keys()):
            trials_to_concatenate = []
            original_times = []
            
            # Create list of trials, each shifted by trial_duration*trial_id (ORIGINAL LOGIC)
            for tr in epoch_data[sua_id]:
                if (hasattr(tr, 'annotations') and 
                    isinstance(tr.annotations, dict) and
                    'trial_id' in tr.annotations):
                    trial_id = tr.annotations['trial_id']
                else:
                    trial_id = 0
                
                # THIS IS THE KEY: Use trial_id from annotations, not sequential index
                shifted_times = (tr.rescale(time_unit).magnitude + 
                               (trial_duration * trial_id).rescale(time_unit).magnitude)
                
                trials_to_concatenate.append(shifted_times)
                original_times.extend(list(tr.magnitude))
            
            # Concatenate the trials (following original logic)
            if len(trials_to_concatenate) > 0:
                trials_to_concatenate = np.hstack(trials_to_concatenate)
            else:
                trials_to_concatenate = np.array([])
            
            # Create concatenated spike train (following original logic)
            total_duration = (trial_duration * max(trial_ids_of_chosen_types) + trial_duration).rescale(pq.s)
            
            concatenated_st = neo.SpikeTrain(
                trials_to_concatenate * time_unit,
                t_start=0 * time_unit,
                t_stop=total_duration
            )
            
            # Copy annotations from first spike train (following original logic)
            if epoch_data[sua_id]:
                reference_st = epoch_data[sua_id][0]
                if (hasattr(reference_st, 'annotations') and 
                    isinstance(reference_st.annotations, dict)):
                    for key, value in reference_st.annotations.items():
                        if key != 'trial_id':  # Exclude trial_id as in original
                            concatenated_st.annotations[key] = value
            
            # Add original times annotation (following original logic)
            concatenated_st.annotations['original_times'] = original_times
            
            concatenated_trains.append(concatenated_st)
            
        self.logger.info(f"Concatenated {len(concatenated_trains)} spike trains")
        
        # Log spike counts to verify concatenation worked
        if concatenated_trains:
            spike_counts = [len(st) for st in concatenated_trains]
            total_spikes = sum(spike_counts)
            self.logger.info(f"  Total spikes after concatenation: {total_spikes}")
            self.logger.info(f"  Sample spike counts: {spike_counts[:5]}...")
            
            if total_spikes == 0:
                self.logger.error("❌ CRITICAL: Concatenation resulted in 0 spikes!")
                # Debug the first unit
                if epoch_data:
                    first_sua_id = list(epoch_data.keys())[0]
                    first_trials = epoch_data[first_sua_id]
                    self.logger.error(f"  First SUA {first_sua_id}: {len(first_trials)} trials")
                    for i, trial in enumerate(first_trials[:3]):
                        self.logger.error(f"    Trial {i}: {len(trial)} spikes, "
                                        f"trial_id: {trial.annotations.get('trial_id', 'missing')}")
        
        return concatenated_trains
    
    def remove_synchrony_artifacts(self, spike_trains: List, synch_size: int) -> List:
        """
        Remove synchrony artifacts using appropriate time resolution for neural data.
        
        Based on the Nature Scientific Data paper (https://www.nature.com/articles/sdata201855),
        the recording system uses much finer temporal resolution than 1 second.
        
        Parameters
        ----------
        spike_trains : List
            List of spike trains
        synch_size : int
            Synchrony threshold
            
        Returns
        -------
        List
            Spike trains with synchrony artifacts removed
        """
        if synch_size <= 0:
            return spike_trains
        
        self.logger.info(f"Applying synchrony removal with threshold {synch_size}")
        
        # Log spike counts BEFORE synchrony removal
        if spike_trains:
            before_counts = [len(st) for st in spike_trains]
            total_before = sum(before_counts)
            self.logger.info(f"BEFORE synchrony removal: {total_before} total spikes across {len(spike_trains)} units")
            self.logger.info(f"  Sample spike counts: {before_counts[:5]}...")
        
        try:
            if spike_trains and len(spike_trains) > 0:
                import quantities as pq
                
                # Based on the Nature paper, use appropriate time resolution for neural synchrony
                # The recording system sampling is much finer than 1 second
                # For Blackrock systems (typical for this type of data), sampling is ~30 kHz
                # So the natural resolution would be ~33 microseconds (1/30000 s)
                
                # Use microsecond precision for synchrony detection (appropriate for neural data)
                # This matches typical electrophysiology recording precision
                Dt = 33.33 * pq.us   # ~33 microseconds (approximate 30 kHz sampling period)
                Dt2 = 33.33 * pq.us  # Same for neighbor removal
                
                self.logger.info(f"Using appropriate neural synchrony detection:")
                self.logger.info(f"  dt (synchrony detection window): {Dt}")
                self.logger.info(f"  dt2 (neighbor removal window): {Dt2}")
                self.logger.info(f"  n (minimum synchrony size): {synch_size}")
                self.logger.info(f"  Note: Using ~30 kHz sampling period precision for neural data")
                
                # Optional: Show synchrony statistics before removal
                try:
                    from rgutils import find_synchrofact_spikes
                    sync_result = find_synchrofact_spikes(spike_trains, n=synch_size, dt=Dt)
                    if len(sync_result) > 1 and sync_result[1] is not None:
                        n_sync_spikes = len(sync_result[1])
                        self.logger.info(f"  Synchronous spikes detected before removal: {n_sync_spikes}")
                    else:
                        self.logger.info(f"  No synchronous spikes detected with current parameters")
                except Exception as e:
                    self.logger.warning(f"Could not get sync statistics: {e}")
                
                # Apply synchrony removal with appropriate time resolution
                cleaned_trains = remove_synchrofact_spikes(spike_trains, n=synch_size, dt=Dt, dt2=Dt2)
                
                # Copy annotations back (original logic)
                if cleaned_trains and len(cleaned_trains) == len(spike_trains):
                    for i in range(len(spike_trains)):
                        if i < len(cleaned_trains) and hasattr(spike_trains[i], 'annotations'):
                            cleaned_trains[i].annotations = spike_trains[i].annotations
                else:
                    self.logger.warning(f"Cleaned trains count mismatch: {len(cleaned_trains)} vs {len(spike_trains)}")
                
                # Log spike counts AFTER synchrony removal
                if cleaned_trains:
                    after_counts = [len(st) for st in cleaned_trains]
                    total_after = sum(after_counts)
                    self.logger.info(f"AFTER synchrony removal: {total_after} total spikes across {len(cleaned_trains)} units")
                    self.logger.info(f"  Sample spike counts: {after_counts[:5]}...")
                    
                    # Calculate removal statistics
                    removal_rate = (total_before - total_after) / total_before * 100 if total_before > 0 else 0
                    removed_spikes = total_before - total_after
                    self.logger.info(f"  Spikes removed: {removed_spikes:,} ({removal_rate:.1f}%)")
                    
                    # Provide guidance on removal rates with appropriate expectations
                    if removal_rate > 20:
                        if removal_rate > 50:
                            self.logger.warning(f"⚠️ High spike removal rate: {removal_rate:.1f}%")
                            self.logger.warning(f"   Consider increasing synchsize threshold (currently {synch_size})")
                            self.logger.warning(f"   Or setting synchsize=0 to disable synchrony removal")
                        else:
                            self.logger.info(f"  Moderate synchrony removal ({removal_rate:.1f}%) - may be normal for dense electrode arrays")
                    elif removal_rate > 5:
                        self.logger.info(f"  Low-moderate synchrony removal - appropriate for multi-electrode data")
                    else:
                        self.logger.info(f"  Minimal synchrony removal - data has few artifacts")
                    
                    if total_after == 0:
                        self.logger.error(f"❌ CRITICAL: All spikes removed by synchrony removal!")
                        self.logger.error(f"   This suggests synchsize={synch_size} is too aggressive")
                        self.logger.error(f"   Returning original spike trains to preserve data.")
                        return spike_trains
                    
                else:
                    self.logger.error(f"❌ Synchrony removal returned empty list!")
                    return spike_trains
                
                # Final logging
                original_count = len(spike_trains)
                cleaned_count = len(cleaned_trains)
                
                if cleaned_count != original_count:
                    self.logger.warning(f"Synchrony removal changed spike train count: "
                                      f"{original_count} → {cleaned_count}")
                else:
                    self.logger.info(f"✅ Synchrony removal completed successfully")
                    self.logger.info(f"   Retained {cleaned_count} spike trains with {total_after:,} total spikes")
                
                return cleaned_trains
            else:
                self.logger.warning("Empty spike trains list provided to synchrony removal")
                return spike_trains
                
        except Exception as e:
            self.logger.error(f"Error in synchrony removal: {e}")
            import traceback
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self.logger.warning("Returning original spike trains without synchrony removal")
            return spike_trains
    
    def _load_session_data(self, session: str, epoch: str, trialtype: str) -> Optional[List]:
        """
        Load spike train data for a specific session, epoch, and trial type.
        
        Parameters
        ----------
        session : str
            Session identifier
        epoch : str
            Epoch identifier
        trialtype : str
            Trial type identifier
            
        Returns
        -------
        Optional[List]
            Loaded and processed spike train data, or None if loading fails
        """
        try:
            self.logger.info(f"Loading data: {session} | {epoch} | {trialtype}")
            
            if self.use_cache and self.cached_loader:
                # Use improved cached loading for epoch data
                epoch_data = self.cached_loader.load_epoch_data_cached(
                    session_name=session,
                    epoch=epoch,
                    trialtypes=trialtype,
                    SNRthresh=self.config['SNR_thresh']
                )
                
                if not epoch_data:
                    self.logger.warning(f"No data found for {session}/{epoch}/{trialtype}")
                    return []
                
                # Concatenate the trials
                spike_trains = self._concatenate_epoch_data(epoch_data, self.separation_time)
                
                # Apply synchrony removal if requested
                synchsize = self.config['synchsize']
                if synchsize > 0 and spike_trains:
                    # SAFETY CHECK: Temporarily disable synchrony removal to test
                    # You can enable this to bypass synchrony removal for debugging
                    DISABLE_SYNCHRONY_REMOVAL = False  # Set to True to skip synchrony removal
                    
                    if DISABLE_SYNCHRONY_REMOVAL:
                        self.logger.warning(f"⚠️ SYNCHRONY REMOVAL DISABLED for debugging")
                        self.logger.info(f"Skipping synchrony removal (would use threshold {synchsize})")
                    else:
                        spike_trains = self.remove_synchrony_artifacts(spike_trains, synchsize)
            else:
                # Fallback to direct loading (you'll need to implement this if needed)
                self.logger.error("Direct loading not implemented. Please use caching.")
                return None
            
            self.logger.info(f"Successfully processed {len(spike_trains)} spike trains")
            return spike_trains
            
        except Exception as e:
            import traceback
            
            # Get the full traceback information
            tb_str = traceback.format_exc()
            
            # Log the detailed error information
            self.logger.error(f"Failed to load data for {session}/{epoch}/{trialtype}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error message: {str(e)}")
            self.logger.error(f"Full traceback:\n{tb_str}")
            
            # Print to console as well for immediate visibility
            print(f"\n{'='*60}")
            print(f"ERROR LOADING DATA: {session}/{epoch}/{trialtype}")
            print(f"{'='*60}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"\nFull traceback:")
            print(tb_str)
            print(f"{'='*60}\n")
            
            return None
    
    def _save_processed_data(self, data: List, output_path: Path) -> bool:
        """
        Save processed spike train data to file.
        
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
                # Convert spike trains to a more standardized format for saving
                processed_data = []
                
                total_spikes_check = 0  # For debugging
                
                for i, spike_train in enumerate(data):
                    # Extract spike times - this is the critical part!
                    if hasattr(spike_train, 'times'):
                        spike_times = spike_train.times.magnitude
                    elif hasattr(spike_train, 'magnitude'):
                        spike_times = spike_train.magnitude
                    else:
                        # Fallback: try to convert to array
                        spike_times = np.array(spike_train).flatten()
                    
                    total_spikes_check += len(spike_times)
                    
                    # Extract the essential data from each spike train
                    spike_data = {
                        'times': spike_times,  # This should be the actual spike times array
                        'units': str(spike_train.units) if hasattr(spike_train, 'units') else 's',
                        't_start': float(spike_train.t_start.magnitude) if hasattr(spike_train, 't_start') else 0.0,
                        't_stop': float(spike_train.t_stop.magnitude) if hasattr(spike_train, 't_stop') else 0.0,
                        'annotations': {},
                        'spike_train_id': i
                    }
                    
                    # Handle annotations more carefully
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
                                self.logger.warning(f"Skipping annotation '{key}': {e}")
                                continue
                    
                    processed_data.append(spike_data)
                
                # Debug: Check if spikes are preserved
                self.logger.info(f"DEBUG: Before save - total spikes across all units: {total_spikes_check}")
                
                # Verify data before saving
                saved_spike_counts = [len(st['times']) for st in processed_data]
                total_saved_spikes = sum(saved_spike_counts)
                
                self.logger.info(f"DEBUG: Prepared for saving - {total_saved_spikes} total spikes")
                self.logger.info(f"DEBUG: Spike counts per unit: {saved_spike_counts[:10]}...")  # First 10
                
                # Save as numpy array with allow_pickle=True
                np.save(output_path, np.array(processed_data, dtype=object), allow_pickle=True)
                
                # Verify what was actually saved
                verification = verify_saved_data(str(output_path))
                actual_total_spikes = verification.get('total_spikes', 0)
                
                self.logger.info(f"Saved {len(processed_data)} spike trains to: {output_path}")
                self.logger.info(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
                self.logger.info(f"   VERIFICATION: Total spikes saved: {actual_total_spikes}")
                
                if actual_total_spikes != total_spikes_check:
                    self.logger.error(f"   ❌ SPIKE COUNT MISMATCH!")
                    self.logger.error(f"   Expected: {total_spikes_check}, Saved: {actual_total_spikes}")
                    
                    # Debug the first few units
                    for j, (original_st, saved_data) in enumerate(zip(data[:3], processed_data[:3])):
                        orig_count = len(original_st) if hasattr(original_st, '__len__') else 0
                        saved_count = len(saved_data['times'])
                        self.logger.error(f"   Unit {j}: Original {orig_count} → Saved {saved_count}")
                        
                        if orig_count > 0 and saved_count == 0:
                            self.logger.error(f"   Unit {j} spike times type: {type(original_st.times)}")
                            self.logger.error(f"   Unit {j} first few times: {original_st.times[:5]}")
                else:
                    self.logger.info(f"   ✅ Spike count verification PASSED")
                
                # Log statistics about the saved data
                if actual_total_spikes > 0:
                    spike_counts = verification.get('spike_counts', [])
                    mean_spikes = verification.get('mean_spikes', 0)
                    
                    self.logger.info(f"   Mean spikes per unit: {mean_spikes:.1f}")
                    self.logger.info(f"   Spike count range: {min(spike_counts)} - {max(spike_counts)}")
                else:
                    self.logger.warning(f"   ⚠️ WARNING: No spikes found in saved data!")
                    
            else:
                # Save empty array
                np.save(output_path, np.array([]), allow_pickle=True)
                self.logger.info(f"Saved empty array to: {output_path}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save data to {output_path}: {e}")
            import traceback
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return False
    
    def process_single_combination(self, session: str, epoch: str, trialtype: str) -> bool:
        """
        Process a single session/epoch/trialtype combination.
        
        Parameters
        ----------
        session : str
            Session identifier
        epoch : str
            Epoch identifier
        trialtype : str
            Trial type identifier
            
        Returns
        -------
        bool
            True if processing was successful, False otherwise
        """
        # Create output directory
        session_dir = self._create_output_directory(session)
        
        # Load data
        spike_trains = self._load_session_data(session, epoch, trialtype)
        if spike_trains is None:
            return False
        
        # Define output path
        output_filename = f"{epoch}_{trialtype}.npy"
        output_path = session_dir / output_filename
        
        # Save processed data
        return self._save_processed_data(spike_trains, output_path)
    
    def process_session_efficiently(self, session: str) -> Dict[str, int]:
        """
        Process all epoch/trialtype combinations for a single session efficiently.
        
        This method leverages the improved caching system to process multiple combinations
        much faster by reusing expensive operations like NIX loading and epoch cutting.
        
        Parameters
        ----------
        session : str
            Session identifier
            
        Returns
        -------
        Dict[str, int]
            Processing statistics for this session
        """
        session_successful = 0
        session_failed = 0
        total_combinations = len(self.epochs) * len(self.trialtypes)
        
        self.logger.info(f"🚀 STARTING session {session} with {total_combinations} combinations")
        
        # Create output directory
        session_dir = self._create_output_directory(session)
        self.logger.info(f"📁 Output directory: {session_dir}")
        
        start_time = time.time()
        
        # Process each combination
        for epoch_idx, epoch in enumerate(self.epochs):
            self.logger.info(f"📋 Processing epoch {epoch_idx+1}/{len(self.epochs)}: '{epoch}'")
            
            for trial_idx, trialtype in enumerate(self.trialtypes):
                combination_start = time.time()
                combination_name = f"{epoch}/{trialtype}"
                
                try:
                    self.logger.info(f"   🔄 Processing {combination_name}...")
                    
                    # Load and process data
                    spike_trains = self._load_session_data(session, epoch, trialtype)
                    
                    if spike_trains is not None:
                        # Save the data
                        output_filename = f"{epoch}_{trialtype}.npy"
                        output_path = session_dir / output_filename
                        
                        if self._save_processed_data(spike_trains, output_path):
                            combination_time = time.time() - combination_start
                            self.logger.info(f"   ✅ {combination_name} COMPLETED in {combination_time:.1f}s "
                                           f"({len(spike_trains)} spike trains)")
                            session_successful += 1
                        else:
                            self.logger.error(f"   ❌ Failed to save {combination_name}")
                            session_failed += 1
                    else:
                        self.logger.error(f"   ❌ Failed to load {combination_name}")
                        session_failed += 1
                        
                except Exception as e:
                    combination_time = time.time() - combination_start
                    self.logger.error(f"   💥 Error processing {combination_name} "
                                    f"after {combination_time:.1f}s: {e}")
                    session_failed += 1
                    
                    # Print stack trace for debugging
                    import traceback
                    self.logger.error(f"   📋 Stack trace:\n{traceback.format_exc()}")
        
        # Print session summary
        total_time = time.time() - start_time
        success_rate = (session_successful / total_combinations) * 100
        
        self.logger.info(f"🏁 Session {session} SUMMARY:")
        self.logger.info(f"   ✅ Successful: {session_successful}/{total_combinations} ({success_rate:.1f}%)")
        self.logger.info(f"   ❌ Failed: {session_failed}")
        self.logger.info(f"   ⏱️  Total time: {total_time:.1f}s")
        self.logger.info(f"   📊 Average per combination: {total_time/total_combinations:.1f}s")
        
        return {
            'total': total_combinations,
            'successful': session_successful,
            'failed': session_failed
        }
    
    def process_all_data(self) -> Dict[str, int]:
        """
        Process all session/epoch/trialtype combinations from configuration.
        
        Returns
        -------
        Dict[str, int]
            Summary statistics of processing results
        """
        total_combinations = len(self.sessions) * len(self.epochs) * len(self.trialtypes)
        total_successful = 0
        total_failed = 0
        
        self.logger.info(f"🚀 STARTING FULL PROCESSING")
        self.logger.info(f"📊 Total combinations: {total_combinations}")
        self.logger.info(f"📋 Sessions: {len(self.sessions)} {self.sessions}")
        self.logger.info(f"📋 Epochs: {len(self.epochs)} {self.epochs}")
        self.logger.info(f"📋 Trial types: {len(self.trialtypes)} {self.trialtypes}")
        
        if self.use_cache:
            self.logger.info("⚡ Using improved cached loading for enhanced performance")
        else:
            self.logger.info("🐌 Using direct loading (no caching)")
        
        overall_start_time = time.time()
        
        # Process each session
        for session_idx, session in enumerate(self.sessions):
            session_start_time = time.time()
            
            self.logger.info(f"\n" + "="*80)
            self.logger.info(f"🎯 PROCESSING SESSION {session_idx+1}/{len(self.sessions)}: {session}")
            self.logger.info(f"="*80)
            
            session_results = self.process_session_efficiently(session)
            
            session_time = time.time() - session_start_time
            total_successful += session_results['successful']
            total_failed += session_results['failed']
            
            # Session completion summary
            session_success_rate = (session_results['successful'] / session_results['total']) * 100
            self.logger.info(f"🏁 SESSION {session} COMPLETED in {session_time:.1f}s")
            self.logger.info(f"   Success rate: {session_success_rate:.1f}% "
                            f"({session_results['successful']}/{session_results['total']})")
            
            # Overall progress
            completed_combinations = total_successful + total_failed
            overall_progress = (completed_combinations / total_combinations) * 100
            elapsed_time = time.time() - overall_start_time
            
            if completed_combinations > 0:
                estimated_total_time = (elapsed_time / completed_combinations) * total_combinations
                remaining_time = estimated_total_time - elapsed_time
                
                self.logger.info(f"📈 OVERALL PROGRESS: {overall_progress:.1f}% "
                            f"({completed_combinations}/{total_combinations})")
                self.logger.info(f"⏱️  Elapsed: {elapsed_time:.1f}s, "
                            f"Estimated remaining: {remaining_time:.1f}s")
        
        # Final processing summary
        total_time = time.time() - overall_start_time
        overall_success_rate = (total_successful / total_combinations) * 100
        
        self.logger.info(f"\n" + "="*80)
        self.logger.info(f"🎉 PROCESSING COMPLETE!")
        self.logger.info(f"="*80)
        self.logger.info(f"✅ Total successful: {total_successful}/{total_combinations} ({overall_success_rate:.1f}%)")
        self.logger.info(f"❌ Total failed: {total_failed}")
        self.logger.info(f"⏱️  Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        self.logger.info(f"⚡ Average per combination: {total_time/total_combinations:.1f}s")
        
        return {
            'total': total_combinations,
            'successful': total_successful,
            'failed': total_failed
        }
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the data processing configuration and planned operations.
        
        Returns
        -------
        Dict[str, Any]
            Summary of processing configuration
        """
        summary = {
            'config_path': str(self.config_path),
            'output_directory': str(self.output_base_dir),
            'sessions': self.sessions,
            'epochs': self.epochs,
            'trialtypes': self.trialtypes,
            'separation_time': str(self.separation_time),
            'snr_threshold': self.config['SNR_thresh'],
            'synch_size': self.config['synchsize'],
            'total_combinations': len(self.sessions) * len(self.epochs) * len(self.trialtypes),
            'caching_enabled': self.use_cache
        }
        
        return summary


def load_processed_spike_trains(file_path: str) -> List:
    """
    Load processed spike train data from a saved file.
    
    Parameters
    ----------
    file_path : str
        Path to the saved spike train file
        
    Returns
    -------
    List
        List of reconstructed neo.SpikeTrain objects
        
    Example
    -------
    >>> spike_trains = load_processed_spike_trains("../data/concatenated_spiketrains/i140613s001/start_PGHF.npy")
    >>> print(f"Loaded {len(spike_trains)} spike trains")
    """
    try:
        # Load the data
        data = np.load(file_path, allow_pickle=True)
        
        if len(data) == 0:
            return []
        
        # Reconstruct spike trains
        spike_trains = []
        
        for spike_data in data:
            # Reconstruct the spike train
            times = spike_data['times']
            units_str = spike_data['units']
            
            # Parse units
            import quantities as pq
            if units_str == 's':
                units = pq.s
            elif units_str == 'ms':
                units = pq.ms
            else:
                units = pq.s  # Default fallback
            
            # Create spike train
            spike_train = neo.SpikeTrain(
                times * units,
                t_start=spike_data['t_start'] * units,
                t_stop=spike_data['t_stop'] * units
            )
            
            # Restore annotations
            spike_train.annotations.update(spike_data['annotations'])
            
            spike_trains.append(spike_train)
        
        return spike_trains
        
    except Exception as e:
        print(f"Error loading spike trains from {file_path}: {e}")
        return []


def verify_saved_data(file_path: str) -> Dict[str, Any]:
    """
    Verify and analyze saved spike train data.
    
    Parameters
    ----------
    file_path : str
        Path to the saved spike train file
        
    Returns
    -------
    Dict[str, Any]
        Analysis summary of the saved data
    """
    try:
        # Load the raw data
        data = np.load(file_path, allow_pickle=True)
        
        if len(data) == 0:
            return {'status': 'empty', 'count': 0}
        
        # Analyze the data
        analysis = {
            'status': 'success',
            'count': len(data),
            'spike_counts': [],
            'total_spikes': 0,
            'time_ranges': [],
            'units': set(),
            'annotations_keys': set()
        }
        
        for spike_data in data:
            spike_count = len(spike_data['times'])
            analysis['spike_counts'].append(spike_count)
            analysis['total_spikes'] += spike_count
            analysis['time_ranges'].append((spike_data['t_start'], spike_data['t_stop']))
            analysis['units'].add(spike_data['units'])
            analysis['annotations_keys'].update(spike_data['annotations'].keys())
        
        # Summary statistics
        analysis['mean_spikes'] = np.mean(analysis['spike_counts'])
        analysis['std_spikes'] = np.std(analysis['spike_counts'])
        analysis['min_spikes'] = min(analysis['spike_counts'])
        analysis['max_spikes'] = max(analysis['spike_counts'])
        analysis['units'] = list(analysis['units'])
        analysis['annotations_keys'] = list(analysis['annotations_keys'])
        
        return analysis
        
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def test_save_load_functionality():
    """
    Test the save and load functionality with sample data.
    """
    print("Testing save/load functionality...")
    
    try:
        # Create some test spike trains
        import neo
        import quantities as pq
        import numpy as np
        
        test_spike_trains = []
        
        for i in range(3):
            # Create spike trains with different lengths
            n_spikes = np.random.randint(10, 100)
            spike_times = np.sort(np.random.uniform(0, 10, n_spikes))
            
            st = neo.SpikeTrain(
                spike_times * pq.s,
                t_start=0 * pq.s,
                t_stop=10 * pq.s
            )
            
            # Add some annotations
            st.annotations = {
                'unit_id': i + 100,
                'SNR': np.random.uniform(2, 8),
                'trial_separation': 0.5,
                'concatenated': True,
                'n_trials': 5
            }
            
            test_spike_trains.append(st)
        
        # Test save
        test_file = Path("test_spike_trains.npy")
        processor = SpikeTrainDataProcessor()
        
        print(f"Saving {len(test_spike_trains)} test spike trains...")
        success = processor._save_processed_data(test_spike_trains, test_file)
        
        if success:
            print("✅ Save successful")
            
            # Test load
            print("Loading spike trains back...")
            loaded_trains = load_processed_spike_trains(str(test_file))
            
            print(f"✅ Loaded {len(loaded_trains)} spike trains")
            
            # Verify data integrity
            for i, (original, loaded) in enumerate(zip(test_spike_trains, loaded_trains)):
                print(f"Unit {i}:")
                print(f"  Original spikes: {len(original)}")
                print(f"  Loaded spikes: {len(loaded)}")
                print(f"  Times match: {np.allclose(original.times.magnitude, loaded.times.magnitude)}")
                print(f"  Annotations: {loaded.annotations.get('unit_id', 'missing')}")
            
            # Test verification function
            print("\nVerifying saved data...")
            verification = verify_saved_data(str(test_file))
            print(f"Verification result: {verification}")
            
            # Clean up
            test_file.unlink()
            print("✅ Test completed successfully")
            
        else:
            print("❌ Save failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function with enhanced logging and progress tracking.
    """
    try:
        start_time = time.time()
        
        # Initialize processor with improved caching
        processor = SpikeTrainDataProcessor(use_cache=True)
        
        # Print processing summary
        summary = processor.get_processing_summary()
        print("\n" + "="*80)
        print("🧠 SPIKE TRAIN DATA PROCESSING - IMPROVED VERSION")
        print("="*80)
        print(f"📁 Configuration file: {summary['config_path']}")
        print(f"📁 Output directory: {summary['output_directory']}")
        print(f"🗂️  Sessions: {len(summary['sessions'])} ({', '.join(summary['sessions'])})")
        print(f"📋 Epochs: {len(summary['epochs'])} ({', '.join(summary['epochs'])})")
        print(f"🏷️  Trial types: {len(summary['trialtypes'])} ({', '.join(summary['trialtypes'])})")
        print(f"⏱️  Separation time: {summary['separation_time']}")
        print(f"📊 SNR threshold: {summary['snr_threshold']}")
        print(f"🔄 Synch size: {summary['synch_size']}")
        print(f"📈 Total combinations: {summary['total_combinations']}")
        print(f"⚡ Caching enabled: {summary['caching_enabled']}")
        print("="*80 + "\n")
        
        # Process all data with improved loader
        results = processor.process_all_data()
        
        # Print final summary
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("🎉 FINAL PROCESSING RESULTS")
        print("="*80)
        print(f"📊 Total combinations: {results['total']}")
        print(f"✅ Successfully processed: {results['successful']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"📈 Success rate: {results['successful']/results['total']*100:.1f}%")
        print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print("="*80)
        
        if results['failed'] > 0:
            print("\n⚠️  WARNING: Some combinations failed to process. Check the detailed logs above.")
            sys.exit(1)
        else:
            print("\n🎉 All data combinations processed successfully!")
            
    except Exception as e:
        print(f"💥 Fatal error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import sys
    
    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_save_load_functionality()
    else:
        main()