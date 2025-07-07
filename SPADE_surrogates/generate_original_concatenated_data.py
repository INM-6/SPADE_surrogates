"""
Script for loading experimental spike train data and saving as concatenated trials.

This script loads spike train data from the 'multielectrode grasp' dataset,
concatenates trials for each session/epoch/trialtype combination, and saves
the processed data for efficient access in downstream analyses.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

import numpy as np
import quantities as pq
import yaml
from yaml import Loader
from tqdm import tqdm

# Import custom utilities
import rgutils


class SpikeTrainDataProcessor:
    """
    A class for processing and concatenating experimental spike train data.
    
    This class handles loading spike train data from experimental sessions,
    concatenating trials, and saving the processed data in a standardized format.
    """
    
    def __init__(self, config_path: str = "configfile.yaml", 
                 output_base_dir: str = "../data/concatenated_spiketrains"):
        """
        Initialize the spike train data processor.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the YAML configuration file, by default "configfile.yaml"
        output_base_dir : str, optional
            Base directory for saving processed data, by default "../data/concatenated_spiketrains"
        """
        self.config_path = Path(config_path)
        self.output_base_dir = Path(output_base_dir)
        self.config = self._load_configuration()
        self._setup_logging()
        
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
    
    def _load_session_data(self, session: str, epoch: str, trialtype: str) -> Optional[np.ndarray]:
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
        Optional[np.ndarray]
            Loaded spike train data, or None if loading fails
        """
        try:
            self.logger.info(f"Loading data: {session} | {epoch} | {trialtype}")
            
            spike_trains = rgutils.load_epoch_concatenated_trials(
                session=session,
                epoch=epoch,
                trialtypes=trialtype,
                SNRthresh=self.config['SNR_thresh'],
                synchsize=self.config['synchsize'],
                sep=self.separation_time
            )
            
            self.logger.info(f"Successfully loaded {len(spike_trains)} spike trains")
            return spike_trains
            
        except Exception as e:
            self.logger.error(f"Failed to load data for {session}/{epoch}/{trialtype}: {e}")
            return None
    
    def _save_processed_data(self, data: np.ndarray, output_path: Path) -> bool:
        """
        Save processed spike train data to file.
        
        Parameters
        ----------
        data : np.ndarray
            Processed spike train data
        output_path : Path
            Output file path
            
        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        try:
            np.save(output_path, data)
            self.logger.info(f"Saved data to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save data to {output_path}: {e}")
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
    
    def process_all_data(self) -> Dict[str, int]:
        """
        Process all session/epoch/trialtype combinations from configuration.
        
        Returns
        -------
        Dict[str, int]
            Summary statistics of processing results
        """
        total_combinations = len(self.sessions) * len(self.epochs) * len(self.trialtypes)
        successful_combinations = 0
        failed_combinations = 0
        
        self.logger.info(f"Starting processing of {total_combinations} data combinations")
        
        # Create progress bar for all combinations
        with tqdm(total=total_combinations, desc="Processing data combinations") as pbar:
            for session in self.sessions:
                for epoch in self.epochs:
                    for trialtype in self.trialtypes:
                        # Update progress bar description
                        pbar.set_description(f"Processing {session}/{epoch}/{trialtype}")
                        
                        # Process the combination
                        success = self.process_single_combination(session, epoch, trialtype)
                        
                        if success:
                            successful_combinations += 1
                        else:
                            failed_combinations += 1
                            
                        pbar.update(1)
        
        # Log summary
        self.logger.info(f"Processing complete: {successful_combinations} successful, "
                        f"{failed_combinations} failed")
        
        return {
            'total': total_combinations,
            'successful': successful_combinations,
            'failed': failed_combinations
        }
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the data processing configuration and planned operations.
        
        Returns
        -------
        Dict[str, Any]
            Summary of processing configuration
        """
        return {
            'config_path': str(self.config_path),
            'output_directory': str(self.output_base_dir),
            'sessions': self.sessions,
            'epochs': self.epochs,
            'trialtypes': self.trialtypes,
            'separation_time': str(self.separation_time),
            'snr_threshold': self.config['SNR_thresh'],
            'synch_size': self.config['synchsize'],
            'total_combinations': len(self.sessions) * len(self.epochs) * len(self.trialtypes)
        }


def main():
    """
    Main function to run the spike train data processing pipeline.
    """
    try:
        # Initialize processor
        processor = SpikeTrainDataProcessor()
        
        # Print processing summary
        summary = processor.get_processing_summary()
        print("\n" + "="*60)
        print("SPIKE TRAIN DATA PROCESSING SUMMARY")
        print("="*60)
        print(f"Configuration file: {summary['config_path']}")
        print(f"Output directory: {summary['output_directory']}")
        print(f"Sessions: {len(summary['sessions'])} ({', '.join(summary['sessions'])})")
        print(f"Epochs: {len(summary['epochs'])} ({', '.join(summary['epochs'])})")
        print(f"Trial types: {len(summary['trialtypes'])} ({', '.join(summary['trialtypes'])})")
        print(f"Separation time: {summary['separation_time']}")
        print(f"SNR threshold: {summary['snr_threshold']}")
        print(f"Synch size: {summary['synch_size']}")
        print(f"Total combinations to process: {summary['total_combinations']}")
        print("="*60 + "\n")
        
        # Process all data
        results = processor.process_all_data()
        
        # Print final summary
        print("\n" + "="*60)
        print("PROCESSING RESULTS")
        print("="*60)
        print(f"Total combinations: {results['total']}")
        print(f"Successfully processed: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Success rate: {results['successful']/results['total']*100:.1f}%")
        print("="*60)
        
        if results['failed'] > 0:
            print("\nWARNING: Some combinations failed to process. Check the log file for details.")
            sys.exit(1)
        else:
            print("\nAll data combinations processed successfully!")
            
    except Exception as e:
        print(f"Fatal error during processing: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()