# -*- coding: utf-8 -*-
"""
Improved NIX file utilities for Reach-to-Grasp data with enhanced readability.

This module provides utilities for loading and processing neural data from NIX files,
with proper handling of spike train annotations and epoch extraction.

Key Features:
- Correct handling of numpy.bool_ SUA annotations
- Proper annotation preservation during epoch cutting
- Integration with neo.utils for robust event extraction
- Kelly SNR calculation for spike train quality assessment
"""

import os
import sys
import copy
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import quantities as pq
import neo
import neo.utils
from neo import Event, SpikeTrain, Segment, Epoch

# Configure logging
logger = logging.getLogger(__name__)


# ==========================================================================
# File path and loading utilities
# ==========================================================================

def get_nix_filename(session_name: str, include_raw: bool = False) -> str:
    """
    Get the full path to a NIX file for the specified session.
    
    Parameters
    ----------
    session_name : str
        Session identifier (e.g., 'i140703s001')
    include_raw : bool, optional
        Whether to include raw 30kHz data. Only applies to certain paths.
        Default: False
        
    Returns
    -------
    str
        Full path to the NIX file
        
    Notes
    -----
    The _no_raw option is only checked for local dataset directories,
    not for the production path (/work/datasets/r2g_nix/).
    """
    # Define search paths in order of preference
    possible_paths = [
        "/work/datasets/r2g_nix/",                        # Production path
        "../data/multielectrode_grasp/datasets_nix/",     # Relative path (supports no_raw)
        "./datasets_nix/",                                # Local path (supports no_raw)
        "./"                                              # Current directory
    ]
    
    # Paths that support the _no_raw option
    no_raw_supported_paths = {
        "../data/multielectrode_grasp/datasets_nix/",
        "./datasets_nix/"
    }
    
    for base_path in possible_paths:
        # For paths that support no_raw, try the no_raw version first (if requested)
        if not include_raw and base_path in no_raw_supported_paths:
            no_raw_filename = f"{session_name}_no_raw.nix"
            no_raw_path = os.path.join(base_path, no_raw_filename)
            if os.path.exists(no_raw_path):
                return no_raw_path
        
        # Try the regular version
        regular_filename = f"{session_name}.nix"
        regular_path = os.path.join(base_path, regular_filename)
        if os.path.exists(regular_path):
            return regular_path
    
    # Return default path if nothing found (for error reporting)
    default_filename = f"{session_name}_no_raw.nix" if not include_raw else f"{session_name}.nix"
    return os.path.join(possible_paths[0], default_filename)


def load_session_nix(session_name: str, include_raw: bool = False) -> neo.Block:
    """
    Load a neural data session from NIX format.
    
    Parameters
    ----------
    session_name : str
        Session identifier (e.g., 'i140703s001')
    include_raw : bool, optional
        Whether to include raw 30kHz data. Default: False
        
    Returns
    -------
    neo.Block
        Loaded neural data block
        
    Raises
    ------
    FileNotFoundError
        If the NIX file cannot be found
    RuntimeError
        If the NIX file cannot be loaded
    """
    nix_filename = get_nix_filename(session_name, include_raw=include_raw)
    
    if not os.path.exists(nix_filename):
        raise FileNotFoundError(f"NIX file not found: {nix_filename}")
    
    try:
        with neo.NixIO(nix_filename, mode='ro') as io:
            block = io.read_block()
        return block
    except Exception as e:
        raise RuntimeError(f"Failed to load NIX file {nix_filename}: {e}")


# ==========================================================================
# Event extraction utilities
# ==========================================================================

def get_events_by_trigger(segment: Segment, trigger_name: str, 
                         performance_code: Optional[int] = None, 
                         trial_type: Optional[str] = None) -> List[Event]:
    """
    Extract events for a specific trigger using neo.utils.
    
    Parameters
    ----------
    segment : neo.Segment
        Segment containing the events
    trigger_name : str
        Name of the trigger to extract (e.g., 'SR', 'TS-ON', 'CUE-ON')
    performance_code : int, optional
        Performance code filter
    trial_type : str, optional
        Trial type filter
        
    Returns
    -------
    List[neo.Event]
        List of events matching the criteria
        
    Raises
    ------
    RuntimeError
        If event extraction fails
    """
    try:
        # Build parameters for neo.utils.get_events
        kwargs = {'trial_event_labels': [trigger_name]}
        
        if performance_code is not None:
            kwargs['performance_in_trial'] = performance_code
            
        if trial_type is not None:
            kwargs['belongs_to_trialtype'] = trial_type
        
        # Use neo.utils for robust event extraction
        events = neo.utils.get_events(segment, **kwargs)
        
        if events:
            logger.info(f"Found {len(events)} events for trigger '{trigger_name}' "
                       f"with {len(events[0].times)} time points")
        else:
            logger.warning(f"No events found for trigger '{trigger_name}'")
        
        return events
        
    except Exception as e:
        error_msg = f"Event extraction failed for trigger '{trigger_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


# ==========================================================================
# Spike train analysis utilities
# ==========================================================================

def is_sua_spike_train(spike_train: SpikeTrain) -> bool:
    """
    Check if a spike train represents single unit activity (SUA).
    
    This function correctly handles numpy.bool_ annotations that are
    common in NIX files.
    
    Parameters
    ----------
    spike_train : neo.SpikeTrain
        Spike train to check
        
    Returns
    -------
    bool
        True if the spike train is marked as SUA
        
    Notes
    -----
    Uses == True instead of is True to correctly handle numpy.bool_ objects.
    """
    if not hasattr(spike_train, 'annotations') or not isinstance(spike_train.annotations, dict):
        return False
    
    sua_value = spike_train.annotations.get('sua')
    
    # Critical: Use == True instead of is True for numpy.bool_ compatibility
    return sua_value == True


def SNR_kelly(spiketrain: SpikeTrain) -> float:
    """
    Calculate signal-to-noise ratio using the Kelly et al (2007) method.
    
    The method computes:
    - Signal: peak-to-trough amplitude of the mean waveform
    - Noise: 2 × standard deviation of waveforms normalized by mean waveform
    
    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        Spike train with waveforms attribute
        
    Returns
    -------
    float
        SNR value, or 0.0 if no waveforms are available
    """
    if spiketrain.waveforms is None:
        return 0.0
        
    mean_waveform = spiketrain.waveforms.mean(axis=0)
    signal = mean_waveform.max() - mean_waveform.min()
    noise_std = (spiketrain.waveforms - mean_waveform).std()
    
    return signal / (2.0 * noise_std)


def st_id(spiketrain: SpikeTrain) -> int:
    """
    Generate a unique identifier for a spike train.
    
    The ID is computed as: channel_id * 100 + unit_id
    Example: channel 7, unit 1 → ID = 701
    
    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        Spike train with channel_id and unit_id annotations
        
    Returns
    -------
    int
        Unique spike train identifier
    """
    channel_id = spiketrain.annotations['channel_id']
    unit_id = spiketrain.annotations['unit_id']
    return channel_id * 100 + unit_id


# ==========================================================================
# Epoch creation and segment cutting utilities
# ==========================================================================

def add_epoch(segment: Segment, event1: Event, event2: Optional[Event] = None, 
              pre: pq.Quantity = 0 * pq.s, post: pq.Quantity = 0 * pq.s,
              attach_result: bool = True, **kwargs) -> Epoch:
    """
    Create epochs around events with specified time offsets.
    
    Parameters
    ----------
    segment : neo.Segment
        Segment to add the epoch to
    event1 : neo.Event
        Start events for the epochs
    event2 : neo.Event, optional
        End events for the epochs. If None, uses event1
    pre : quantities.Quantity, optional
        Time offset before event1. Default: 0 s
    post : quantities.Quantity, optional
        Time offset after event2. Default: 0 s
    attach_result : bool, optional
        Whether to attach the epoch to the segment. Default: True
    **kwargs
        Additional parameters for epoch creation
        
    Returns
    -------
    neo.Epoch
        Created epoch object
        
    Raises
    ------
    TypeError
        If inputs are not of correct types
    ValueError
        If events have incompatible lengths or would create invalid epochs
    """
    if event2 is None:
        event2 = event1

    # Input validation
    if not isinstance(segment, neo.Segment):
        raise TypeError(f'segment must be neo.Segment, got {type(segment)}')

    for event in [event1, event2]:
        if not isinstance(event, neo.Event):
            raise TypeError(f'events must be neo.Event, got {type(event)}')

    if len(event1) != len(event2):
        raise ValueError(f'event1 and event2 must have same length: '
                        f'{len(event1)} vs {len(event2)}')

    # Calculate epoch times and durations
    times = event1.times + pre
    durations = event2.times + post - times

    # Validate epoch durations
    if np.any(durations < 0):
        raise ValueError(f'Cannot create epochs with negative duration: {durations}')
    if np.any(durations == 0):
        raise ValueError('Cannot create epochs with zero duration')

    # Set default parameters
    if 'name' not in kwargs:
        kwargs['name'] = 'epoch'
    if 'labels' not in kwargs:
        kwargs['labels'] = [f"{kwargs['name']}_{i}" for i in range(len(times))]

    # Create epoch
    epoch = neo.Epoch(times=times, durations=durations, **kwargs)
    
    # Copy annotations from source event
    if hasattr(event1, 'annotations'):
        epoch.annotations.update(event1.annotations)

    if attach_result:
        segment.epochs.append(epoch)

    return epoch


def seg_time_slice(segment: Segment, t_start: pq.Quantity, t_stop: pq.Quantity, 
                   reset_time: bool = False) -> Segment:
    """
    Extract a time slice from a segment with proper annotation preservation.
    
    Parameters
    ----------
    segment : neo.Segment
        Source segment to slice
    t_start : quantities.Quantity
        Start time for the slice
    t_stop : quantities.Quantity
        Stop time for the slice
    reset_time : bool, optional
        Whether to reset time axis to start from 0. Default: False
        
    Returns
    -------
    neo.Segment
        New segment containing the time slice
        
    Raises
    ------
    TypeError
        If segment is not a neo.Segment
    ValueError
        If t_start >= t_stop
    """
    # Input validation
    if not isinstance(segment, neo.Segment):
        raise TypeError(f'segment must be neo.Segment, got {type(segment)}')
        
    if t_start >= t_stop:
        raise ValueError(f't_start ({t_start}) must be less than t_stop ({t_stop})')
    
    # Create new segment
    new_segment = neo.Segment(name=f"{segment.name}_slice")
    if hasattr(segment, 'annotations'):
        new_segment.annotations.update(segment.annotations)
    
    # Process each spike train
    for spike_train in segment.spiketrains:
        try:
            sliced_spike_train = _slice_spike_train(spike_train, t_start, t_stop, reset_time)
            new_segment.spiketrains.append(sliced_spike_train)
        except Exception as e:
            logger.warning(f"Failed to slice spike train: {e}")
            continue
    
    return new_segment


def _slice_spike_train(spike_train: SpikeTrain, t_start: pq.Quantity, 
                      t_stop: pq.Quantity, reset_time: bool) -> SpikeTrain:
    """
    Slice a single spike train while preserving all annotations.
    
    This is a helper function that ensures proper deep copying of annotations,
    which is critical for preserving SUA markings and other metadata.
    """
    # Find spikes within the time window
    mask = (spike_train >= t_start) & (spike_train < t_stop)
    
    if np.any(mask):
        spike_times = spike_train[mask]
    else:
        spike_times = [] * spike_train.units
    
    # Set time boundaries
    if reset_time:
        new_t_start = 0 * spike_train.units
        new_t_stop = (t_stop - t_start).rescale(spike_train.units)
        if len(spike_times) > 0:
            spike_times = spike_times - t_start
    else:
        new_t_start = t_start.rescale(spike_train.units)
        new_t_stop = t_stop.rescale(spike_train.units)
    
    # Create new spike train
    new_spike_train = neo.SpikeTrain(
        spike_times,
        t_start=new_t_start,
        t_stop=new_t_stop,
        units=spike_train.units
    )
    
    # CRITICAL: Deep copy all annotations to preserve numpy.bool_ and other types
    if hasattr(spike_train, 'annotations') and spike_train.annotations:
        new_spike_train.annotations = copy.deepcopy(spike_train.annotations)
    
    # Copy waveforms if present
    if (hasattr(spike_train, 'waveforms') and 
        spike_train.waveforms is not None and 
        np.any(mask)):
        new_spike_train.waveforms = spike_train.waveforms[mask]
        
    # Copy sampling rate
    if hasattr(spike_train, 'sampling_rate'):
        new_spike_train.sampling_rate = spike_train.sampling_rate
    
    return new_spike_train


def cut_segment_by_epoch(segment: Segment, epoch: Epoch, 
                        reset_time: bool = False) -> List[Segment]:
    """
    Cut a segment into multiple segments based on epoch boundaries.
    
    Parameters
    ----------
    segment : neo.Segment
        Source segment to cut
    epoch : neo.Epoch
        Epoch defining the cut boundaries
    reset_time : bool, optional
        Whether to reset time axis for each cut segment. Default: False
        
    Returns
    -------
    List[neo.Segment]
        List of cut segments, one per epoch
        
    Raises
    ------
    TypeError
        If inputs are not of correct types
    """
    # Input validation
    if not isinstance(segment, neo.Segment):
        raise TypeError(f'segment must be neo.Segment, got {type(segment)}')

    if not isinstance(epoch, neo.Epoch):
        raise TypeError(f'epoch must be neo.Epoch, got {type(epoch)}')

    segments = []
    
    for ep_id in range(len(epoch)):
        try:
            # Calculate time boundaries for this epoch
            t_start = epoch.times[ep_id]
            t_stop = epoch.times[ep_id] + epoch.durations[ep_id]
            
            # Create time slice
            subsegment = seg_time_slice(segment, t_start, t_stop, reset_time=reset_time)
            
            # Add epoch-specific annotations
            _add_epoch_annotations(subsegment, epoch, ep_id)
            
            segments.append(subsegment)
            
        except Exception as e:
            logger.warning(f"Failed to cut segment for epoch {ep_id}: {e}")
            continue

    return segments


def _add_epoch_annotations(segment: Segment, epoch: Epoch, epoch_id: int) -> None:
    """Add epoch-specific annotations to a segment."""
    # Copy epoch annotations
    if hasattr(epoch, 'annotations'):
        for key, value in epoch.annotations.items():
            if isinstance(value, list) and len(value) == len(epoch):
                segment.annotations[key] = copy.deepcopy(value[epoch_id])
            else:
                segment.annotations[key] = copy.deepcopy(value)
    
    # Add identification annotations
    segment.annotations['epoch_id'] = epoch_id
    if hasattr(epoch, 'labels') and epoch_id < len(epoch.labels):
        segment.annotations['epoch_label'] = epoch.labels[epoch_id]


# ==========================================================================
# Main data loader class
# ==========================================================================

class CachedDataLoader:
    """
    Main data loader for NIX files with caching and proper annotation handling.
    
    This class provides a high-level interface for loading neural data from
    NIX files, with support for epoch extraction, spike train filtering,
    and SNR-based quality assessment.
    
    Parameters
    ----------
    verbose : bool, optional
        Whether to print detailed progress information. Default: True
        
    Attributes
    ----------
    verbose : bool
        Verbosity flag
    logger : logging.Logger
        Logger for this instance
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = logging.getLogger(f"{__name__}.CachedDataLoader")
        
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
    
    @lru_cache(maxsize=4)
    def load_session_cached(self, session_name: str, include_raw: bool = False) -> neo.Block:
        """
        Load a session with caching to avoid repeated file I/O.
        
        Parameters
        ----------
        session_name : str
            Session identifier
        include_raw : bool, optional
            Whether to include raw data. Default: False
            
        Returns
        -------
        neo.Block
            Loaded neural data block
        """
        try:
            block = load_session_nix(session_name, include_raw)
            
            if self.verbose:
                segment = block.segments[0]
                sua_count = sum(1 for st in segment.spiketrains if is_sua_spike_train(st))
                print(f"✅ Session '{session_name}' loaded: "
                      f"{len(segment.spiketrains)} total spike trains, "
                      f"{sua_count} SUA units")
            
            return block
            
        except Exception as e:
            self.logger.error(f"Failed to load session {session_name}: {e}")
            raise
    
    def load_epoch_data_cached(self, session_name: str, epoch: Union[str, Tuple], 
                              trialtypes: Optional[str] = None, 
                              SNRthresh: float = 0.0, **kwargs) -> Dict[int, List[SpikeTrain]]:
        """
        Load spike train data for a specific epoch with caching and filtering.
        
        Parameters
        ----------
        session_name : str
            Session identifier (e.g., 'i140703s001')
        epoch : str or tuple
            Epoch specification. Can be:
            - 'start': Trial start period
            - 'cue1': First cue presentation
            - 'movement': Movement period
            - ('trigger', pre_ms, post_ms): Custom epoch definition
        trialtypes : str, optional
            Trial type filter (e.g., 'SGHF')
        SNRthresh : float, optional
            Minimum SNR threshold for spike trains. Default: 0.0
        **kwargs
            Additional parameters
            
        Returns
        -------
        Dict[int, List[neo.SpikeTrain]]
            Dictionary mapping unit IDs to lists of spike trains (one per trial)
            
        Example
        -------
        >>> loader = CachedDataLoader()
        >>> data = loader.load_epoch_data_cached('i140703s001', 'movement', SNRthresh=2.0)
        >>> print(f"Found {len(data)} units")
        >>> for unit_id, trials in data.items():
        ...     print(f"Unit {unit_id}: {len(trials)} trials")
        """
        # Parse epoch specification
        trigger, t_pre, t_post = self._parse_epoch(epoch)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"LOADING EPOCH DATA")
            print(f"{'='*60}")
            print(f"Session: {session_name}")
            print(f"Epoch: {epoch} → Trigger: {trigger}")
            print(f"Time window: {t_pre} to {t_post}")
            print(f"Trial type filter: {trialtypes}")
            print(f"SNR threshold: {SNRthresh}")
        
        # Load session data
        block = self.load_session_cached(session_name)
        segment = block.segments[0]
        
        # Extract events for the specified trigger
        events = get_events_by_trigger(segment, trigger, trial_type=trialtypes)
        
        if not events:
            print(f"❌ No events found for trigger '{trigger}'")
            return {}
        
        if self.verbose:
            print(f"✅ Found {len(events)} events with {len(events[0].times)} time points")
        
        # Create epoch around the events
        epoch_obj = add_epoch(segment, events[0], pre=t_pre, post=t_post,
                             attach_result=False, name=f'epoch_{trigger}')
        
        if self.verbose:
            print(f"✅ Created epoch with {len(epoch_obj.times)} time points")
        
        # Cut the segment into trials
        cut_segments = cut_segment_by_epoch(segment, epoch_obj, reset_time=True)
        
        if self.verbose:
            print(f"✅ Cut into {len(cut_segments)} trial segments")
            
            # Verify SUA preservation in first few segments
            if cut_segments:
                for i, seg in enumerate(cut_segments[:3]):
                    sua_count = sum(1 for st in seg.spiketrains if is_sua_spike_train(st))
                    print(f"   Trial {i}: {len(seg.spiketrains)} total spike trains, "
                          f"{sua_count} SUA units")
        
        # Extract and organize spike train data
        data = self._extract_spike_data(cut_segments, trialtypes, SNRthresh, 
                                       epoch, trigger, t_pre, t_post)
        
        if self.verbose:
            print(f"✅ FINAL RESULT: {len(data)} units extracted")
        
        return data
    
    def _parse_epoch(self, epoch: Union[str, Tuple]) -> Tuple[str, pq.Quantity, pq.Quantity]:
        """
        Parse epoch specification into trigger name and time offsets.
        
        Parameters
        ----------
        epoch : str or tuple
            Epoch specification
            
        Returns
        -------
        tuple
            (trigger_name, pre_time, post_time)
        """
        epoch_definitions = {
            'start': ('TS-ON', -250 * pq.ms, 250 * pq.ms),
            'cue1': ('CUE-ON', -250 * pq.ms, 250 * pq.ms),
            'earlydelay': ('CUE-OFF', 0 * pq.ms, 500 * pq.ms),
            'latedelay': ('GO-ON', -500 * pq.ms, 0 * pq.ms),
            'movement': ('SR', -200 * pq.ms, 300 * pq.ms),
            'hold': ('RW-ON', -500 * pq.ms, 0 * pq.ms),
        }
        
        if isinstance(epoch, str):
            if epoch not in epoch_definitions:
                raise ValueError(f"Unknown epoch '{epoch}'. "
                               f"Available: {list(epoch_definitions.keys())}")
            return epoch_definitions[epoch]
        
        elif isinstance(epoch, (tuple, list)) and len(epoch) == 3:
            return epoch
        
        else:
            raise ValueError(f"Invalid epoch specification: {epoch}. "
                           f"Must be a string or 3-element tuple.")
    
    def _extract_spike_data(self, segments: List[Segment], trialtypes: Optional[str], 
                           SNRthresh: float, epoch: Union[str, Tuple], 
                           trigger: str, t_pre: pq.Quantity, 
                           t_post: pq.Quantity) -> Dict[int, List[SpikeTrain]]:
        """
        Extract and organize spike train data from cut segments.
        
        This method processes each trial segment, filters for SUA spike trains,
        applies SNR thresholding, and organizes the data by unit ID.
        """
        data = {}
        stats = {
            'total_segments': len(segments),
            'total_spike_trains': 0,
            'sua_spike_trains': 0,
            'passed_snr': 0,
            'final_units': 0
        }
        
        if self.verbose:
            print(f"\nExtracting spike data from {len(segments)} trial segments...")
        
        for trial_id, segment in enumerate(segments):
            stats['total_spike_trains'] += len(segment.spiketrains)
            
            # Process each spike train in this trial
            for spike_train in segment.spiketrains:
                # Filter for SUA spike trains only
                if not is_sua_spike_train(spike_train):
                    continue
                
                stats['sua_spike_trains'] += 1
                
                # Calculate SNR if not already present
                if 'SNR' not in spike_train.annotations:
                    spike_train.annotations['SNR'] = SNR_kelly(spike_train)
                
                # Apply SNR threshold
                snr_value = spike_train.annotations['SNR']
                if snr_value <= SNRthresh:
                    continue
                
                stats['passed_snr'] += 1
                
                # Add trial metadata
                spike_train.annotations.update({
                    'trial_id': trial_id,
                    'trial_type': trialtypes,
                    'epoch': epoch,
                    'trigger': trigger,
                    't_pre': t_pre,
                    't_post': t_post
                })
                
                # Group by unit ID
                unit_id = st_id(spike_train)
                if unit_id not in data:
                    data[unit_id] = []
                data[unit_id].append(spike_train)
        
        stats['final_units'] = len(data)
        
        if self.verbose:
            print(f"  Extraction results:")
            print(f"    Total spike trains processed: {stats['total_spike_trains']:,}")
            print(f"    SUA spike trains found: {stats['sua_spike_trains']:,}")
            print(f"    Passed SNR threshold (>{SNRthresh}): {stats['passed_snr']:,}")
            print(f"    Final units extracted: {stats['final_units']}")
        
        return data


# ==========================================================================
# Convenience functions and testing
# ==========================================================================

def calc_spiketrains_SNR(session_name: str, units: str = 'all', 
                        include_raw: bool = False) -> Dict[int, float]:
    """
    Calculate SNR for all spike trains in a session.
    
    Parameters
    ----------
    session_name : str
        Session identifier
    units : {'all', 'sua', 'mua'}
        Which units to include in the calculation
    include_raw : bool, optional
        Whether to include raw data. Default: False
        
    Returns
    -------
    Dict[int, float]
        Dictionary mapping unit IDs to SNR values
    """
    block = load_session_nix(session_name, include_raw=include_raw)
    spike_trains = block.segments[0].spiketrains
    
    # Filter spike trains based on unit type
    if units == 'sua':
        spike_trains = [st for st in spike_trains if st.annotations.get('sua', False)]
    elif units == 'mua':
        spike_trains = [st for st in spike_trains if st.annotations.get('mua', False)]
    
    # Calculate SNR for each spike train
    snr_dict = {}
    for spike_train in spike_trains:
        unit_id = st_id(spike_train)
        snr_dict[unit_id] = SNR_kelly(spike_train)
    
    return snr_dict


def test_nix_loader(session_name: str = 'i140703s001') -> bool:
    """
    Test the NIX loader functionality with a sample session.
    
    Parameters
    ----------
    session_name : str, optional
        Session to test with. Default: 'i140703s001'
        
    Returns
    -------
    bool
        True if test passes, False otherwise
    """
    print("Testing NIX loader functionality...")
    print("=" * 60)
    
    try:
        # Initialize loader
        loader = CachedDataLoader(verbose=True)
        
        # Test basic session loading
        print("\n1. Testing session loading...")
        block = loader.load_session_cached(session_name)
        print(f"   ✅ Successfully loaded session with {len(block.segments)} segments")
        
        # Test SNR calculation
        print("\n2. Testing SNR calculation...")
        snr_dict = calc_spiketrains_SNR(session_name, units='sua')
        print(f"   ✅ Calculated SNR for {len(snr_dict)} SUA units")
        
        # Test epoch data extraction
        print("\n3. Testing epoch data extraction...")
        data = loader.load_epoch_data_cached(session_name, 'movement', SNRthresh=1.0)
        
        if data:
            print(f"   ✅ Successfully extracted {len(data)} units")
            
            # Show sample results
            sample_units = list(data.items())[:3]
            for unit_id, trials in sample_units:
                print(f"      Unit {unit_id}: {len(trials)} trials")
                if trials:
                    example_trial = trials[0]
                    snr = example_trial.annotations.get('SNR', 'N/A')
                    spike_count = len(example_trial)
                    print(f"        Example: {spike_count} spikes, SNR={snr:.2f}")
        else:
            print("   ❌ No units extracted")
            return False
        
        # Test different epochs
        print("\n4. Testing different epochs...")
        epochs_to_test = ['start', 'cue1', 'movement']
        
        for epoch in epochs_to_test:
            try:
                test_data = loader.load_epoch_data_cached(
                    session_name, epoch, SNRthresh=2.0
                )
                print(f"   ✅ {epoch}: {len(test_data)} units")
            except Exception as e:
                print(f"   ❌ {epoch}: Failed ({e})")
                return False
        
        print(f"\n{'='*60}")
        print("🎉 ALL TESTS PASSED!")
        print(f"{'='*60}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==========================================================================
# Example usage and documentation
# ==========================================================================

def example_usage():
    """
    Demonstrate typical usage patterns for the NIX loader.
    
    This function shows common workflows for loading and analyzing
    neural data from NIX files.
    """
    print("NIX Loader Usage Examples")
    print("=" * 40)
    
    # Initialize the loader
    loader = CachedDataLoader(verbose=True)
    
    # Example 1: Load movement epoch data
    print("\nExample 1: Load movement epoch data")
    print("-" * 40)
    
    movement_data = loader.load_epoch_data_cached(
        session_name='i140703s001',
        epoch='movement',
        SNRthresh=2.0
    )
    
    print(f"Found {len(movement_data)} high-quality units")
    
    # Example 2: Load different epochs for comparison
    print("\nExample 2: Compare different epochs")
    print("-" * 40)
    
    epochs = ['start', 'movement', 'hold']
    epoch_data = {}
    
    for epoch in epochs:
        data = loader.load_epoch_data_cached(
            session_name='i140703s001',
            epoch=epoch,
            SNRthresh=1.5
        )
        epoch_data[epoch] = data
        print(f"{epoch:>10}: {len(data)} units")
    
    # Example 3: Analyze spike counts across trials
    print("\nExample 3: Analyze spike patterns")
    print("-" * 40)
    
    for unit_id, trials in list(movement_data.items())[:3]:
        spike_counts = [len(trial) for trial in trials]
        mean_spikes = np.mean(spike_counts)
        std_spikes = np.std(spike_counts)
        
        print(f"Unit {unit_id}:")
        print(f"  Trials: {len(trials)}")
        print(f"  Mean spikes per trial: {mean_spikes:.1f} ± {std_spikes:.1f}")
        print(f"  SNR: {trials[0].annotations['SNR']:.2f}")
    
    # Example 4: Custom epoch definition
    print("\nExample 4: Custom epoch")
    print("-" * 40)
    
    custom_data = loader.load_epoch_data_cached(
        session_name='i140703s001',
        epoch=('SR', -100 * pq.ms, 200 * pq.ms),  # Custom time window
        SNRthresh=1.0
    )
    
    print(f"Custom epoch (SR -100 to +200 ms): {len(custom_data)} units")


# Usage example
def example_usage():
    """
    Example of how to use the cached data loader.
    """
    # Initialize the loader with both memory and disk caching
    loader = CachedDataLoader(
        cache_dir='./nix_cache',
        use_memory_cache=True,
        use_disk_cache=True
    )
    
    # Define what you want to load
    session_name = 'i140703-001'
    epochs = ['start', 'cue1', 'movement', 'hold']
    trialtypes = ['SGHF', 'SGLF', 'PGHF', 'PGLF']
    
    # Load all combinations efficiently
    print("First run (will cache everything):")
    results = loader.load_multiple_epochs_trialtypes(
        session_name=session_name,
        epochs=epochs,
        trialtypes=trialtypes,
        SNRthresh=2.0,
        verbose=True
    )
    
    print("\nSecond run (should be much faster due to caching):")
    results2 = loader.load_multiple_epochs_trialtypes(
        session_name=session_name,
        epochs=epochs,
        trialtypes=trialtypes,
        SNRthresh=2.0,
        verbose=True
    )
    
    # Check cache statistics
    loader.cache_info()
    
    # Access specific data
    movement_sghf_data = results['movement']['SGHF']
    print(f"\nMovement/SGHF has {len(movement_sghf_data)} SUA IDs")
    
    return results, loader


# Alternative: Simple function-based caching
@lru_cache(maxsize=8)
def load_session_simple_cache(session_name: str, include_raw: bool = False):
    """Simple cached session loader using functools.lru_cache."""
    return load_session_nix(session_name, include_raw)


@lru_cache(maxsize=32)
def get_cut_segments_simple_cache(session_name: str, trigger: str, 
                                 t_pre_ms: float, t_post_ms: float):
    """Simple cached cut segments function."""
    block = load_session_simple_cache(session_name)
    data_segment = block.segments[0]
    
    t_pre = t_pre_ms * pq.ms
    t_post = t_post_ms * pq.ms
    
    start_events = get_events(data_segment, properties={'trial_event_labels': trigger})
    if not start_events:
        raise ValueError(f"No events found for trigger '{trigger}'")
    
    start_event = start_events[0]
    epoch_obj = add_epoch(data_segment, event1=start_event, pre=t_pre, post=t_post, attach_result=False)
    return cut_segment_by_epoch(data_segment, epoch_obj, reset_time=True)


# ==========================================================================
# Synchrony detection and removal functions
# ==========================================================================


def get_trial_events(segment, trigger_name, performance_code=None):
    """
    Simplified function specifically for getting trial events from NIX format.
    
    Parameters
    ----------
    segment : neo.Segment
        The segment to search
    trigger_name : str
        The trigger name to look for (e.g., 'TS-ON', 'CUE-ON')
    performance_code : int, optional
        Performance code filter (e.g., 255 for correct trials)
        
    Returns
    -------
    events : list
        List of matching events
    """
    matching_events = []
    
    for event in segment.events:
        # Look for events that have the trigger in their labels
        if hasattr(event, 'labels') and trigger_name in event.labels:
            # If performance code specified, filter by it
            if performance_code is not None:
                if (hasattr(event, 'annotations') and 
                    isinstance(event.annotations, dict) and
                    'performance_in_trial' in event.annotations):
                    # Check if any of the performance codes match
                    perf_codes = event.annotations['performance_in_trial']
                    if hasattr(perf_codes, '__iter__'):
                        if performance_code not in perf_codes:
                            continue
                    elif perf_codes != performance_code:
                        continue
            
            matching_events.append(event)
    
    return matching_events





# ==========================================================================
# routines to remove synchrofacts
# ==========================================================================


def sts2gdf(sts, ids=[]):
    """
    Converts a list of spike trains to gdf format.

    Gdf is a 2-column data structure containing neuron ids on the first
    column and spike times (sorted in increasing order) on the second column.
    Information about the time unit, not preserved in the float-like gdf, is
    returned as a second output

    Arguments
    ---------
    sts : list
        a list of neo spike trains.
    ids : list, optional
        List of neuron IDs. Id[i] is the id associated to spike train sts[i].
        If empty list provided (default), ids are assigned as integers from 0
        to n_spiketrains-1.
        Default: []

    Returns
    -------
    gdf : ndarray of floats with shape (n_spikes, 2)]:
        ndarray of unit ids (first column) and spike times (second column)
    time_unit : Quantity
        the time unit of the spike times in gdf[:, 1]
    """
    # By default assign integers 0,1,... as ids of sts[0],sts[1],...
    if len(ids) == 0:
        ids = list(range(len(sts)))

    # Find smallest time unit
    time_unit = sts[0].units
    for st in sts[1:]:
        if st.units < time_unit:
            time_unit = st.units

    gdf = np.zeros((1, 2))
    # Rescale all spike trains to that time unit, extract the magnitude
    # and add to the gdf
    for st_idx, st in zip(ids, sts):
        to_be_added = np.array(
            [[st_idx] * len(st),
             st.view(pq.Quantity).rescale(time_unit).magnitude]).T
        gdf = np.vstack([gdf, to_be_added])

    # Eliminate first row in gdf and sort the others by increasing spike times
    gdf = gdf[1:]
    gdf = gdf[np.argsort(gdf[:, 1])]

    # Return gdf and time unit corresponding to second column
    return gdf, time_unit


def find_synchrofact_spikes(sts, n=2, dt=0 * pq.ms, ids=[]):
    """
    Given a list *sts* of spike trains, finds spike times composing
    synchronous events (up to a time lag dt) of size n or higher.

    Returns the times of the spikes composing such events, and the associated
    spike ids.

    Arguments
    ---------
    sts : list
        a list of neo SpikeTrains
    n : int
        minimum number of coincident spikes to report synchrony
    dt : Quantity, optional
        size of time lag for synchrony. Starting from the very first spike,
        a moving window of size dt slides through the spikes and captures
        events of size n or higher (greedy approach).
        If 0 (default), synchronous events are composed of spikes with the
        very same time only
        Default: 0*ms
    ids : list, optional
        List of neuron IDs. Id[i] is the id associated to spike train sts[i].
        If empty list provided (default), ids are assigned as the integers
        0, 1, ..., len(sts)-1
        Default: []

    Returns
    -------
    neur_ids : ndarray
        array of spike train ids composing the synchronous events, sorted
        by spike time
    times : Quantity
        a Quantity array of spike times for the spikes forming events of
        size >=n, in increasing order
    dt : Quantity
        the time width used to determine synchrony
    """

    gdf, time_unit = sts2gdf(sts, ids=ids)  # Convert sts list to sorted gdf
    dt_dimless = dt.rescale(time_unit).magnitude  # Make dt dimension-free
    if dt_dimless == 0:  # if dt_dimless is 0, set to half the min positive ISI
        dt_dimless = np.diff(np.unique(gdf[:, 1])).min() / 2.

    idx_synch = []
    time = gdf[0, 1]  # Set the init time for synchrony search to 1st spiketime
    # starting from the very first spike in the gdf
    idx_start, idx_stop = 0, 0
    # until end of gdf is reached,
    while idx_stop < gdf.shape[0] - 2:
        # Until the next spike falls in [time, time+dt)
        while time <= gdf[idx_stop + 1, 1] < time + dt_dimless:
            # Include that spike in the transaction
            idx_stop += 1
            # And stop if end of gdf reached
            if idx_stop + 1 >= gdf.shape[0]:
                break
        # If at least n spikes fall between idx_start and idx_stop
        if idx_stop >= idx_start + n - 1:
            idx_synch.extend(  # extend the range of indexes of synch spikes
                list(range(idx_start, idx_stop + 1)))
        idx_start += 1  # Set new idx_start to the next spike
        idx_stop = idx_start  # and idx_stop to idx_start
        time = gdf[idx_stop, 1]  # and set the new corresponding spike time
    idx_synch = np.array(np.unique(idx_synch), dtype=int)

    # Return transactions of >=n synchronous spikes, and the times of these
    # transactions (first spike time in each transaction)
    return (gdf[idx_synch][:, 0], gdf[idx_synch][:, 1] * time_unit,
            dt_dimless * time_unit)


def find_synchronous_events(sts, n, dt, ids=[]):
    """
    Given a list *sts* of spike trains, finds spike times composing
    synchronous events (up to a time lag dt) of size n or higher.

    Uses a greedy approach identical to CoCONAD (see fima.coconad()):
    starting from the first spike among all, as soon as m>=n consecutive
    spikes falling closer than a time dt are found, they are classified as
    a synchronous event and the next window is moved to the next spike.

    Differently from CoCONAD, this does not allow to count the support of
    each pattern. However, allows to reconstruct the occurrence times of
    spike patterns found as significant by the SPADE analysis.
    For instance, if the pattern (1, 2, 3) is found as significant, using
    >>> find_synchronous_events([st1, st2, st3], n=3, dt=3*ms, ids=[1,2,3])
    allows to retrieve the spike times composing each pattern's occurrence

    Returns the ids of the spikes composing such events, the event's ids
    and the associated spike times.

    Arguments
    ---------
    sts : list
        a list of neo SpikeTrain objects
    n : int
        minimum number of coincident spikes to report synchrony
    dt : Quantity
        size of time lag for synchrony. Starting from the very first spike,
        a moving window of size dt slides through the spikes and captures
        events of size n or higher (greedy approach).
    ids : list, optional
        list of neuron IDs. Id[i] is the id associated to spike train sts[i].
        If empty list provided (default), ids are assigned as the integers
        0, 1, ..., len(sts)-1.

    Returns
    -------
    neur_ids : ndarray
        array of spike train ids composing the synchronous events, sorted
        by spike time
    event_ids : ndarray
        an array of integers representing event ids. Spikes with the same
        event id form a synchronous event (of size >=n)
    times : Quantity
        a Quantity array of spike times for the spikes forming events of
        size >=n, in increasing order
    dt : Quantity
        the time width used to determine synchrony
    """

    gdf, time_unit = sts2gdf(sts, ids=ids)  # Convert sts list to sorted gdf
    dt_dimless = dt.rescale(time_unit).magnitude  # Make dt dimension-free
    if dt_dimless == 0:  # if dt_dimless is 0, set to half the min positive ISI
        dt_dimless = np.diff(np.unique(gdf[:, 1])).min() / 2.

    idx_synch = []
    time = gdf[0, 1]  # Set the init time for synchrony search to 1st spiketime
    event_ids = np.array([])
    # starting from the very first spike in the gdf
    idx_start, idx_stop, event_id = 0, 0, 0
    while idx_stop < gdf.shape[0] - 2:  # until end of gdf is reached,
        # Until the next spike falls in [time, time+dt)
        while time <= gdf[idx_stop + 1, 1] < time + dt_dimless:
            idx_stop += 1  # Include that spike in the transaction
            # And stop if end of gdf reached
            if idx_stop >= gdf.shape[0]:
                break
        # If at least n spikes fall between idx_start and idx_stop
        if idx_stop >= idx_start + n - 1:
            idx_synch.extend(  # extend the range of indexes of synch spikes
                list(range(idx_start, idx_stop + 1)))
            event_ids = np.hstack(  # and the indexes of synchronous events
                [event_ids, [event_id] * (idx_stop + 1 - idx_start)])
            event_id += 1  # and increase the event id
        # Set new idx_start to the next spike
        idx_start = max(idx_stop, idx_start + 1)
        idx_stop = idx_start  # and idx_stop to idx_start
        time = gdf[idx_stop, 1]  # and set the new corresponding spike time
    idx_synch = np.array(np.unique(idx_synch), dtype=int)
    return (gdf[idx_synch][:, 0], event_ids, gdf[idx_synch][:, 1] * time_unit,
            dt_dimless * time_unit)


def remove_synchrofact_spikes(sts, n=2, dt=0 * pq.ms, dt2=0 * pq.ms):
    """
    Given a list *sts* of spike trains, delete from them all spikes engaged
    in synchronous events of size *n* or higher. If specified, delete spikes
    close to such syncrhonous events as well.

    *Args*
    ------
    sts [list]:
        a list of SpikeTrains
    n [int]:
        minimum number of coincident spikes to report synchrony
    dt [Quantity. Default: 0 ms]:
        size of time lag for synchrony. Spikes closer than *dt* are
        considered synchronous. Groups of *n* or more synchronous spikes are
        deleted from the spike trains.
    dt2 [int. Default: 0 ms]:
        maximum distance for two spikes to be "close". Spikes "close" to
        synchronous spikes are eliminated as well.

    *Returns*
    ---------
    sts_new : list of SpikeTrains
        returns the SpikeTrains given in input, cleaned from spikes forming
        almost-synchronous events (time lag <= dt) of size >=n, and all
        spikes additionally falling within a time lag dt2 from such events.
    """
    # Find times of synchrony of size >=n
    spike_ids, times, dt = find_synchrofact_spikes(sts, n=n, dt=dt, ids=[])

    # delete unnecessary large object
    del spike_ids

    # Return "cleaned" spike trains
    if len(times) == 0:
        return sts

    sts2 = []  # initialize the list of new spike trains
    # and copy in it the original ones devoided of the synchrony times
    for st in sts:
        sts2.append(st.take(np.where(
            [np.abs(t - times).min() > dt2 for t in st])[0]))
    return sts2

# ==========================================================================
# Utility functions
# ==========================================================================


# Additional helper function for NIX format
def list_available_sessions(base_path=None):
    """
    List all available sessions in the NIX datasets directory.
    
    Parameters
    ----------
    base_path : str, optional
        Path to the datasets_nix directory. If None, uses default path.
        
    Returns
    -------
    sessions : dict
        Dictionary with session names as keys and available file types as values
    """
    if base_path is None:
        base_path = '../data/multielectrode_grasp/datasets_nix/'
    
    base_path = os.path.abspath(base_path)
    
    if not os.path.exists(base_path):
        print(f"Directory not found: {base_path}")
        return {}
    
    sessions = {}
    for filename in os.listdir(base_path):
        if filename.endswith('.nix'):
            if filename.endswith('_no_raw.nix'):
                session_name = filename.replace('_no_raw.nix', '')
                if session_name not in sessions:
                    sessions[session_name] = []
                sessions[session_name].append('no_raw')
            else:
                session_name = filename.replace('.nix', '')
                if session_name not in sessions:
                    sessions[session_name] = []
                sessions[session_name].append('raw')
    
    return sessions


# ==========================================================================
# Spike train utility functions
# ==========================================================================


def shift_spiketrain(spiketrain, t):
    """
    Shift the times of a SpikeTrain by an amount t.
    Shifts also t_start and t_stop by t.
    Retains the spike train's annotations, waveforms, sampling rate.
    """
    st = spiketrain
    st_shifted = neo.SpikeTrain(
        st.view(pq.Quantity) + t, t_start=st.t_start + t,
        t_stop=st.t_stop + t, waveforms=st.waveforms)
    st_shifted.sampling_period = st.sampling_period
    st_shifted.annotations = st.annotations

    return st_shifted


def calc_spiketrains_SNR(session_name, units='all', include_raw=False):
    """
    Calculates the signal-to-noise ratio (SNR) of each SpikeTrain in the
    specified session loaded from NIX format.

    Parameters:
    -----------
    session_name : str
        The name of a recording session. E.g: 'i140703-001' or 'l101210-001'
    units : str
        which type of units to consider:
        * 'all': returns the SNR values of all units in the session
        * 'sua': returns SUAs' SNR values only
        * 'mua': returns MUAs' SNR values only
    include_raw : bool, optional
        Whether to load raw data. Default: False

    Returns:
    --------
    SNRdict : dict
        a dictionary of unit ids and associated SNR values
    """
    block = load_session_nix(session_name, include_raw=include_raw)

    sts = [st for st in block.segments[0].spiketrains]
    if units == 'sua':
        sts = [st for st in sts if st.annotations.get('sua', False)]
    elif units == 'mua':
        sts = [st for st in sts if st.annotations.get('mua', False)]

    SNRdict = {}
    for st in sts:
        sua_id = st_id(st)
        SNRdict[sua_id] = SNR_kelly(st)

    return SNRdict

# ==========================================================================
# Example usage and demonstration functions  
# ==========================================================================

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


# ==========================================================================
# Example usage and demonstration functions  
# ==========================================================================

# if __name__ == "__main__":
#     # Run example
#     results, loader = example_usage()