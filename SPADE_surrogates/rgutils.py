# -*- coding: utf-8 -*-
"""
Utility functions for Reach-to-Grasp data using NIX format.
Updated to use the NIX files from datasets_nix directory instead of original Blackrock files.
"""
import os
import copy

import numpy as np
import quantities as pq
import neo


# ==========================================================================
# Neo utility functions for epoch processing
# ==========================================================================

def _get_from_list(obj_list, prop=None):
    """
    Helper function to filter objects from a list based on properties.
    """
    if prop is None:
        return obj_list
    
    output = []
    for obj in obj_list:
        valid = True
        for key, value in prop.items():
            obj_value = None
            
            # First check if it's an attribute
            if hasattr(obj, key):
                obj_value = getattr(obj, key)
            # Then check if it's in annotations
            elif hasattr(obj, 'annotations') and key in obj.annotations:
                obj_value = obj.annotations[key]
            else:
                valid = False
                break
            
            # Handle different types of value matching
            if isinstance(value, list):
                if isinstance(obj_value, list):
                    # Check if any values match
                    if not any(v in obj_value for v in value):
                        valid = False
                        break
                else:
                    # Check if single value is in the list
                    if obj_value not in value:
                        valid = False
                        break
            else:
                if isinstance(obj_value, list):
                    # Check if value is in the object's list
                    if value not in obj_value:
                        valid = False
                        break
                else:
                    # Direct comparison
                    if obj_value != value:
                        valid = False
                        break
        
        if valid:
            output.append(obj)
    
    return output


def get_events(container, properties=None):
    """
    This function returns a list of Neo Event objects, corresponding to given
    key-value pairs in the attributes or annotations of the Event.

    Parameter:
    ---------
    container: neo.Block or neo.Segment
        The Neo Block or Segment object to extract data from.
    properties: dictionary
        A dictionary that contains the Event keys and values to filter for.
        Each key of the dictionary is matched to a attribute or an an
        annotation of Event. The value of each dictionary entry corresponds to
        a valid entry or a list of valid entries of the attribute or
        annotation.

        If the value belonging to the key is a list of entries of the same
        length as the number of events in the Event object, the list entries
        are matched to the events in the Event object. The resulting Event
        object contains only those events where the values match up.

        Otherwise, the value is compared to the attributes or annotation of the
        Event object as such, and depending on the comparison, either the
        complete Event object is returned or not.

        If None or an empty dictionary is passed, all Event Objects will be
        returned in a list.

    Returns:
    --------
    events: list
        A list of Event objects matching the given criteria.
    """
    if isinstance(container, neo.Segment):
        return _get_from_list(container.events, prop=properties)

    elif isinstance(container, neo.Block):
        event_lst = []
        for seg in container.segments:
            event_lst += _get_from_list(seg.events, prop=properties)
        return event_lst
    else:
        raise TypeError(
            'Container needs to be of type neo.Block or neo.Segment, not %s '
            'in order to extract Events.' % (type(container)))


def add_epoch(
        segment, event1, event2=None, pre=0 * pq.s, post=0 * pq.s,
        attach_result=True, **kwargs):
    """
    Create epochs around a single event, or between pairs of events. Starting
    and end time of the epoch can be modified using pre and post as offsets
    before the and after the event(s). Additional keywords will be directly
    forwarded to the epoch intialization.

    Parameters:
    -----------
    segment : neo.Segment
        The segment in which the final Epoch object is added.
    event1 : neo.Event
        The Neo Event objects containing the start events of the epochs. If no
        event2 is specified, these event1 also specifies the stop events, i.e.,
        the epoch is cut around event1 times.
    event2: neo.Event
        The Neo Event objects containing the stop events of the epochs. If no
        event2 is specified, event1 specifies the stop events, i.e., the epoch
        is cut around event1 times. The number of events in event2 must match
        that of event1.
    pre, post: Quantity (time)
        Time offsets to modify the start (pre) and end (post) of the resulting
        epoch. Example: pre=-10*ms and post=+25*ms will cut from 10 ms before
        event1 times to 25 ms after event2 times
    attach_result: bool
        If True, the resulting Neo Epoch object is added to segment.

    Keyword Arguments:
    ------------------
    Passed to the Neo Epoch object.

    Returns:
    --------
    epoch: neo.Epoch
        An Epoch object with the calculated epochs (one per entry in event1).
    """
    if event2 is None:
        event2 = event1

    if not isinstance(segment, neo.Segment):
        raise TypeError(
            'Segment has to be of type neo.Segment, not %s' % type(segment))

    for event in [event1, event2]:
        if not isinstance(event, neo.Event):
            raise TypeError(
                'Events have to be of type neo.Event, not %s' % type(event))

    if len(event1) != len(event2):
        raise ValueError(
            'event1 and event2 have to have the same number of entries in '
            'order to create epochs between pairs of entries. Match your '
            'events before generating epochs. Current event lengths '
            'are %i and %i' % (len(event1), len(event2)))

    times = event1.times + pre
    durations = event2.times + post - times

    if any(durations < 0):
        raise ValueError(
            'Can not create epoch with negative duration. '
            'Requested durations %s.' % durations)
    elif any(durations == 0):
        raise ValueError('Can not create epoch with zero duration.')

    if 'name' not in kwargs:
        kwargs['name'] = 'epoch'
    if 'labels' not in kwargs:
        kwargs['labels'] = [
            '%s_%i' % (kwargs['name'], i) for i in range(len(times))]

    ep = neo.Epoch(times=times, durations=durations, **kwargs)

    ep.annotations.update(event1.annotations)

    if attach_result:
        segment.epochs.append(ep)
        # Note: create_relationship() might not exist in newer Neo versions
        if hasattr(segment, 'create_relationship'):
            segment.create_relationship()

    return ep


def seg_time_slice(segment, t_start, t_stop, reset_time=False):
    """
    Helper function to slice a segment in time.
    This replaces the original seg_time_slice function from neo_utils.
    """
    # Create new segment
    new_seg = neo.Segment(name=segment.name + '_slice')
    new_seg.annotations.update(segment.annotations)
    
    # Slice spike trains
    for st in segment.spiketrains:
        # Find spikes within time window
        mask = (st >= t_start) & (st < t_stop)
        
        if np.any(mask):
            # Extract spikes in window
            spike_times = st[mask]
            
            # Reset time if requested
            if reset_time:
                spike_times = spike_times - t_start
                new_t_start = 0 * st.units
                new_t_stop = (t_stop - t_start).rescale(st.units)
            else:
                new_t_start = t_start.rescale(st.units)
                new_t_stop = t_stop.rescale(st.units)
        else:
            # No spikes in window
            spike_times = [] * st.units
            if reset_time:
                new_t_start = 0 * st.units
                new_t_stop = (t_stop - t_start).rescale(st.units)
            else:
                new_t_start = t_start.rescale(st.units)
                new_t_stop = t_stop.rescale(st.units)
        
        # Create new spike train
        new_st = neo.SpikeTrain(
            spike_times,
            t_start=new_t_start,
            t_stop=new_t_stop,
            units=st.units
        )
        
        # Copy annotations and waveforms
        new_st.annotations.update(st.annotations)
        if hasattr(st, 'waveforms') and st.waveforms is not None:
            if np.any(mask):
                new_st.waveforms = st.waveforms[mask]
            new_st.sampling_rate = getattr(st, 'sampling_rate', None)
        
        new_seg.spiketrains.append(new_st)
    
    # Slice analog signals if present
    for asig in segment.analogsignals:
        # Calculate sample indices
        start_idx = int((t_start - asig.t_start) * asig.sampling_rate)
        stop_idx = int((t_stop - asig.t_start) * asig.sampling_rate)
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        stop_idx = min(len(asig), stop_idx)
        
        if start_idx < stop_idx:
            # Extract signal slice
            new_signal = asig[start_idx:stop_idx]
            
            # Reset time if requested
            if reset_time:
                new_signal.t_start = 0 * asig.units
            else:
                new_signal.t_start = t_start
            
            new_signal.annotations.update(asig.annotations)
            new_seg.analogsignals.append(new_signal)
    
    return new_seg


def cut_segment_by_epoch(seg, epoch, reset_time=False):
    """
    Cuts a Neo Segment according to a neo Epoch object

    The function returns a list of neo Segments, where each segment corresponds
    to an epoch in the neo Epoch object and contains the data of the original
    Segment cut to that particular Epoch.

    The resulting segments may either retain their original time stamps,
    or can be shifted to a common time axis.

    Parameters
    ----------
    seg: Neo Segment
        The Segment containing the original uncut data.
    epoch: Neo Epoch
        For each epoch in this input, one segment is generated according to
         the epoch time and duration.
    reset_time: bool
        If True the times stamps of all sliced objects are set to fall
        in the range from 0 to the duration of the epoch duration.
        If False, original time stamps are retained.
        Default is False.

    Returns:
    --------
    segments: list of Neo Segments
        Per epoch in the input, a neo.Segment with AnalogSignal and/or
        SpikeTrain Objects will be generated and returned. Each Segment will
        receive the annotations of the corresponding epoch in the input.
    """
    
    if not isinstance(seg, neo.Segment):
        raise TypeError(
            'Seg needs to be of type neo.Segment, not %s' % type(seg))

    # Check if segment has a parent block (may not be required in newer Neo)
    if hasattr(seg, 'parents') and seg.parents and type(seg.parents[0]) != neo.Block:
        raise ValueError(
            'Segment has no block as parent. Can not cut segment.')

    if not isinstance(epoch, neo.Epoch):
        raise TypeError(
            'Epoch needs to be of type neo.Epoch, not %s' % type(epoch))

    segments = []
    for ep_id in range(len(epoch)):
        subseg = seg_time_slice(seg,
                                epoch.times[ep_id],
                                epoch.times[ep_id] + epoch.durations[ep_id],
                                reset_time=reset_time)
        
        # Add annotations of Epoch
        for a in epoch.annotations:
            if type(epoch.annotations[a]) is list \
                    and len(epoch.annotations[a]) == len(epoch):
                subseg.annotations[a] = copy.copy(epoch.annotations[a][ep_id])
            else:
                subseg.annotations[a] = copy.copy(epoch.annotations[a])
        
        # Add additional annotations for trial identification
        subseg.annotations['epoch_id'] = ep_id
        if hasattr(epoch, 'labels') and ep_id < len(epoch.labels):
            subseg.annotations['epoch_label'] = epoch.labels[ep_id]
        
        segments.append(subseg)

    return segments


# ==========================================================================
# NIX file loading and path functions
# ==========================================================================


def data_path(session):
    """
    Finds the path associated to a given session for NIX files.

    Parameters
    ----------
    session : str
        The name of a recording session. E.g: 'i140703-001' or 'l101210-001'

    Returns
    -------
    path : str
        The path to the datasets_nix directory
    """
    path = '../data/multielectrode_grasp/datasets_nix/'
    path = os.path.abspath(path) + '/'
    return path


def get_nix_filename(session_name, include_raw=False):
    """
    Get the appropriate NIX filename for a session.
    
    Parameters
    ----------
    session_name : str
        The name of a recording session. E.g: 'i140703-001' or 'l101210-001'
    include_raw : bool, optional
        Whether to include raw 30kHz data. If False, uses the lighter '_no_raw' version.
        Default: False
        
    Returns
    -------
    filename : str
        The full path to the NIX file
    """
    base_path = data_path(session_name)
    
    if include_raw:
        filename = f"{session_name}.nix"
    else:
        filename = f"{session_name}_no_raw.nix"
    
    return os.path.join(base_path, filename)


def load_session_nix(session_name, include_raw=False):
    """
    Load a session from NIX format files.
    
    Parameters
    ----------
    session_name : str
        The name of a recording session. E.g: 'i140703-001' or 'l101210-001'
    include_raw : bool, optional
        Whether to include raw 30kHz data. If False, uses the lighter '_no_raw' version.
        Default: False
        
    Returns
    -------
    block : neo.Block
        The loaded data block with full annotations
    """
    nix_filename = get_nix_filename(session_name, include_raw=include_raw)
    
    if not os.path.exists(nix_filename):
        raise FileNotFoundError(f"NIX file not found: {nix_filename}")
    
    with neo.NixIO(nix_filename, mode='ro') as io:
        block = io.read_block()
    
    return block


# ==========================================================================
# Spike train utility functions
# ==========================================================================


def st_id(spiketrain):
    """
    Associates to a SpikeTrain an unique ID, given by the float
    100 * electrode_id + unit_id.
    E.g.: electrode_id = 7, unit_id = 1 -> st_id = 701
    """
    return spiketrain.annotations['channel_id'] * 100 + 1 * \
        spiketrain.annotations['unit_id']


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


def SNR_kelly(spiketrain):
    """
    Returns the SNR of the waveforms of spiketrains, as computed in
    Kelly et al (2007):
    * compute the mean waveform
    * define the signal as the peak-to-trough of such mean waveform
    * define the noise as double the std.dev. of the values of all waveforms,
      each normalised by subtracting the mean waveform

    Parameters:
    -----------
    spiketrain : SpikeTrain
        spike train with waveforms attribute

    Returns:
    --------
    snr: float
        The SNR of the input spike train
    """
    if spiketrain.waveforms is None:
        return 0.0
        
    mean_waveform = spiketrain.waveforms.mean(axis=0)
    signal = mean_waveform.max() - mean_waveform.min()
    SD = (spiketrain.waveforms - mean_waveform).std()
    return signal / (2. * SD)


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
# Main loading routines for NIX format
# ==========================================================================


def load_session_sts(
        session_name, units='sua', SNRthresh=0, synchsize=0, dt=None,
        dt2=None, include_raw=False, verbose=False):
    """
    Load SUA spike trains of a full specific session from NIX files.

    Parameters:
    -----------
    session_name : str
        The name of a recording session. E.g: 'i140703-001' or 'l101210-001'
    units : str, optional
        Type of units to load: 'sua', 'mua', or 'all'. Default: 'sua'
    SNRthresh: float, optional
        Lower threshold for the waveforms' SNR of SUAs to be considered.
        Default: 0
    synchsize: int, optional
        Minimum size of synchronous events to be removed from the data.
        If 0, no synchronous events are removed. Default: 0
    dt: Quantity, optional
        Time lag for synchrony detection. Default: None
    dt2: Quantity, optional
        Time lag for removing spikes near synchronous events. Default: None
    include_raw : bool, optional
        Whether to load raw data. Default: False
    verbose : bool, optional
        Whether to print progress information. Default: False

    Returns:
    --------
    sts : list of SpikeTrain
        A list of all SpikeTrains in the session
    """
    if verbose:
        print(f'Load data (session: {session_name}) from NIX format...')

    block = load_session_nix(session_name, include_raw=include_raw)
    sts = list(block.segments[0].spiketrains)

    if units == 'sua':
        sts = [st for st in sts if st.annotations.get('sua', False)]
    elif units == 'mua':
        sts = [st for st in sts if st.annotations.get('mua', False)]

    # Remove synchronous events if requested
    if not (synchsize == 0 or synchsize is None):
        time_unit = sts[0].units
        Dt = time_unit if dt is None else dt
        Dt2 = time_unit if dt2 is None else dt2
        if verbose:
            print(f'  > remove synchrofacts (precision={Dt})'
                  f' of size {synchsize:d}')
            print(f'    and their neighbours at distance <= {Dt2}...')
            print(
                f'    (# synch. spikes before removal: '
                f'{len(find_synchrofact_spikes(sts, n=synchsize, dt=Dt)[1]):d}'
                f')')

        sts_new = remove_synchrofact_spikes(sts, n=synchsize, dt=Dt, dt2=Dt2)

        for i in range(len(sts)):
            sts_new[i].annotations = sts[i].annotations
        sts = sts_new[:]

    # Remove SpikeTrains with low SNR
    if SNRthresh > 0:
        if verbose:
            print('  > remove low-SNR SpikeTrains...')
        # Calculate SNR for spike trains that don't have it
        for st in sts:
            if 'SNR' not in st.annotations:
                st.annotations['SNR'] = SNR_kelly(st)
        sts = [st for st in sts if st.annotations['SNR'] > SNRthresh]

    return sts


# ==========================================================================
# Synchrony detection and removal functions
# ==========================================================================


def load_epoch_as_lists(session_name, epoch, trialtypes=None, SNRthresh=0,
                        include_raw=False, verbose=False):
    """
    Load SUA spike trains of specific session and epoch from NIX files.

    The epoch definitions remain the same as in the original code:
    * epoch='start'     :  trigger='TS-ON'   t_pre=250 ms   t_post=250 ms
    * epoch='cue1'      :  trigger='CUE-ON'  t_pre=250 ms   t_post=250 ms
    * epoch='earlydelay':  trigger='CUE-OFF' t_pre=0 ms     t_post=500 ms
    * epoch='latedelay' :  trigger='GO-ON'   t_pre=500 ms   t_post=0 ms
    * epoch='movement'  :  trigger='SR'      t_pre=200 ms   t_post=300 ms
    * epoch='hold'      :  trigger='RW-ON'   t_pre=500 ms   t_post=0 ms

    Parameters:
    -----------
    session_name : str
        The name of a recording session. E.g: 'i140703-001' or 'l101210-001'
    epoch : str or tuple
        Epoch specification (same as original)
    trialtypes : str, optional
        Trial type to consider. Default: None
    SNRthresh : float, optional
        SNR threshold. Default: 0
    include_raw : bool, optional
        Whether to load raw data. Default: False
    verbose : bool, optional
        Whether to print progress information. Default: False

    Returns:
    --------
    data : dict
        Dictionary with SUA IDs as keys and lists of SpikeTrains as values
    """
    # Define trigger, t_pre, t_post depending on epoch
    if epoch == 'start':
        trigger, t_pre, t_post = 'TS-ON', -250 * pq.ms, 250 * pq.ms
    elif epoch == 'cue1':
        trigger, t_pre, t_post = 'CUE-ON', -250 * pq.ms, 250 * pq.ms
    elif epoch == 'earlydelay':
        trigger, t_pre, t_post = 'CUE-OFF', -0 * pq.ms, 500 * pq.ms
    elif epoch == 'latedelay':
        trigger, t_pre, t_post = 'GO-ON', -500 * pq.ms, 0 * pq.ms
    elif epoch == 'movement':
        trigger, t_pre, t_post = 'SR', -200 * pq.ms, 300 * pq.ms
    elif epoch == 'hold':
        trigger, t_pre, t_post = 'RW-ON', -500 * pq.ms, 0 * pq.ms
    elif isinstance(epoch, str):
        raise ValueError("epoch '%s' not defined" % epoch)
    elif len(epoch) == 3:
        trigger, t_pre, t_post = epoch
    else:
        raise ValueError('epoch must be either a string or a tuple of len 3')

    if verbose:
        print(f'Load data (session: {session_name}, epoch: {epoch}, '
              f'trialtype: {trialtypes}) from NIX format...')
        print(f"  > load session {session_name}, and define Block around "
              f"trigger '{trigger}'...")

    # Load session from NIX file
    block = load_session_nix(session_name, include_raw=include_raw)
    data_segment = block.segments[0]

    # Get performance codes - try to find them in block annotations or use defaults
    if hasattr(block, 'annotations') and 'performance_codes' in block.annotations:
        performance_codes = block.annotations['performance_codes']
        correct_trial_code = performance_codes.get('correct_trial', 255)
    else:
        # Default performance code for correct trials
        correct_trial_code = 255

    # Find events matching the trigger and correct performance
    start_events = get_events(
        data_segment,
        properties={
            'trial_event_labels': trigger,
            'performance_in_trial': correct_trial_code
        })
    
    if not start_events:
        # Try without performance filter
        start_events = get_events(
            data_segment,
            properties={'trial_event_labels': trigger})
    
    if not start_events:
        raise ValueError(f"No events found for trigger '{trigger}'")
    
    start_event = start_events[0]

    # Create epoch around the trigger
    epoch_obj = add_epoch(
        data_segment,
        event1=start_event, 
        event2=None,
        pre=t_pre, 
        post=t_post,
        attach_result=False,
        name='{}'.format(epoch))
    
    # Cut the segment by epoch
    cut_trial_block = neo.Block(name="Cut_Trials")
    cut_trial_block.segments = cut_segment_by_epoch(
        data_segment, epoch_obj, reset_time=True)
    
    # Filter segments by trial type if specified
    if trialtypes is not None:
        selected_trial_segments = []
        for seg in cut_trial_block.segments:
            if hasattr(seg, 'annotations') and seg.annotations.get('belongs_to_trialtype') == trialtypes:
                selected_trial_segments.append(seg)
    else:
        selected_trial_segments = cut_trial_block.segments
    
    # Extract spike trains and organize by SUA ID
    data = {}
    for seg_id, seg in enumerate(selected_trial_segments):
        # Filter for SUA spike trains
        sua_spiketrains = [st for st in seg.spiketrains 
                          if st.annotations.get('sua', False)]
        
        for st in sua_spiketrains:
            # Check SNR
            if 'SNR' not in st.annotations:
                st.annotations['SNR'] = SNR_kelly(st)
            
            if st.annotations['SNR'] > SNRthresh:
                # Add trial information to annotations
                st.annotations['trial_id'] = seg.annotations.get('trial_id', seg_id)
                st.annotations['trial_type'] = seg.annotations.get('belongs_to_trialtype', trialtypes)
                st.annotations['trial_id_trialtype'] = seg_id
                st.annotations['epoch'] = epoch
                st.annotations['trigger'] = trigger
                st.annotations['t_pre'] = t_pre
                st.annotations['t_post'] = t_post
                
                # Get SUA ID
                sua_id = st_id(st)
                
                # Add to data dictionary
                if sua_id not in data:
                    data[sua_id] = []
                data[sua_id].append(st)
    
    return data


def load_epoch_concatenated_trials(
        session_name, epoch, trialtypes=None, SNRthresh=0, synchsize=0, dt=None,
        dt2=None, sep=100 * pq.ms, include_raw=False, verbose=False):
    """
    Load a slice of spike train data in a specified epoch from NIX files,
    select spike trains corresponding to specific trialtypes only, and concatenate them.

    The epoch is either one of 6 specific epochs defined, following a
    discussion with Sonja, Alexa, Thomas, as 500 ms-long time segments
    each around a specific trigger, or a triplet consisting of a trigger
    and two time spans delimiting a time segment around the trigger.

    The pre-defined epochs, the associated triggers, and the time spans
    t_pre and t_post (before and after the trigger, respectively) are:
    * epoch='start'     :  trigger='TS-ON'   t_pre=250 ms   t_post=250 ms
    * epoch='cue1'      :  trigger='CUE-ON'  t_pre=250 ms   t_post=250 ms
    * epoch='earlydelay':  trigger='CUE-OFF' t_pre=0 ms     t_post=500 ms
    * epoch='latedelay' :  trigger='GO-ON'   t_pre=500 ms   t_post=0 ms
    * epoch='movement'  :  trigger='SR'      t_pre=200 ms   t_post=300 ms
    * epoch='hold'      :  trigger='RW-ON'   t_pre=500 ms   t_post=0 ms

    Parameters:
    -----------
    session_name : str
        The name of a recording session. E.g: 'i140703-001' or 'l101210-001'
    epoch : str or triplet
        if str, defines a trigger and a time segment around it (see above).
        if a triplet (tuple with 3 elements), its elements are, in order:
        * trigger [str] : a trigger (any string in session.trial_events)
        * t_pre [Quantity] : the left end of the time segment around the
          trigger. (> 0 for times before the trigger, < 0 for time after it)
        * t_post [Quantity] : the right end of the time segment around the
          trigger. (> 0 for times after the trigger, < 0 for time before it)
    trialtypes : str | list of str | None, optional
        One or more trial types, among those present in the session.
        8 Classical trial types for Lilou's sessions are:
        'SGHF', 'SGLF', 'PGHF', PGLF', 'HFSG', 'LFSG', 'HFPG', 'LFPG'.
        trialtypes can be one of such strings, of a list of them, or None.
        If None, all trial types in the session are considered.
        Default: None
    SNRthresh : float, optional
        lower threshold for the waveforms' SNR of SUAs to be considered.
        SUAs with a lower or equal SNR are not loaded.
        Default: 0
    synchsize : int, optional
        minimum size of synchronous events to be removed from the data.
        If 0, no synchronous events are removed.
        Synchrony is defined by the parameter dt.
    dt : Quantity, optional
        time lag within which synchronous spikes are considered highly
        synchronous ("synchrofacts"). If None, the sampling period of the
        recording system (1 * session.nev_unit) is used.
        Default: None
    dt2 : Quantity, optional
        isolated spikes falling within a time lag dt2 from synchrofacts (see
        parameter dt) to be removed (see parameter synchsize) are also
        removed. If None, the sampling period of the recording system
        (1 * session.nev_unit) is used.
        Default: None
    sep : Quantity
        Time interval used to separate consecutive trials.
        Default: 100 ms
    include_raw : bool, optional
        Whether to load raw data from NIX files. Default: False
    verbose : bool
        Whether to print information as different steps are run
    
    Returns:
    --------
    data : list
        a list of SpikeTrains, each obtained by concatenating all trials
        of the desired type(s) and during the specified epoch for that SUA.
    """
    # Load the data as a dictionary of SUA_id: [list of trials]
    data = load_epoch_as_lists(session_name, epoch, trialtypes=trialtypes,
                               SNRthresh=SNRthresh, include_raw=include_raw,
                               verbose=verbose)

    if not data:
        if verbose:
            print("No data found matching the criteria.")
        return []

    # Check that all spike trains in all lists have same t_start, t_stop
    sample_st_list = list(data.values())[0]
    if not sample_st_list:
        if verbose:
            print("No spike trains found in data.")
        return []
    
    sample_st = sample_st_list[0]
    t_pre = abs(sample_st.t_start)
    t_post = abs(sample_st.t_stop)
    
    # Validate that all spike trains have consistent timing
    for sua_id, st_list in data.items():
        for st in st_list:
            if abs(abs(st.t_start) - t_pre) > 1e-10 * st.t_start.units:
                raise ValueError(
                    f'SpikeTrains have inconsistent t_start; cannot be concatenated. '
                    f'Expected {t_pre}, got {abs(st.t_start)} for SUA {sua_id}')
            if abs(abs(st.t_stop) - t_post) > 1e-10 * st.t_stop.units:
                raise ValueError(
                    f'SpikeTrains have inconsistent t_stop; cannot be concatenated. '
                    f'Expected {t_post}, got {abs(st.t_stop)} for SUA {sua_id}')

    # Define time unit, trial duration, trial IDs to consider
    time_unit = sample_st.units
    trial_duration = (t_post + t_pre + sep).rescale(time_unit)
    
    # Get all unique trial IDs from the data
    all_trial_ids = set()
    for st_list in data.values():
        for st in st_list:
            all_trial_ids.add(st.annotations['trial_id_trialtype'])
    
    trial_ids_of_chosen_types = sorted(list(all_trial_ids))

    # Concatenate the lists of spike trains into a single SpikeTrain
    if verbose:
        print('  > concatenate trials...')
    
    conc_data = []
    for sua_id in sorted(data.keys()):
        trials_to_concatenate = []
        original_times = []
        
        # Create list of trials, each shifted by trial_duration*trial_id
        for tr in data[sua_id]:
            trial_offset = (trial_duration * tr.annotations['trial_id_trialtype']).rescale(time_unit)
            shifted_times = tr.rescale(time_unit).magnitude + trial_offset.magnitude
            trials_to_concatenate.extend(shifted_times)
            original_times.extend(list(tr.magnitude))
        
        # Sort the concatenated spike times
        if len(trials_to_concatenate) > 0:
            trials_to_concatenate = np.sort(trials_to_concatenate)

        # Re-transform the concatenated spikes into a SpikeTrain
        total_duration = trial_duration * (max(trial_ids_of_chosen_types) + 1)
        st = neo.SpikeTrain(
            trials_to_concatenate * time_unit,
            t_start=0 * time_unit,
            t_stop=total_duration.rescale(time_unit)
        )

        # Copy annotations from the original spike trains
        if data[sua_id]:
            reference_st = data[sua_id][0]
            for key, value in reference_st.annotations.items():
                if key not in ['trial_id', 'trial_id_trialtype']:
                    st.annotations[key] = value
        
        # Add concatenation-specific annotations
        st.annotations['original_times'] = original_times
        st.annotations['trial_separation'] = sep
        st.annotations['concatenated'] = True
        st.annotations['n_trials'] = len(data[sua_id])
        
        conc_data.append(st)
    
    # Remove exactly synchronous spikes from data if requested
    if not (synchsize == 0 or synchsize is None):
        Dt = time_unit if dt is None else dt
        Dt2 = time_unit if dt2 is None else dt2
        if verbose:
            try:
                sync_spikes_before_removal = len(
                    find_synchrofact_spikes(conc_data, n=synchsize, dt=Dt)[1])
                print(f'  > remove synchrofacts (precision={Dt}) of'
                      f' size {synchsize:d}')
                print(f'    and their neighbours at distance <= {Dt2}...')
                print(
                    f'    (# synch. spikes before removal: '
                    f'{sync_spikes_before_removal:d})')
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not count synchronous spikes: {e}")

        sts = remove_synchrofact_spikes(conc_data, n=synchsize, dt=Dt, dt2=Dt2)
        
        # Restore annotations
        for i, conc_st in enumerate(conc_data):
            if i < len(sts):
                sts[i].annotations.update(conc_st.annotations)
    else:
        sts = conc_data
    
    # Return the list of SpikeTrains
    return sts


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
# Example usage and demonstration functions  
# ==========================================================================


# Example usage function
def example_usage():
    """
    Example of how to use the updated NIX-based loader with load_epoch_concatenated_trials.
    """
    # List available sessions
    sessions = list_available_sessions()
    print("Available sessions:", sessions)
    
    # Load a session (use 'i140703-001' or 'l101210-001')
    session_name = 'i140703-001'  # or 'l101210-001'
    
    # Example parameters
    epoch = 'movement'  # or 'start', 'cue1', 'earlydelay', 'latedelay', 'hold'
    trialtype = 'SGHF'  # or any other trial type, or None for all
    snr_threshold = 2.0
    synchrony_size = 3
    trial_separation = 100 * pq.ms
    
    # Load concatenated trials using the main function
    print(f"\nLoading concatenated trials for session {session_name}...")
    sts = load_epoch_concatenated_trials(
        session_name=session_name,
        epoch=epoch,
        trialtypes=trialtype,
        SNRthresh=snr_threshold,
        synchsize=synchrony_size,
        sep=trial_separation,
        include_raw=False,  # Use lighter files without raw data
        verbose=True
    )
    
    print(f"Loaded {len(sts)} concatenated spike trains")
    
    if sts:
        # Show some statistics
        print(f"First spike train duration: {sts[0].t_stop}")
        print(f"First spike train has {len(sts[0])} spikes")
        print(f"Annotations: {list(sts[0].annotations.keys())}")
    
    return sts


# Function to demonstrate different epoch loading options
def demonstrate_epoch_loading():
    """
    Demonstrate loading different epochs and trial types.
    """
    session_name = 'i140703-001'
    
    # Test different epochs
    epochs = ['start', 'cue1', 'earlydelay', 'latedelay', 'movement', 'hold']
    
    for epoch in epochs:
        try:
            print(f"\nTesting epoch: {epoch}")
            sts = load_epoch_concatenated_trials(
                session_name=session_name,
                epoch=epoch,
                trialtypes=None,  # All trial types
                SNRthresh=0,
                include_raw=False,
                verbose=True
            )
            print(f"  -> Successfully loaded {len(sts)} spike trains")
            
        except Exception as e:
            print(f"  -> Error loading epoch {epoch}: {e}")
    
    # Test custom epoch definition
    print(f"\nTesting custom epoch definition...")
    try:
        custom_epoch = ('CUE-ON', -100 * pq.ms, 200 * pq.ms)
        sts = load_epoch_concatenated_trials(
            session_name=session_name,
            epoch=custom_epoch,
            trialtypes=None,
            SNRthresh=0,
            include_raw=False,
            verbose=True
        )
        print(f"  -> Successfully loaded {len(sts)} spike trains with custom epoch")
        
    except Exception as e:
        print(f"  -> Error loading custom epoch: {e}")


if __name__ == "__main__":
    # Run the main example
    sts = example_usage()
    
    # Uncomment to test different epochs
    # demonstrate_epoch_loading()
