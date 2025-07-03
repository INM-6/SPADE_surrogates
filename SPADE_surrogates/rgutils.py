# -*- coding: utf-8 -*-
"""
Utility functions for Reach-to-Grasp data.
"""
import os
import sys

import numpy as np
import quantities as pq

# sys.path.insert(0, '../data/multielectrode_grasp/code/python-neo')
sys.path.insert(0, '../data/multielectrode_grasp/code/python-odml')
import neo

sys.path.insert(0, '../data/multielectrode_grasp/code/reachgraspio')
import reachgraspio as rgio

# sys.path.insert(0, '../data/multielectrode_grasp/code')
# from neo_utils import add_epoch, cut_segment_by_epoch, get_events


def data_path(session):
    """
    Finds the path associated to a given session.

    Parameters
    ----------
    session : str or ReachGraspIO object
        if a string, the name of a recording subsession. E.g: 'l101126-002'
        Otherwise, a rg.rgio.ReachGraspIO object.

    Returns
    -------
    path : str
        the path of the given session

    """
    if isinstance(session, str):
        path = '../data/multielectrode_grasp/datasets/'
    elif isinstance(session, rgio.ReachGraspIO):
        fullpath = session.filename
        path = ''
        for s in fullpath.split('/')[:-1]:
            path = path + s + '/'
    path = os.path.abspath(path) + '/'
    return path


def odml_path(session):
    """
    Finds the path of the odml file associated to a given session.

    Parameters
    ----------
    session : str or ReachGraspIO object
        if a string, the name of a recording subsession. E.g: 'l101126-002'
        Otherwise, a rg.rgio.ReachGraspIO object.

    Returns
    -------
    path : str
        the path of the given session odml

    """
    if isinstance(session, str):
        path = '../data/multielectrode_grasp/datasets/'
    return path


def _session(session_name):
    """
    Wrapper to load a ReachGraspIO session if input is a str, and do nothing
    if input is already a ReachGraspIO object.
    Returns the session and its associated filename.

    Parameters:
    -----------
    session : str of session loaded with ReachGraspIO
        if a string, the name of a recording subsession. E.g: 'l101126-002'
        Otherwise, a rg.rgio.ReachGraspIO object.

    Returns
    -------
    session : ReachGraspIO
    session_name : str
    """
    path = data_path(session_name)
    path_odml = odml_path(session_name)
    session = rgio.ReachGraspIO(path + session_name, odml_directory=path_odml)
    return session


def st_id(spiketrain):
    """
    associates to a Lilou's SpikeTrain st an unique ID, given by the float
    100* electrode_id + unit_id.
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
    returns the SNR of the waveforms of spiketrains, as computed in
    Kelly et al (2007):
    * compute the mean waveform
    * define the signal as the peak-to-through of such mean waveform
    * define the noise as double the std.dev. of the values of all waveforms,
      each normalised by subtracting the mean waveform

    Parameters:
    -----------
    spiketrain : SpikeTrain
        spike train loaded with rgio (has attribute "vaweforms")

    Returns:
    --------
    snr: float
        The SNR of the input spike train
    """
    mean_waveform = spiketrain.waveforms.mean(axis=0)
    signal = mean_waveform.max() - mean_waveform.min()
    SD = (spiketrain.waveforms - mean_waveform).std()
    return signal / (2. * SD)


def calc_spiketrains_SNR(session, units='all'):
    """
    Calculates the signal-to-noise ratio (SNR) of each SpikeTrain in the
    specified session.

    Parameters:
    -----------
    session : str of session loaded with ReachGraspIO
        if a string, the name of a recording subsession. E.g: 'l101126-002'
        Otherwise, a rg.rgio.ReachGraspIO object.
    units : str
        which type of units to consider:
        * 'all': returns the SNR values of all units in the session
        * 'sua': returns SUAs' SNR values only
        * 'mua': returns MIAs' SNR values only

    Returns:
    --------
    SNRdict : dict
        a dictionary of unit ids and associated SNR values
    """
    session = _session(session)

    block = session.read_block(
        channel_list=list(range(1, 97)), nsx=[2], units=[],
        waveforms=True)

    sts = [st for st in block.segments[0].spiketrains]
    if units == 'sua':
        sts = [st for st in sts if st.annotations['sua']]
    elif units == 'mua':
        sts = [st for st in sts if st.annotations['mua']]

    SNRdict = {}
    for st in sts:
        sua_id = st_id(st)
        SNRdict[sua_id] = SNR_kelly(st)

    return SNRdict


# ==========================================================================
# Loading routines:
# load_session(): loads spike trains from a full session
# load_epoch_as_list(): load spike trains from an epoch; for each SUA, a list
# load_epoch_concatenated_trials(): load concat'd spike trains from an epoch
# ==========================================================================


def load_session_sts(
        session_name, units='sua', SNRthresh=0, synchsize=0, dt=None,
        dt2=None, verbose=False):
    """
    Load SUA spike trains of a full specific session from Lilou's data.

    Parameters:
    -----------
    session_name : str of session loaded with ReachGraspIO
        if a string, the name of a recording subsession. E.g: 'l101126-002'
        Otherwise, a rg.rgio.ReachGraspIO object.
    SNRthresh: float, optional
        lower threshold for the waveforms' SNR of SUAs to be considered.
        SUAs with a lower or equal SNR are not loaded.
        Default: 0
    synchsize: int, optional
        minimum size of synchronous events to be removed from the data.
        If 0, no synchronous events are removed.
        Synchrony is defined by the parameter dt.
    dt: Quantity, optional
        time lag within which synchronous spikes are considered highly
        synchronous ("synchrofacts"). If None, the sampling period of the
        recording system (1 * session.nev_unit) is used.
        Default: None
    dt2: Quantity, optional
        isolated spikes falling within a time lag dt2 from synchrofacts (see
        parameter dt) to be removed (see parameter synchsize) are also
        removed. If None, the sampling period of the recording system
        (1 * session.nev_unit) is used.
        Default: None

    Returns:
    --------
    data : list of SpikeTrain
        a list of all SpikeTrains in the session
    """
    # Load session, and create block depending on the trigger
    session = _session(session_name)

    if verbose:
        print(('Load data (session: %s)...' % session_name))

    block = session.read_block(channels=list(range(1, 97)), nsx_to_load=None,
                               units='all', load_waveforms=False)

    sts = list(block.segments[0].spiketrains)

    if units == 'sua':
        sts = [st for st in sts if st.annotations['sua']]
    elif units == 'mua':
        sts = [st for st in sts if st.annotations['mua']]

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
        sts = [st for st in sts if st.annotations['SNR'] > SNRthresh]

    return sts


def load_epoch_as_lists(session_name, epoch, trialtypes=None, SNRthresh=0, verbose=False):
    """
    Load SUA spike trains of specific session and epoch from Lilou's data.

    * The output is a dictionary, with SUA ids as keys.
    * Each SUA id is associated to a list of spike trains, one per trial.
    * Each SpikeTrain is aligned to the epoch's trigger (see below) and has
      annotations indicating the corresponding trial type and much more.

    The epoch is either one of 6 specific epochs defined, following a
    discussion with Sonja, Alexa, Thomas, as 500 ms-long time segments
    each around a specific trigger, or a triplet consisting of a trigger
    and two time spans delimiting a time segment around the trigger.
    Additionally allows to select only one or more trialtypes, to consider
    SUAs with a minimum waveforms' signal-to-noise ratio, to remove
    synchronous spiking events of a certain minimum size (defined at a given
    time scale).
    The spike trains can be centered at the trigger associated to the epoch,
    or at the left or right end of the corresponding time segment.

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
        Name of a recording subsession. E.g: 'l101126-002'
    epoch : str or triplet
        if str, defines a trigger and a time segment around it (see above).
        if a triplet (tuple with 3 elements), its elements are, in order:
        * trigger [str] : a trigger (any string in session.trial_events)
        * t_pre [Quantity] : the left end of the time segment around the
          trigger. (> 0 for times before the trigger, < 0 for time after it)
        * t_post [Quantity] : the right end of the time segment around the
          trigger. (> 0 for times after the trigger, < 0 for time before it)
    trialtypes : str, optional
        One trial type, among those present in the session.
        8 Classical trial types for Lilou's sessions are:
        'SGHF', 'SGLF', 'PGHF', 'PGLF', 'HFSG', 'LFSG', 'HFPG', 'LFPG'.
        trialtypes can be one of such strings, or None.
        If None, all trial types in the session are considered.
        Default: None
    SNRthresh : float, optional
        Lower threshold for the waveforms' SNR of SUAs to be considered.
        SUAs with a lower or equal SNR are not loaded.
        Default: 0
    verbose : bool, optional
        Whether to print information as different steps are run

    Returns:
    --------
    data : dict
        A dictionary having SUA IDs as keys and lists of SpikeTrains as 
        corresponding values. Each SpikeTrain corresponds to the SUA spikes 
        in one trial (having the specified trial type(s)), during the 
        specified epoch.
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

    # Load session, and create block depending on the trigger
    session = _session(session_name)
    if verbose:
        print(('Load data (session: %s, epoch: %s, trialtype: %s)...' % (
            session_name, epoch, trialtypes)))
        print("  > load session %s, and define Block around trigger '%s'..." %
              (session_name, trigger))

    # Read block with all data
    block = session.read_block(
        nsx_to_load=None,
        n_starts=None,
        n_stops=None,
        channels=list(range(1, 97)),
        units='all',
        load_events=True,
        load_waveforms=False,
        scaling='raw'
    )

    data_segment = block.segments[0]
    
    # Find events with the specified trigger using modern Neo filter method
    start_events = _get_events_by_properties(
        data_segment,
        properties={
            'trial_event_labels': trigger,
            'performance_in_trial': session.performance_codes['correct_trial']
        }
    )
    
    if not start_events:
        if verbose:
            print(f"No events found for trigger '{trigger}'")
        return {}
    
    start_event = start_events[0]
    
    # Create epochs around the events using modern Neo approach
    epoch_obj = _create_epoch_from_events(
        data_segment,
        start_event,
        t_pre=t_pre,
        t_post=t_post,
        name=str(epoch)
    )
    
    # Cut segments by epoch using modern Neo time slicing
    cut_trial_segments = _cut_segment_by_epoch_modern(
        data_segment, 
        epoch_obj, 
        reset_time=True
    )
    
    # Filter segments by trial type using modern Neo filter
    if trialtypes is not None:
        selected_trial_segments = [
            seg for seg in cut_trial_segments 
            if seg.annotations.get('belongs_to_trialtype') == trialtypes
        ]
    else:
        selected_trial_segments = cut_trial_segments
    
    # Extract spike trains with SNR filtering
    data = {}
    for seg_id, seg in enumerate(selected_trial_segments):
        # Filter spike trains that are SUAs using modern approach
        sua_spiketrains = [
            st for st in seg.spiketrains 
            if st.annotations.get('sua', False)
        ]
        
        for st in sua_spiketrains:
            # Check the SNR threshold
            if st.annotations.get('SNR', 0) > SNRthresh:
                sua_id = _check_snr(st, seg, seg_id)
                if sua_id not in data:
                    data[sua_id] = []
                data[sua_id].append(st)
    
    return data


def _get_events_by_properties(segment, properties):
    """
    Modern replacement for get_events function.
    Filter events in a segment based on properties.
    """
    matching_events = []
    
    for event in segment.events:
        match = True
        for prop_name, prop_value in properties.items():
            event_value = event.annotations.get(prop_name)
            if event_value != prop_value:
                match = False
                break
        
        if match:
            matching_events.append(event)
    
    return matching_events


def _create_epoch_from_events(segment, event, t_pre, t_post, name=None):
    """
    Modern replacement for add_epoch function.
    Create an Epoch object from events with pre and post time offsets.
    """
    # Calculate start times and durations
    start_times = event.times + t_pre
    durations = (t_post - t_pre) * np.ones(len(event.times))
    
    # Create epoch object
    epoch = neo.Epoch(
        times=start_times,
        durations=durations,
        labels=event.labels if hasattr(event, 'labels') else None,
        name=name
    )
    
    # Copy relevant annotations
    for key, value in event.annotations.items():
        epoch.annotate(**{key: value})
    
    return epoch


def _cut_segment_by_epoch_modern(segment, epoch, reset_time=True):
    """
    Modern replacement for cut_segment_by_epoch function.
    Cut a segment into multiple segments based on epoch timing using modern Neo.
    """
    cut_segments = []
    
    for i, (start_time, duration) in enumerate(zip(epoch.times, epoch.durations)):
        stop_time = start_time + duration
        
        # Use modern Neo time_slice method
        try:
            cut_segment = segment.time_slice(start_time, stop_time)
            
            if reset_time:
                # Shift time to start at 0
                cut_segment = cut_segment.time_shift(-start_time)
            
            # Add epoch-specific annotations
            cut_segment.annotate(
                epoch_index=i,
                epoch_name=epoch.name,
                original_start_time=start_time,
                epoch_duration=duration
            )
            
            # Copy trial-related annotations if they exist
            if hasattr(epoch, 'labels') and epoch.labels is not None and i < len(epoch.labels):
                cut_segment.annotate(epoch_label=epoch.labels[i])
            
            cut_segments.append(cut_segment)
            
        except Exception as e:
            print(f"Warning: Could not cut segment {i}: {e}")
            continue
    
    return cut_segments


def _check_snr(st, seg, seg_id):
    st.annotations['trial_id'] = seg.annotations[
        'trial_id']
    st.annotations['trial_type'] = seg.annotations[
        'belongs_to_trialtype']
    st.annotate(trial_id_trialtype=seg_id)
    el = st.annotations['channel_id']
    sua = st.annotations['unit_id']
    sua_id = el * 100 + sua * 1
    return sua_id


def load_epoch_concatenated_trials(
        session, epoch, trialtypes=None, SNRthresh=0, synchsize=0, dt=None,
        dt2=None, sep=100 * pq.ms, verbose=False):
    """
    Load a slice of Lilou's spike train data in a specified epoch
    (corresponding to a trigger and a time segment aroun it), select spike
    trains corresponding to specific trialtypes only, and concatenate them.

    The epoch is either one of 6 specific epochs defined, following a
    discussion with Sonja, Alexa, Thomas, as 500 ms-long time segments
    each around a specific trigger, or a triplet consisting of a trigger
    and two time spans delimiting a time segment around the trigger.

    The pre-defined epochs, the associated triggers, and the time spans
    t_pre and t_post (before and after the trigger, respectively) are:
    * epoch='start'     :  trigger='FP-ON'   t_pre=250 ms   t_post=250 ms
    * epoch='cue1'      :  trigger='CUE-ON'  t_pre=250 ms   t_post=250 ms
    * epoch='earlydelay':  trigger='FP-ON'   t_pre=0 ms     t_post=500 ms
    * epoch='latedelay' :  trigger='GO-ON'   t_pre=500 ms   t_post=0 ms
    * epoch='movement'  :  trigger='SR'      t_pre=200 ms   t_post=300 ms
    * epoch='hold'      :  trigger='RW'      t_pre=500 ms   t_post=0 ms

    Parameters:
    -----------
    session : str of session loaded with ReachGraspIO
        if a string, the name of a recording subsession. E.g: 'l101126-002'
        Otherwise, a rg.rgio.ReachGraspIO object.
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
    verbose : bool
        Whether to print information as different steps are run
    firing_rate_threshold: None or float
        Threshold for excluding neurons with high firing rate
        Default: None

    Returns:
    --------
    data : list
        a list of SpikeTrains, each obtained by concatenating all trials
        of the desired type(s) and during the specified epoch for that SUA.
    """
    # Load the data as a dictionary of SUA_id: [list of trials]
    data = load_epoch_as_lists(session, epoch, trialtypes=trialtypes,
                               SNRthresh=SNRthresh,
                               verbose=verbose)

    # Check that all spike trains in all lists have same t_start, t_stop
    t_pre = abs(list(data.values())[0][0].t_start)
    t_post = abs(list(data.values())[0][0].t_stop)
    if not all([np.all([abs(st.t_start) == t_pre
                        for st in st_list])
                for st_list in list(data.values())]):
        raise ValueError(
            'SpikeTrains have not same t_pre; cannot be concatenated')
    if not all([np.all([abs(st.t_stop) == t_post
                        for st in st_list])
                for st_list in list(data.values())]):
        raise ValueError(
            'SpikeTrains have not same t_post; cannot be concatenated')

    # Define time unit (nev_unit), trial duration, trial IDs to consider
    time_unit = list(data.values())[0][0].units
    trial_duration = (t_post + t_pre + sep).rescale(time_unit)
    trial_ids_of_chosen_types = np.unique(np.hstack([
        [st.annotations['trial_id_trialtype'] for st in st_list]
        for st_list in list(data.values())]))

    # Concatenate the lists of spike trains into a single SpikeTrain
    if verbose:
        print('  > concatenate trials...')
    conc_data = []
    for sua_id in sorted(data.keys()):
        trials_to_concatenate = []
        original_times = []
        # Create list of trials, each shifted by trial_duration*trial_id
        for tr in data[sua_id]:
            trials_to_concatenate.append(
                tr.rescale(time_unit).magnitude + (
                    (trial_duration * tr.annotations[
                        'trial_id_trialtype']).rescale(time_unit)).magnitude)
            original_times.extend(list(tr.magnitude))
        # Concatenate the trials (time unit lost!)
        if len(trials_to_concatenate) > 0:
            trials_to_concatenate = np.hstack(trials_to_concatenate)

        # Re-transform the concatenated spikes into a SpikeTrain
        st = neo.SpikeTrain(
            trials_to_concatenate * time_unit,
            t_stop=trial_duration * max(trial_ids_of_chosen_types)
            + trial_duration).rescale(pq.s)

        # Copy into the SpikeTrain the original annotations
        for key, value in list(data[sua_id][0].annotations.items()):
            if key != 'trial_id':
                st.annotations[key] = value
        st.annotate(original_times=original_times)
        conc_data.append(st)
    # Remove exactly synchronous spikes from data
    if not (synchsize == 0 or synchsize is None):
        Dt = time_unit if dt is None else dt
        Dt2 = time_unit if dt2 is None else dt2
        if verbose:
            sync_spikes_before_removal = len(
                find_synchrofact_spikes(conc_data, n=synchsize, dt=Dt)[1])
            print(f'  > remove synchrofacts (precision={Dt}) of'
                  f' size {synchsize:d}')
            print(f'    and their neighbours at distance <= {Dt2}...')
            print(
                f'    (# synch. spikes before removal: '
                f'{sync_spikes_before_removal:d})')

        sts = remove_synchrofact_spikes(conc_data, n=synchsize, dt=Dt, dt2=Dt2)
        for i, conc_st in enumerate(conc_data):
            sts[i].annotations = conc_st.annotations
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
