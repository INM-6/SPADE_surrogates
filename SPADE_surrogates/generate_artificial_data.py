"""
Spike Train Analysis with Artificial Data Generation

This generates artificial data using
Poisson processes with refractory periods and Gamma processes.
"""

# -*- coding: utf-8 -*-
import math
import sys
import os
import warnings

import numpy as np
import quantities as pq
import elephant.spike_train_generation as stg
import elephant.statistics as stat
import elephant.kernels as kernels
import scipy

from scipy.special import gamma as gamma_function, erf
from scipy.optimize import brentq
from tqdm import tqdm

sys.path.insert(0, '../data/multielectrode_grasp/code/python-neo')
import neo

sys.path.append('./')


def estimate_rate_deadtime(
        spiketrain,
        max_refractory,
        sep,
        sampling_period=0.1 * pq.ms,
        sigma=100 * pq.ms,
        trial_length=500 * pq.ms):
    """
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

    # create list of trials and deconcatenate data
    trial_list = create_st_list(spiketrain=spiketrain, sep=sep)

    # using Shinomoto rate estimation
    if all(len(trial) > 10 for trial in trial_list):
        rate_list = [instantaneous_rate(
            spiketrains=trial,
            sampling_period=sampling_period,
            border_correction=True) for trial in trial_list]
    else:
        rate_list = [instantaneous_rate(
            spiketrains=trial,
            sampling_period=sampling_period,
            kernel=stat.kernels.GaussianKernel(sigma=sigma),
            border_correction=True) for trial in trial_list]

    # reconcatenating rates
    rate = np.zeros(
        shape=len(trial_list) * int(
            (trial_length.simplified.magnitude +
             sep.simplified.magnitude)/sampling_period.simplified.magnitude))
    for trial_id, rate_trial in enumerate(rate_list):
        start_id = trial_id * int(
            ((trial_length + sep)/sampling_period).simplified.magnitude)
        stop_id = start_id + int(
            (trial_length/sampling_period).simplified.magnitude)
        rate[start_id: stop_id] = rate_trial.flatten().magnitude

    rate = neo.AnalogSignal(signal=rate,
                            units=rate_list[0].units,
                            sampling_period=sampling_period)

    # calculating refractory period
    refractory_period = estimate_deadtime(
        spiketrain, max_dead_time=max_refractory)

    return rate, refractory_period, rate_list


def instantaneous_rate(spiketrains, sampling_period, kernel='auto',
                       cutoff=5.0, t_start=None, t_stop=None, trim=False,
                       center_kernel=True, border_correction=False):
    """
    Estimates instantaneous firing rate by kernel convolution.

    Visualization of this function is covered in Viziphant:
    :func:`viziphant.statistics.plot_instantaneous_rates_colormesh`.


    Parameters
    ----------
    spiketrains : neo.SpikeTrain or list of neo.SpikeTrain
        Neo object(s) that contains spike times, the unit of the time stamps,
        and `t_start` and `t_stop` of the spike train.
    sampling_period : pq.Quantity
        Time stamp resolution of the spike times. The same resolution will
        be assumed for the kernel.
    kernel : 'auto' or Kernel, optional
        The string 'auto' or callable object of class `kernels.Kernel`.
        The kernel is used for convolution with the spike train and its
        standard deviation determines the time resolution of the instantaneous
        rate estimation. Currently implemented kernel forms are rectangular,
        triangular, epanechnikovlike, gaussian, laplacian, exponential, and
        alpha function.
        If 'auto', the optimized kernel width (that is not adaptive)
        for the rate estimation is calculated according to
        :cite:`statistics-Shimazaki2010_171` and with
        this width a gaussian kernel is constructed. Automatized calculation
        of the kernel width is not available for other than gaussian kernel
        shapes.
        Default: 'auto'
    cutoff : float, optional
        This factor determines the cutoff of the probability distribution of
        the kernel, i.e., the considered width of the kernel in terms of
        multiples of the standard deviation sigma.
        Default: 5.0
    t_start : pq.Quantity, optional
        Start time of the interval used to compute the firing rate.
        If None, `t_start` is assumed equal to `t_start` attribute of
        `spiketrain`.
        Default: None
    t_stop : pq.Quantity, optional
        End time of the interval used to compute the firing rate (included).
        If None, `t_stop` is assumed equal to `t_stop` attribute of
        `spiketrain`.
        Default: None
    trim : bool, optional
        Accounts for the asymmetry of a kernel.
        If False, the output of the Fast Fourier Transformation being a longer
        vector than the input vector by the size of the kernel is reduced back
        to the original size of the considered time interval of the
        `spiketrain` using the median of the kernel. False (no trimming) is
        equivalent to 'same' convolution mode for symmetrical kernels.
        If True, only the region of the convolved signal is returned, where
        there is complete overlap between kernel and spike train. This is
        achieved by reducing the length of the output of the Fast Fourier
        Transformation by a total of two times the size of the kernel, and
        `t_start` and `t_stop` are adjusted. True (trimming) is equivalent to
        'valid' convolution mode for symmetrical kernels.
        Default: False
    center_kernel : bool, optional
        If set to True, the kernel will be translated such that its median is
        centered on the spike, thus putting equal weight before and after the
        spike. If False, no adjustment is performed such that the spike sits at
        the origin of the kernel.
        Default: True
    border_correction : bool, optional
        Apply a boundary correction.
        Only possible in the case of a Gaussian kernel.
        Default: False

    Returns
    -------
    rate : neo.AnalogSignal
        2D matrix that contains the rate estimation in unit hertz (Hz) of shape
        ``(time, len(spiketrains))`` or ``(time, 1)`` in case of a single
        input spiketrain. `rate.times` contains the time axis of the rate
        estimate: the unit of this property is the same as the resolution that
        is given via the argument `sampling_period` to the function.

    Raises
    ------
    TypeError
        If `spiketrain` is not an instance of `neo.SpikeTrain`.

        If `sampling_period` is not a `pq.Quantity`.

        If `sampling_period` is not larger than zero.

        If `kernel` is neither instance of `kernels.Kernel` nor string 'auto'.

        If `cutoff` is neither `float` nor `int`.

        If `t_start` and `t_stop` are neither None nor a `pq.Quantity`.

        If `trim` is not `bool`.
    ValueError
        If `sampling_period` is smaller than zero.

        If `kernel` is 'auto' and the function was unable to calculate optimal
        kernel width for instantaneous rate from input data.

    Warns
    -----
    UserWarning
        If `cutoff` is less than `min_cutoff` attribute of `kernel`, the width
        of the kernel is adjusted to a minimally allowed width.

    Notes
    -----
    The resulting instantaneous firing rate values smaller than ``0``, which
    can happen due to machine precision errors, are clipped to zero.

    Examples
    --------
    Example 1. Automatic kernel estimation.

    >>> import neo
    >>> import quantities as pq
    >>> from elephant import statistics
    >>> spiketrain = neo.SpikeTrain([0.3, 4.5, 6.7, 9.3], t_stop=10, units='s')
    >>> rate = statistics.instantaneous_rate(spiketrain,
    ...                                      sampling_period=10 * pq.ms,
    ...                                      kernel='auto')
    >>> rate
    AnalogSignal with 1 channels of length 1000; units Hz; datatype float64
    annotations: {'t_stop': array(10.) * s,
      'kernel': {'type': 'GaussianKernel',
       'sigma': '7.273225922958104 s',
       'invert': False}}
    sampling rate: 0.1 1/ms
    time: 0.0 s to 10.0 s

    Example 2. Manually set kernel.

    >>> from elephant import kernels
    >>> spiketrain = neo.SpikeTrain([0], t_stop=1, units='s')
    >>> kernel = kernels.GaussianKernel(sigma=300 * pq.ms)
    >>> rate = statistics.instantaneous_rate(spiketrain,
    ...        sampling_period=200 * pq.ms, kernel=kernel, t_start=-1 * pq.s)
    >>> rate
    AnalogSignal with 1 channels of length 10; units Hz; datatype float64
    annotations: {'t_stop': array(1.) * s,
      'kernel': {'type': 'GaussianKernel',
       'sigma': '300.0 ms',
       'invert': False}}
    sampling rate: 0.005 1/ms
    time: -1.0 s to 1.0 s
    >>> rate.magnitude
    array([[0.01007419],
       [0.05842767],
       [0.22928759],
       [0.60883028],
       [1.0938699 ],
       [1.3298076 ],
       [1.0938699 ],
       [0.60883028],
       [0.22928759],
       [0.05842767]])

    """
    def optimal_kernel(st):
        width_sigma = None
        if len(st) > 0:
            width_sigma = stat.optimal_kernel_bandwidth(
                st.magnitude, times=None, bootstrap=False)['optw']
        if width_sigma is None:
            raise ValueError("Unable to calculate optimal kernel width for "
                             "instantaneous rate from input data.")
        return kernels.GaussianKernel(width_sigma * st.units)

    if border_correction and not \
            (kernel == 'auto' or isinstance(kernel, kernels.GaussianKernel)):
        raise ValueError(
            'The boundary correction is only implemented'
            ' for Gaussian kernels.')

    if isinstance(spiketrains, neo.SpikeTrain):
        if kernel == 'auto':
            kernel = optimal_kernel(spiketrains)
        spiketrains = [spiketrains]
    elif not isinstance(spiketrains, (list, tuple)):
        raise TypeError(
            "'spiketrains' must be a list of neo.SpikeTrain's or a single "
            "neo.SpikeTrain. Found: '{}'".format(type(spiketrains)))

    if not stat.is_time_quantity(sampling_period):
        raise TypeError(
            "The 'sampling_period' must be a time Quantity. \n"
            "Found: {}".format(type(sampling_period)))

    if sampling_period.magnitude < 0:
        raise ValueError("The 'sampling_period' ({}) must be non-negative.".
                         format(sampling_period))

    if not (isinstance(kernel, kernels.Kernel) or kernel == 'auto'):
        raise TypeError(
            "'kernel' must be either instance of class elephant.kernels.Kernel"
            " or the string 'auto'. Found: %s, value %s" % (type(kernel),
                                                            str(kernel)))

    if not isinstance(cutoff, (float, int)):
        raise TypeError("'cutoff' must be float or integer")

    if not stat.is_time_quantity(t_start, allow_none=True):
        raise TypeError("'t_start' must be a time Quantity")

    if not stat.is_time_quantity(t_stop, allow_none=True):
        raise TypeError("'t_stop' must be a time Quantity")

    if not isinstance(trim, bool):
        raise TypeError("'trim' must be bool")

    stat.check_neo_consistency(
        spiketrains,
        object_type=neo.SpikeTrain,
        t_start=t_start, t_stop=t_stop)
    if kernel == 'auto':
        if len(spiketrains) == 1:
            kernel = optimal_kernel(spiketrains[0])
        else:
            raise ValueError("Cannot estimate a kernel for a list of spike "
                             "trains. Please provide a kernel explicitly "
                             "rather than 'auto'.")

    if t_start is None:
        t_start = spiketrains[0].t_start
    if t_stop is None:
        t_stop = spiketrains[0].t_stop

    units = pq.CompoundUnit(
        "{}*s".format(sampling_period.rescale('s').item()))
    t_start = t_start.rescale(spiketrains[0].units)
    t_stop = t_stop.rescale(spiketrains[0].units)

    n_bins = int(((t_stop - t_start) / sampling_period).simplified) + 1
    time_vectors = np.zeros((len(spiketrains), n_bins), dtype=np.float64)
    hist_range_end = t_stop + sampling_period.rescale(spiketrains[0].units)
    hist_range = (t_start.item(), hist_range_end.item())
    for i, st in enumerate(spiketrains):
        time_vectors[i], _ = np.histogram(st.magnitude, bins=n_bins,
                                          range=hist_range)

    if cutoff < kernel.min_cutoff:
        cutoff = kernel.min_cutoff
        warnings.warn("The width of the kernel was adjusted to a minimally "
                      "allowed width.")

    # An odd number of points correctly resolves the median index and the
    # fact that the peak of an instantaneous rate should be centered at t=0
    # for symmetric kernels applied on a single spike at t=0.
    # See issue https://github.com/NeuralEnsemble/elephant/issues/360
    n_half = math.ceil(cutoff * (
            kernel.sigma / sampling_period).simplified.item())
    cutoff_sigma = cutoff * kernel.sigma.rescale(units).magnitude
    if center_kernel:
        # t_arr must be centered at the kernel median.
        # Not centering on the kernel median leads to underestimating the
        # instantaneous rate in cases when sampling_period >> kernel.sigma.
        median = kernel.icdf(0.5).rescale(units).item()
    else:
        median = 0
    t_arr = np.linspace(-cutoff_sigma + median, stop=cutoff_sigma + median,
                        num=2 * n_half + 1, endpoint=True) * units

    if center_kernel:
        # keep the full convolve range and do the trimming afterwards;
        # trimming is performed according to the kernel median index
        fft_mode = 'full'
    elif trim:
        # no median index trimming is involved
        fft_mode = 'valid'
    else:
        # no median index trimming is involved
        fft_mode = 'same'

    time_vectors = time_vectors.T  # make it (time, units)
    kernel_arr = np.expand_dims(kernel(t_arr).rescale(pq.Hz).magnitude, axis=1)
    rate = scipy.signal.fftconvolve(time_vectors,
                                    kernel_arr,
                                    mode=fft_mode)
    # the convolution of non-negative vectors is non-negative
    rate = np.clip(rate, a_min=0, a_max=None, out=rate)

    if center_kernel:  # account for the kernel asymmetry
        median_id = kernel.median_index(t_arr)
        # the size of kernel() output matches the input size, len(t_arr)
        kernel_array_size = len(t_arr)
        if not trim:
            rate = rate[median_id: -kernel_array_size + median_id]
        else:
            rate = rate[2 * median_id: -2 * (kernel_array_size - median_id)]
            t_start = t_start + median_id * units
            t_stop = t_stop - (kernel_array_size - median_id) * units
    else:
        rate = rate[:-1]

    kernel_annotation = dict(type=type(kernel).__name__,
                             sigma=str(kernel.sigma),
                             invert=kernel.invert)

    rate = neo.AnalogSignal(signal=rate,
                            sampling_period=sampling_period,
                            units=pq.Hz, t_start=t_start, t_stop=t_stop,
                            kernel=kernel_annotation)

    if border_correction:
        sigma = kernel.sigma.simplified.magnitude
        times = rate.times.simplified.magnitude
        correction_factor = 2 / (
                erf((t_stop.simplified.magnitude - times) / (
                            np.sqrt(2.) * sigma))
                - erf((t_start.simplified.magnitude - times) / (
                    np.sqrt(2.) * sigma)))

        rate *= correction_factor[:, None]

        duration = t_stop.simplified.magnitude - t_start.simplified.magnitude
        # ensure integral over firing rate yield the exact number of spikes
        for i, spiketrain in enumerate(spiketrains):
            if len(spiketrain) > 0:
                rate[:, i] *= len(spiketrain) /\
                              (np.mean(rate[:, i]).magnitude * duration)

    return rate


def estimate_deadtime(spiketrain, max_dead_time):
    """
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
    if len(spiketrain) > 1:
        isi = np.diff(spiketrain.simplified.magnitude)
        dead_time = np.min(
            isi, initial=max_dead_time.simplified.magnitude) * pq.s
    else:
        dead_time = max_dead_time

    return dead_time


def create_st_list(spiketrain, sep,
                   epoch_length=0.5*pq.s):
    """
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
    spiketrain_list = []

    t_max = spiketrain.t_stop

    trial = 0
    t_start = 0.0 * pq.s
    sep = sep.rescale(pq.s)
    epoch_length = epoch_length.rescale(pq.s)
    while t_start < t_max - sep:
        t_start = trial * (epoch_length + sep)
        t_stop = trial * (epoch_length + sep) + epoch_length

        if t_start > t_max - sep:
            break
        cutted_st = spiketrain[np.all(
            [spiketrain > t_start, spiketrain < t_stop], axis=0)] - t_start
        cutted_st.t_start = 0.0 * pq.s
        cutted_st.t_stop = epoch_length
        spiketrain_list.append(cutted_st)
        trial += 1
    return spiketrain_list


def get_cv2(spiketrain, sep):
    """
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
    if len(spiketrain) > 5:
        spiketrain_list = create_st_list(spiketrain, sep=sep)
        isis = [np.diff(st.magnitude)
                for st in spiketrain_list
                if len(st) > 2]
        cv2_list = [stat.cv2(isi, with_nan=True) for isi in isis]
        return np.nanmean(cv2_list)
    return np.nan


def get_cv2_from_shape_factor(shape_factor):
    """
    calculates cv2 from shape factor based on van Vreswijk's formula

    Parameters
    ----------
    shape_factor : float

    Returns
    -------
    cv2 : float
    """
    return gamma_function(2 * shape_factor) / \
        (shape_factor * (2 ** (shape_factor - 1)
                         * gamma_function(shape_factor))) ** 2


def get_shape_factor_from_cv2(cv2):
    """
    calculates shape factor from cv2 based on van Vreswijk's formula

    Parameters
    ----------
    cv2 : float

    Returns
    -------
    shape_factor : float
    """

    return brentq(lambda x: get_cv2_from_shape_factor(x) - cv2, 0.05, 50.)


def get_cv_operational_time(spiketrain, rate_list, sep):
    """
    Calculate coefficient of variation (CV) of spike train in operational time.
    
    Operational time is defined as the integral of the firing rate over time,
    which accounts for time-varying firing rates.
    
    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        Input spike train data
    rate_list : list
        List of firing rates corresponding to each trial
    sep : pq.Quantity
        Separation time between trials
        
    Returns
    -------
    cv : float
        Coefficient of variation in operational time
    """
    # Split concatenated spike train into individual trials
    trial_list = create_st_list(spiketrain, sep=sep)
    isis_operational_time = []

    # Return CV=1 if there's less than one spike per trial on average
    total_spikes = sum(len(trial) for trial in trial_list)
    if total_spikes < len(trial_list):
        return 1

    # Process each trial with its corresponding firing rate
    for firing_rate, trial_spikes in zip(rate_list, trial_list):
        # Skip trials with fewer than 2 spikes (no ISIs possible)
        if len(trial_spikes) <= 1:
            continue
            
        # Create time points including start and stop times
        real_time_points = np.hstack((
            firing_rate.times.simplified.magnitude,
            firing_rate.t_stop.simplified.magnitude
        ))
        
        # Find which time bins contain each spike
        spike_bin_indices = np.searchsorted(real_time_points, trial_spikes)

        # Calculate operational time as cumulative integral of firing rate
        rate_magnitude = firing_rate.simplified.magnitude
        sampling_period = firing_rate.sampling_period.simplified.magnitude
        operational_time_cumsum = np.cumsum(rate_magnitude * sampling_period)
        operational_time_points = np.hstack((0., operational_time_cumsum))
        
        # Map spikes to operational time
        # Start with left border alignment
        spike_operational_times = operational_time_points[spike_bin_indices - 1]
        
        # Calculate relative position within each bin
        spike_positions_in_bins = (
            (trial_spikes - real_time_points[spike_bin_indices - 1]) / 
            sampling_period
        )
        
        # Add interpolated position within operational time bins
        bin_operational_durations = (
            operational_time_points[spike_bin_indices] - 
            operational_time_points[spike_bin_indices - 1]
        )
        spike_operational_times += (
            bin_operational_durations * spike_positions_in_bins
        )
        
        # Calculate inter-spike intervals in operational time
        trial_isis = np.diff(spike_operational_times)
        isis_operational_time.append(trial_isis)

    # Calculate coefficient of variation from all ISIs
    total_isis = np.sum([len(trial_isi) for trial_isi in isis_operational_time])
    cv = 1  # Default value
    
    if total_isis > 3:  # Need sufficient data for reliable CV estimate
        # Calculate mean ISI across all trials
        mean_isi = (
            np.sum([np.sum(trial_isi) for trial_isi in isis_operational_time]) / 
            total_isis
        )
        
        # Calculate variance of ISIs
        variance_isi = np.sum([
            np.sum((trial_isi - mean_isi) ** 2) 
            for trial_isi in isis_operational_time
        ]) / total_isis
        
        # CV = standard deviation / mean
        cv = np.sqrt(variance_isi) / mean_isi

    return cv



def generate_artificial_data(data, seed, max_refractory, processes, sep):
    """
    Generate artificial spike train data using Poisson and Gamma processes.
    
    This function creates artificial data that matches the statistical properties
    of the input data, including firing rates, refractory periods, and 
    coefficient of variation.
    
    Parameters
    ----------
    data : list
        List of neo.SpikeTrain objects (experimental data)
    seed : int
        Random seed for reproducible generation
    max_refractory : pq.Quantity
        Maximum refractory period to consider
    processes : list
        List of process types to generate ('ppd' and/or 'gamma')
    sep : pq.Quantity
        Buffer time between trials
        
    Returns
    -------
    ppd_spiketrains : list
        Poisson processes with refractory period (if 'ppd' in processes)
    gamma_spiketrains : list
        Gamma processes (if 'gamma' in processes)
    cv2_values : list
        CV2 values estimated from original data
    """
    # Set random seed for reproducibility
    np.random.seed(seed=seed)

    # Initialize storage lists
    estimated_rates = []
    estimated_refractory_periods = []
    ppd_spiketrains = []
    gamma_spiketrains = []
    cv2_values = []

    print(f"Generating artificial data for {len(data)} spike trains...")
    
    for spiketrain in tqdm(data, desc="Processing spike trains"):
        # Estimate firing rate and refractory period from data
        firing_rate, refractory_period, rate_profile_list = (
            estimate_rate_deadtime(
                spiketrain=spiketrain,
                max_refractory=max_refractory,
                sep=sep
            )
        )
        
        # Calculate CV2 (coefficient of variation based on adjacent ISIs)
        cv2 = get_cv2(spiketrain, sep=sep)
        
        # Store estimated parameters
        estimated_rates.append(firing_rate)
        estimated_refractory_periods.append(refractory_period)
        cv2_values.append(cv2)

        # Generate Poisson process with refractory period
        if 'ppd' in processes:
            ppd_spiketrain = stg.inhomogeneous_poisson_process(
                rate=firing_rate,
                as_array=False,
                refractory_period=refractory_period
            )
            # Preserve original annotations and units
            ppd_spiketrain.annotate(**spiketrain.annotations)
            ppd_spiketrain = ppd_spiketrain.rescale(pq.s)
            ppd_spiketrains.append(ppd_spiketrain)

        # Generate Gamma process
        if 'gamma' in processes:
            # Calculate CV in operational time for shape parameter
            cv_operational = get_cv_operational_time(
                spiketrain=spiketrain,
                rate_list=rate_profile_list,
                sep=sep
            )
            
            # Shape factor is inverse of CV squared
            shape_factor = 1 / (cv_operational ** 2)
            print(f'CV = {cv_operational:.3f}, Shape factor = {shape_factor:.3f}')
            
            # Generate Gamma process if there's any activity
            if np.count_nonzero(firing_rate) != 0:
                gamma_spiketrain = stg.inhomogeneous_gamma_process(
                    rate=firing_rate, 
                    shape_factor=shape_factor
                )
            else:
                # Create empty spike train for inactive neurons
                gamma_spiketrain = neo.SpikeTrain(
                    [] * pq.s,
                    t_start=spiketrain.t_start,
                    t_stop=spiketrain.t_stop
                )
            
            # Preserve original annotations
            gamma_spiketrain.annotate(**spiketrain.annotations)
            gamma_spiketrains.append(gamma_spiketrain)
    
    return ppd_spiketrains, gamma_spiketrains, cv2_values


def main():
    """
    Main execution function that processes experimental data and generates
    artificial spike trains based on configuration parameters.
    """
    import rgutils
    import yaml
    from yaml import Loader

    # Load configuration parameters
    print("Loading configuration...")
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
    
    # Analysis parameters
    SNR_threshold = config['SNR_thresh']
    synch_size = config['synchsize']
    trial_separation = 2 * winlen * binsize
    max_refractory_period = 4 * pq.ms
    load_original_data = False

    print(f"Configuration loaded. Processing {len(sessions)} sessions...")
    print(f"Epochs: {epochs}")
    print(f"Trial types: {trialtypes}")
    print(f"Processes to generate: {processes}")

    # Process each session, epoch, and trial type combination
    for session in sessions:
        # Create output directories
        ppd_dir = f'../data/artificial_data/ppd/{session}'
        gamma_dir = f'../data/artificial_data/gamma/{session}'
        
        if not os.path.exists(ppd_dir):
            os.makedirs(ppd_dir)
        if not os.path.exists(gamma_dir):
            os.makedirs(gamma_dir)

        for epoch in epochs:
            for trialtype in trialtypes:
                print(f'\n--- Processing: {session} | {epoch} | {trialtype} ---')
                
                # Load spike train data
                if load_original_data:
                    print('Loading experimental data...')
                    spike_trains = rgutils.load_epoch_concatenated_trials(
                        session,
                        epoch,
                        trialtypes=trialtype,
                        SNRthresh=SNR_threshold,
                        synchsize=synch_size,
                        sep=trial_separation
                    )
                else:
                    print('Loading pre-concatenated spike trains...')
                    data_path = (
                        f'../data/concatenated_spiketrains/{session}/'
                        f'{epoch}_{trialtype}.npy'
                    )
                    spike_trains = np.load(data_path, allow_pickle=True)

                # Generate artificial data
                print("Generating artificial data...")
                ppd_trains, gamma_trains, cv2_list = generate_artificial_data(
                    data=spike_trains,
                    seed=seed,
                    max_refractory=max_refractory_period,
                    processes=processes,
                    sep=trial_separation
                )

                print('Data generation complete!')

                # Save generated data
                print('Saving data...')
                if 'ppd' in processes:
                    ppd_filename = f'{ppd_dir}/ppd_{epoch}_{trialtype}.npy'
                    np.save(ppd_filename, ppd_trains)
                    print(f'Saved PPD data: {ppd_filename}')
                
                if 'gamma' in processes:
                    gamma_filename = f'{gamma_dir}/gamma_{epoch}_{trialtype}.npy'
                    cv2_filename = f'{gamma_dir}/cv2s_{epoch}_{trialtype}.npy'
                    np.save(gamma_filename, gamma_trains)
                    np.save(cv2_filename, cv2_list)
                    print(f'Saved Gamma data: {gamma_filename}')
                    print(f'Saved CV2 data: {cv2_filename}')

    print("\n=== All processing complete! ===")


if __name__ == '__main__':
    main()
