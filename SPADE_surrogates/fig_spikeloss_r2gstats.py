"""
Script to create the figure of the manuscript which shows the spike count
reduction found in experimental data.
"""
import itertools

import numpy as np

import quantities as pq
from elephant import conversion
from elephant import spike_train_surrogates
import elephant.statistics as stat

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from generate_artificial_data import estimate_rate_deadtime, create_st_list

from rgutils import load_processed_spike_trains


XLABELPAD = -0.25


def _total_count(binned_st):
    """
    Function calculates number of spikes if the spike train.

    Parameters
    ----------
    binned_st: elephant.conversion.BinnedSpikeTrain
        Binned spike train of input

    Returns
    -------
        number of spikes of the binned spike train
    -------

    """
    binned_st = binned_st.to_array()[0, :]
    binned_st = np.array(binned_st > 0, dtype=int)
    return sum(binned_st)


def _mean_std_spike_counts_after_binning(sts, mean_list, std_list):
    """
    Function calculating a list of mean and standard deviations of the
    spike counts of the input spike trains.

    Parameters
    ----------
    sts: list
        list of spiketrains in input
    mean_list: list
        list of mean spike counts
    std_list: list
        list of std of spike counts

    Returns
    -------
    mean_list, std_list calculated inplace

    """
    spike_counts = []
    for st in sts:
        binned_st = conversion.BinnedSpikeTrain(st, bin_size=binsize)
        spike_counts.append(_total_count(binned_st))
    mean_list.append(np.mean(spike_counts))
    std_list.append(np.std(spike_counts))


def _mean_std_residuals(binned_single_spike_count,
                        surrogates,
                        mean_residuals,
                        std_residuals):
    """
    Function calculating a list of mean and standard deviations of the
    residuals between the one original spike train and its uniformly dithered
    surrogates.

    Parameters
    ----------
    binned_single_spike_count: int
        spike count of the considered neuron
    surrogates: list
        list of uniformly dithered surrogates of the original neuron
    mean_residuals: list
        list of mean residuals of spike counts between neurons and their UD
        surrogates
    std_residuals: list
        list of std residuals of spike counts between neurons and their UD
        surrogates

    Returns
    -------
    mean_residuals, std_residuals calculated inplace

    """
    spike_counts = []
    residuals_surrogates = []
    for st in surrogates:
        binned_st = conversion.BinnedSpikeTrain(st, binsize=binsize)
        spike_counts.append(_total_count(binned_st))
        residuals_surrogates.append(
            binned_single_spike_count - _total_count(binned_st))
    mean_residuals.append(np.mean(np.array(residuals_surrogates)))
    std_residuals.append(np.std(np.array(residuals_surrogates)))


def calculate_residuals(sts, dither, binsize,
                        n_surr, epoch_length,
                        winlen):
    """
    Function calculating the residuals, their means and standard deviations
    between all spike counts of all neurons and the uniform dithering
    surrogates generated.

    Parameters
    ----------
    sts: list of neo.SpikeTrains
        spiketrains of the data set
    dither: pq.quantities
        dithering parameter of the surrogate generation
    binsize: pq.quantities
        binsize of the analysis
    n_surr: int
        number of surrogates generated
    epoch_length: pq.quantities
        length of each trial
    winlen: int
        window length of the spade analysis

    Returns
    -------
    firing_rates: list
        list of average firing rates of all neurons
    mean_residuals: list
        list of mean residuals of spike counts between neurons and their UD
        surrogates
    std_residuals: list
        list of std residuals of spike counts between neurons and their UD
        surrogates

    """

    # define separation interval between successive trials
    sep = 2 * binsize * winlen

    original_spike_counts = []
    binned_spike_counts = []
    mean_residuals = []
    std_residuals = []

    # calculate number of trials
    number_of_trials = int(sts[0].t_stop / (epoch_length + sep))

    # loop over spike trains
    for st in sts:
        # bin original spike train
        binned_st = conversion.BinnedSpikeTrain(st, bin_size=binsize)
        original_spike_counts.append(len(st))
        # calculate number of spikes in binned spike train
        binned_single_spike_count = _total_count(binned_st)
        binned_spike_counts.append(binned_single_spike_count)
        # dithering spikes
        surrogate_sts = spike_train_surrogates.dither_spikes(
            st, dither, n_surrogates=n_surr, edges=True)
        # calculating mean and std and residuals of original to binned data
        _mean_std_residuals(
            binned_single_spike_count,
            surrogate_sts,
            mean_residuals,
            std_residuals)

    firing_rates = original_spike_counts / (number_of_trials * epoch_length)

    return firing_rates, mean_residuals, std_residuals


def plot_loss_top_panel(
        ax_loss,
        ax_residuals,
        sts, dither, binsize,
        n_surr, epoch_length,
        winlen, fontsize):
    """
    Function producing the top panel of figure 2, representing the spike
    count reduction resulting by clipping and UD surrogate generation.
    with crosses of different colors, we represent the reduction in the spike
    count in function of the average firing rate.
    Each cross represent one neuron. In blue, we indicate the spike count
    reduction caused by clipping the original spike train.
    In orange, we indicate the spike count reduction followed by generating
    surrogates by UD and then clipping. The spike count reduction is
    expressed in percentage with respect to the original continuous-time
    spike train. Bars indicate the standard deviation of the spike count
    reduction calculated across n_surr surrogates.

    Parameters
    ----------
    ax_loss: plt.axes
        ax of the spike count reduction panel
    ax_residuals: plt.axes
        ax of the residuals panel
    sts: list of neo.SpikeTrains
        spiketrains of the data set
    dither: pq.quantities
        dithering parameter of the surrogate generation
    binsize: pq.quantities
        binsize of the analysis
    n_surr: int
        number of surrogates generated
    epoch_length: pq.quantities
        length of each trial
    winlen: int
        window length of the spade analysis
    fontsize: int
        fontsize for plotting

    """
    # define separation interval between trials
    sep = 2 * binsize * winlen

    original_spike_counts = []
    binned_spike_counts = []
    mean_spike_counts_dither = []
    std_spike_counts_dither = []

    number_of_trials = int(sts[0].t_stop / (epoch_length + sep))
    for st in sts:
        binned_st = conversion.BinnedSpikeTrain(st, binsize=binsize)
        original_spike_counts.append(len(st))
        binned_spike_counts.append(_total_count(binned_st))

        surrogate_sts = spike_train_surrogates.dither_spikes(
            st, dither, n=n_surr, edges=True)
        _mean_std_spike_counts_after_binning(
            surrogate_sts, mean_spike_counts_dither,
            std_spike_counts_dither)

    original_spike_counts = np.array(original_spike_counts)
    binned_spike_counts = np.array(binned_spike_counts)

    mean_spike_counts_dither = np.array(mean_spike_counts_dither)
    std_spike_counts_dither = np.array(std_spike_counts_dither)

    firing_rates = original_spike_counts / (number_of_trials * epoch_length)
    binned_loss = 1. - binned_spike_counts / original_spike_counts
    mean_loss_dither = 1. - mean_spike_counts_dither / original_spike_counts
    std_loss_dither = std_spike_counts_dither / original_spike_counts

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    ax_loss.scatter(firing_rates, binned_loss, label='Original',
                    color=colors[0], marker='x')

    ax_loss.errorbar(firing_rates, mean_loss_dither, yerr=std_loss_dither,
                     fmt='o', label='UD Surrogate', color=colors[1],
                     marker='x')

    ax_loss.set_ylabel('Spike count\ndecrease', fontsize=fontsize)

    ax_loss.set_xlim(left=2. / epoch_length.magnitude, right=65)
    ax_loss.set_ylim(bottom=-0.01, top=0.23)
    ax_loss.tick_params(axis="x", labelsize=8)
    ax_loss.tick_params(axis="y", labelsize=8)

    ax_loss.legend(fontsize=fontsize - 2, loc='upper right')

    ax_residuals.errorbar(
        firing_rates, -binned_loss + mean_loss_dither, yerr=std_loss_dither,
        fmt='o', label='UD Surrogate + clipping', color='grey',
        marker='x')

    ax_residuals.set_xlabel(
        'Average Firing rate (Hz)',
        fontsize=fontsize,
        labelpad=XLABELPAD)
    ax_residuals.set_ylabel('Residuals',
                            fontsize=fontsize)
    ax_residuals.set_xlim(left=0, right=65)
    ax_residuals.set_ylim(bottom=-0.05, top=0.125)
    ax_residuals.tick_params(axis="x", labelsize=8)
    ax_residuals.tick_params(axis="y", labelsize=8)
    ax_residuals.set_xticks(np.arange(0, 65, 20))


def plot_residuals(ax, sts, dither, binsize, fontsize, epoch_length,
                   winlen, n_surr):
    """
    Function producing the lower panel of Figure 2. Residuals computed as
    the difference between the original clipped spike trains and the UD
    binned and clipped surrogates, i.e., between the blue and the orange
    crosses.

    Parameters
    ----------
    ax: plt.axes
        ax of the panel
    sts: list of neo.SpikeTrains
        spiketrains of the data set
    dither: pq.quantities
        dithering parameter of the surrogate generation
    binsize: pq.quantities
        binsize of the analysis
    n_surr: int
        number of surrogates generated
    epoch_length: pq.quantities
        length of each trial
    winlen: int
        window length of the spade analysis

    """

    firing_rates, mean_residuals, std_residuals = \
        calculate_residuals(sts,
                            dither=dither * pq.s,
                            binsize=binsize,
                            n_surr=n_surr,
                            epoch_length=epoch_length,
                            winlen=winlen)
    ax.errorbar(firing_rates, mean_residuals, yerr=std_residuals,
                fmt='o', label='UD Surrogate + clipping', color='grey',
                marker='x', alpha=0.8)
    ax.set_xlabel(
        'Average Firing rate (Hz)',
        fontsize=fontsize,
        labelpad=XLABELPAD)
    ax.set_ylabel('Residuals', fontsize=fontsize)
    ax.set_xlim(left=0, right=65)
    ax.set_ylim(bottom=0, top=100)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xticks(np.arange(0, 65, 20))


def plot_isi_surr(st, ax, dither, num_surr=500, show_ylabel=True,
                  show_xlabel=True, fontsize=14, legend=False):
    """
    Function calculating the ISI distributions of two example
    neurons and the ISI distribution of UD surrogates generated from the
    original neurons. Neurons represented are, with (channel-id,unit-id)
    notation, (7,1) and (16,1) for monkey N, (5,1) and (6.1) for monkey L.
    or each monkey, on the left we show the ISI distributions of two example
    neurons (blue line) and the ISI distribution of UD surrogates generated
    from the original neurons (gray line: average across 500 surrogates;
    gray band represents the first standard deviation) on a 1ms resolution.

    Parameters
    ----------
    st: neo.SpikeTrain
    ax: plt.axes
        ax where to plot
    dither: pq.Quantity
        dithering parameter of the surrogate generation
    num_surr: int
        number of surrogates to generate to show the ISI distribution
    """
    # calculate isi of original spike train
    isi = stat.isi(st)
    # generate surrogates
    surr_list = spike_train_surrogates.dither_spikes(
        st, n_surrogates=num_surr, dither=dither * pq.s)
    # calculate isi of surrogates
    isis = [stat.isi(dithered_st) for dithered_st in surr_list]
    bin_distance = 0.001
    bins = np.arange(0, 0.05, bin_distance)
    ax.set_xlim(0, 0.05)
    # generate histogram of original st
    isi_distr, bin_edges = np.histogram(isi, bins=bins,
                                        density=True)
    # generate histogram of surrogates
    hist_list = [
        np.histogram(
            isi_item,
            bins=bins,
            density=True)[0] for isi_item in isis]
    # calculate mean and std of surrogates
    hist_mean = np.mean(hist_list, axis=0)
    hist_std = np.std(hist_list, axis=0)
    # bin coordinates
    bin_coordinates = bin_edges[:-1] + bin_distance / 2
    # plot the ISI distribution of the original spike train
    ax.plot(bin_coordinates, isi_distr, label='Original')
    # plot the mean ISI distribution of the UD surrogates
    ax.fill_between(bin_coordinates, hist_mean - hist_std,
                    hist_mean + hist_std, color='lightgrey')
    ax.plot(bin_coordinates, hist_mean, label='UD surrogates', color='grey')
    if legend:
        ax.legend(fontsize=fontsize - 2, loc='upper right')
    # red bar for indication of binsize
    ax.axvline(x=bin_coordinates[5], color='navy', linestyle='--')
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    if show_xlabel:
        ax.set_xlabel(
            'ISI (s)',
            fontsize=fontsize,
            labelpad=XLABELPAD)
    if show_ylabel:
        ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_ylim(-1, 81)


def create_sts_list(sts, sep, epoch_length):
    """
    The function generates a list of spiketrains from the concatenated data,
    where each spiketrain is a list of trials.

    Parameters
    ----------
    sts : list of neo.SpikeTrain
        spiketrains which are concatenated over trials for a
        certain epoch.
    sep: pq.Quantity
        buffer in between concatenated trials
    epoch_length: pq.Quantity
        length of each trial

    Returns
    -------
    sts_list : list of neo.SpikeTrain
        List of spiketrains, where each spiketrain corresponds to a list of
        trials of a certain epoch.
    """
    sts_list = []

    for st in sts:
        single_concatenated_st = create_st_list(spiketrain=st,
                                                sep=sep,
                                                epoch_length=epoch_length)
        sts_list.append(single_concatenated_st)
    return sts_list


def get_cv2(isis):
    """
    Function calculating the CV2 given a list of ISIs extracted from one
    spiketrain. Function from Van Vreiswijk 2010.

    Parameters
    ----------
    isis: list
        list of ISIs
    """
    cv2 = np.sum(
        [2 * np.sum(
            np.abs(trial_isi[:-1] - trial_isi[1:])
            / (trial_isi[:-1] + trial_isi[1:]))
         for trial_isi in isis]
        ) / np.sum([len(trial_isi) - 1 if len(trial_isi) > 0 else 0
                    for trial_isi in isis])
    return cv2


def plot_cv2(sts, ax, epoch_length, sep, show_xlabel=True,
             show_ylabel=True, fontsize=14):
    """
    Function producing the distribution of CV2 of all neurons in the
    considered dataset.

    Parameters
    ----------
    sts: list
        list of spiketrains
    ax: plt.axes
        ax where to plot
    epoch_length: pq.quantities
        length of each trial
    sep: pq.quantities
        separation time between trials
    """
    sts_list = create_sts_list(sts, epoch_length=epoch_length,
                               sep=sep)
    # loop over the neurons
    cv2_list = []
    for conc_st in sts_list:
        isis = [np.diff(st.magnitude)
                for st in conc_st
                if len(st) > 1]
        cv2 = get_cv2(isis)
        cv2_list.append(cv2)

    bin_width = 0.1
    bins = np.arange(0, 2, bin_width)
    ax.hist(cv2_list, bins, alpha=1)
    ax.set_xticks(np.arange(0, 1.9, 0.5))
    ax.set_xlim(0.2, 1.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    if show_xlabel:
        ax.set_xlabel(
            'CV2',
            fontsize=fontsize,
            labelpad=XLABELPAD)
    if show_ylabel:
        ax.set_ylabel('Count', fontsize=fontsize)


def plot_dt(sts, ax, sorting_dead_time, sep, max_refractory=4 * pq.ms,
            show_xlabel=True, show_ylabel=True, fontsize=14.):
    """
    Function producing the distribution of dead times (calculated as minimal
    ISI) of all neurons in the considered dataset.

    Parameters
    ----------
    sts: list
        list of spiketrains
    ax: plt.axes
        ax where to plot
    sorting_dead_time: dict
        dictionary of dead times for all neurons fixed during spike sorting
    sep: pq.Quantity
        separation time between trials
    max_refractory: pq.Quantity, optional
        maximal refractory period as a top boundary
        Default: 4*pq.ms
    show_xlabel : bool, optional
        Default: True
    show_ylabel : bool, optional
        Default: True
    fontsize : float, optional
        Default: 14.
    """
    rp_list = []
    # loop over the neurons
    for st in sts:
        rp = estimate_rate_deadtime(st,
                                    max_refractory=max_refractory,
                                    sampling_period=1 * pq.ms,
                                    sep=sep)[1]
        rp_list.append(rp * 1000)
    rp_array = np.array(rp_list)[~np.isnan(np.array(rp_list))]
    bin_width = 0.1
    sorting_dead_time = sorting_dead_time.rescale(pq.ms).magnitude
    bins = np.arange(0, 4, bin_width)
    ax.set_xticks(np.arange(0, 4, 1))
    ax.hist(rp_array, bins, alpha=1)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.axvline(x=sorting_dead_time, color='grey', linestyle='--')
    if show_xlabel:
        ax.set_xlabel(
            r'd (ms)',
            fontsize=fontsize,
            labelpad=XLABELPAD)
    if show_ylabel:
        ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_xlim(-0.2, 3.5)


def fig_2(folder, sessions, epoch, trialtype, dither, binsize, n_surr,
          winlen, epoch_length, sorting_deadtime, sep, fontsize,
          data_type='original'):
    # big gridspec
    fig = plt.figure(figsize=(5.2, 5.5), dpi=300)
    gsfig = gridspec.GridSpec(
        nrows=3, ncols=1, figure=fig, hspace=0.4, wspace=0.2,
        left=0.12, right=0.97, bottom=0.075, top=0.95,
        height_ratios=(1.25, 1., 0.75))

    # ################### Panel A ####################
    np.random.seed(0)
    # gridspec inside gridspec

    wspace_between_columns = 0.15

    gs_panel_a = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gsfig[0],
        wspace=wspace_between_columns)

    # Nikos
    # loading
    extra_part = f'{data_type}_' if data_type != 'original' else ''
    file_nikos = f'{folder}{sessions[0]}/{extra_part}{epoch}_{trialtype}.npy'
    sts_N = load_processed_spike_trains(
        file_nikos)

    # plotting
    gs0 = gridspec.GridSpecFromSubplotSpec(
        nrows=2, ncols=1,
        subplot_spec=gs_panel_a[0], hspace=0,
        height_ratios=[2, 1])
    ax01 = fig.add_subplot(gs0[0])
    ax02 = fig.add_subplot(gs0[1], sharex=ax01)
    plot_loss_top_panel(ax_loss=ax01,
                        ax_residuals=ax02,
                        sts=sts_N,
                        binsize=binsize,
                        dither=dither * pq.s,
                        n_surr=n_surr,
                        epoch_length=epoch_length,
                        winlen=winlen,
                        fontsize=fontsize)
    # hide ticklabels of first ISI figure
    plt.setp(ax01.get_xticklabels(), visible=False)

    ax01.set_title('Monkey N', fontsize=10)
    plt.figtext(
        x=0.055, y=0.95, s='A', fontsize=12, multialignment='center')
    # Lilou
    sts_L = load_processed_spike_trains(
        f'{folder}{sessions[1]}/{extra_part}{epoch}_{trialtype}.npy')

    gs1 = gridspec.GridSpecFromSubplotSpec(
        nrows=2, ncols=1,
        subplot_spec=gs_panel_a[1], hspace=0,
        height_ratios=[2, 1])

    ax11 = fig.add_subplot(gs1[0])
    ax12 = fig.add_subplot(gs1[1], sharex=ax11)
    plot_loss_top_panel(ax_loss=ax11,
                        ax_residuals=ax12,
                        sts=sts_L,
                        binsize=binsize,
                        dither=dither * pq.s,
                        fontsize=fontsize,
                        n_surr=n_surr,
                        epoch_length=epoch_length,
                        winlen=winlen)
    ax11.set_ylabel('')
    ax11.get_legend().remove()

    # hide ticklabels of first ISI figure
    plt.setp(ax11.get_xticklabels(), visible=False)

    ax12.set_ylabel('')
    ax11.set_title('Monkey L', fontsize=10)

    plt.rcParams.update({'font.size': 10})

    # ################### Panel B & C ####################

    gs_panel_b = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2,
        subplot_spec=gsfig[1],
        wspace=wspace_between_columns,
        hspace=0.8)

    gs_panel_c = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2,
        subplot_spec=gsfig[2],
        wspace=wspace_between_columns,
        hspace=0.8)

    plt.figtext(
        x=0.055, y=0.57, s='B', fontsize=12, multialignment='center')

    # figure Nikos

    # figures ISI distribution of 2 example neurons

    gs00 = gridspec.GridSpecFromSubplotSpec(
        nrows=2, ncols=1,
        subplot_spec=gs_panel_b[0],
        hspace=0)
    ax01 = fig.add_subplot(gs00[0])
    plot_isi_surr(sts_N[10], ax01, dither=dither,
                  show_xlabel=False, fontsize=fontsize,
                  legend=True)

    # hide ticklabels of first ISI figure
    plt.setp(ax01.get_xticklabels(), visible=False)

    ax02 = fig.add_subplot(gs00[1], sharex=ax01)
    plot_isi_surr(sts_N[27], ax02, dither=dither,
                  fontsize=fontsize)

    gs01 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2,
        subplot_spec=gs_panel_c[0],
        wspace=0.35)

    # figure CV2 histogram
    ax03 = fig.add_subplot(gs01[0])
    plot_cv2(
        sts_N, ax03, epoch_length=epoch_length, sep=sep, fontsize=fontsize)

    # figure Dead time histogram
    ax04 = fig.add_subplot(gs01[1])
    plot_dt(sts_N, ax04, sorting_dead_time=sorting_deadtime['N'],
            sep=sep, fontsize=fontsize)
    ax04.set_ylabel('')

    # figure Lilou

    # figures ISI distribution of 2 example neurons

    gs10 = gridspec.GridSpecFromSubplotSpec(
        nrows=2, ncols=1,
        subplot_spec=gs_panel_b[1], hspace=0)

    ax11 = fig.add_subplot(gs10[0])
    plot_isi_surr(sts_L[7], ax11, dither=dither,
                  show_xlabel=False, fontsize=fontsize,
                  legend=False)
    ax11.set_ylabel('')

    # hide ticklabels of first ISI figure
    plt.setp(ax11.get_xticklabels(), visible=False)

    ax12 = fig.add_subplot(gs10[1], sharex=ax11)
    plot_isi_surr(sts_L[8], ax12, dither=dither,
                  fontsize=fontsize)
    ax12.set_ylabel('')

    gs11 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2,
        subplot_spec=gs_panel_c[1],
        wspace=0.35)

    # figure CV2 histogram
    ax13 = fig.add_subplot(gs11[0], sharex=ax03, sharey=ax03)
    plot_cv2(sts_L, ax13, epoch_length=epoch_length, sep=sep,
             fontsize=fontsize)
    ax13.set_ylabel('')

    # figure Dead time histogram
    ax14 = fig.add_subplot(gs11[1], sharex=ax04, sharey=ax04)
    plot_dt(sts_L, ax14, sorting_dead_time=sorting_deadtime['L'],
            sep=sep, fontsize=fontsize)
    ax14.set_ylabel('')

    plt.figtext(
        x=0.055, y=0.25, s='C', fontsize=12, multialignment='center')
    fig.align_ylabels()

    # plt.show()
    fig.savefig('../plots/fig_spikeloss_r2gstats.png')
    # convert manually to eps
    # inkscape fig2_spikeloss_r2gstats.pdf
    # --export-eps=fig2_spikeloss_r2gstats.eps
    fig.savefig('../plots/fig_spikeloss_r2gstats.pdf')


if __name__ == '__main__':
    import yaml
    from yaml import Loader

    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    # The sessions to analyze
    sessions = config['sessions']
    # Magnitude of the binsize used
    binsize = config['binsize'] * pq.s
    # The winlen parameter for the SPADE analysis
    winlen = config['winlen']
    # Data being generated
    processes = config['processes']
    # SNR threshold
    SNR_thresh = config['SNR_thresh']
    # size for removal of synchrofacts from data
    synchsize = config['synchsize']
    # Firing rate threshold to exclude neurons
    firing_rate_threshold = config['firing_rate_threshold']
    # Dithering to use to generate surrogates in seconds
    dither = config['dither']

    epoch = 'movement'
    trialtype = 'PGLF'
    sep = 2 * winlen * binsize
    max_refractory = 4 * pq.ms
    sorting_deadtime = {'N': 1.26666 * pq.ms,
                        'L': 1.6 * pq.ms}
    data_path = '../data/concatenated_spiketrains/'
    epoch_length = 0.5 * pq.s

    data_types = ['original']
    data_types.extend(config['processes'])

    data_type_fixed = 'original'
    epoch_fixed = 'movement'
    trialtype_fixed = 'PGHF'

    n_surr = 100
    fontsize = 9

    for epoch, trialtype, data_type in itertools.product(
            config['epochs'], config['trialtypes'], data_types):
        if not (data_type == data_type_fixed
                and trialtype == trialtype_fixed
                and epoch == epoch_fixed):
            continue

        if data_type == 'original':
            sts_folder = '../data/concatenated_spiketrains/'
        else:
            sts_folder = f'../data/artificial_data/{data_type}/'

        fig_2(sts_folder, sessions, epoch, trialtype, dither, binsize,
              n_surr, winlen, epoch_length, sorting_deadtime, sep,
              fontsize, data_type='original')
