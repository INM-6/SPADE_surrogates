"""
Script to create the figure that shows the patterns shown in the experimental
data.
"""
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
import yaml
from yaml import Loader
from SPADE_surrogates.analyse_data_utils.filter_results import load_filtered_results


def _load_results(results, surrogate, session_name, epoch, tt):
    """
    Loading results from corresponding folder
    """
    directory = f'../results/experimental_data/{surrogate}/' \
        f'{results}/{session_name}/{epoch}_{tt}'
    filepath = directory + '/filtered_res.npy'
    
    # Use the new loading function from filter_results module
    # This returns all components, but we only need patterns
    patterns, _, _, _, _ = load_filtered_results(filepath)
    
    return patterns


def create_dictionaries_statistics(surrogate, results, sessions, trialtypes,
                                   epoch_tags, binsize, winlen):
    """
    Function creating numerous histograms of important features of results.
    Histograms are stored in dictionaries and calculated across sessions,
    epochs and trialtypes.

    Parameters
    ----------
    surrogate: int
        surrogate technique used
    results: list
        list of patterns detected in a particular session and behavioral
        context
    sessions: list
        list of strings corresponding to the analyzed sessions
    trialtypes: list
        list of strings of the trialtypes of the experiment
    epoch_tags: list
        list of strings of the epochs of one trial of the experiment
    binsize: pq.quantity
        binsize of the SPADE analysis
    winlen: int
        window length of the SPADE analysis

    Returns
    -------
    num_patt, lags, num_spikes, length, max_patt_per_ep, hist_spikes, \
           hist_lags
    num_patt: dict
        dictionary of number of patterns detected across sessions, epochs and
        trial types
    lags: dict
        dictionary of lags detected across sessions, epochs and
        trial types
    num_spikes: dict
        dictionary of pattern sizes across sessions, epochs and trialtypes
    length: dict
        dictionary of pattern durations across sessions, epochs and trialtypes
    max_patt_per_ep: int
        maximum number of patterns across epochs (to find ylim for plotting)
    hist_spikes: np.hist
        histogram of pattern sizes across sessions, epochs and trialtypes
    hist_lags: np.hist
        histogram of pattern lags across sessions, epochs and trialtypes

    """
    # Initializing the total maximum number of pattern per epoch
    max_patt_per_ep = 0

    # Initializing Dictionaries for the statistics
    num_patt = {}
    num_spikes = {}
    lags = {}
    length = {}
    # Initialize histograms
    hist_lags = {}
    hist_spikes = {}

    for session_name in sessions:

        # Loading patt time histogram
        num_patt[session_name] = {}
        num_spikes[session_name] = {}
        lags[session_name] = {}
        length[session_name] = {}

        # Initialize histograms
        hist_lags[session_name] = 0
        hist_spikes[session_name] = 0

        # Loop across all epochs
        for epoch_id, epoch in enumerate(epoch_tags):
            # Initializing Dictionaries for the statistics for each epoch
            num_patt[session_name][epoch] = 0
            num_spikes[session_name][epoch] = []
            lags[session_name][epoch] = []
            length[session_name][epoch] = []

            # Loops across all trialtypes
            for tt in trialtypes:
                if epoch_id == 0:
                    # Initializing Dictionaries for the statistics
                    # for one trialtype
                    num_patt[session_name][tt] = 0
                    num_spikes[session_name][tt] = []
                    lags[session_name][tt] = []
                    length[session_name][tt] = []

                # Loading the filtered results
                behav_context = epoch + '_' + tt
                num_patt[session_name][behav_context] = 0
                num_spikes[session_name][behav_context] = []
                lags[session_name][behav_context] = []
                length[session_name][behav_context] = []

                patterns = _load_results(surrogate=surrogate,
                                         results=results,
                                         session_name=session_name,
                                         epoch=epoch,
                                         tt=tt)

                # Appending the number of patterns
                num_patt[session_name][behav_context] = len(patterns)
                num_patt[session_name][epoch] += len(patterns)
                num_patt[session_name][tt] += len(patterns)

                # Loop across all the patterns to collect all the statistics
                for patt in patterns:
                    for label in [behav_context, epoch, tt]:
                        num_spikes[session_name][label].append(
                            len(patt['neurons']))
                        lags[session_name][label] = np.hstack((
                            lags[session_name][label],
                            patt['lags'].rescale(pq.ms))) * pq.ms
                        length[session_name][label] = np.hstack((
                            length[session_name][label],
                            np.max(patt['lags']))) * pq.s
            # Updating the maximum number of patterns for ylim in plot
            max_patt_per_ep = max(max_patt_per_ep,
                                  num_patt[session_name][epoch])

            # Making histogram of lags for all epochs
            if len(lags[session_name][epoch]) > 0:
                hist_lags[session_name] += np.histogram(
                    lags[session_name][epoch],
                    bins=np.arange(
                        0, (winlen + 1) * binsize.magnitude, binsize.magnitude)
                )[0]
                hist_spikes[session_name] += np.histogram(
                    num_spikes[session_name][epoch], bins=np.arange(0, 10))[0]

    return (num_patt, lags, num_spikes, length, max_patt_per_ep, hist_spikes,
            hist_lags)


def plot_experimental_data_results(surrogates, tag_surrogates,
                                   sessions, trialtypes,
                                   epoch_tags, epoch_tags_short,
                                   binsize, winlen):
    """
    Function reproducing results of the analysis of experimental data. Results
    for two sessions of experimental data described in Brochier et al. 2018:
    session i140703-001 (left) and session l101210-001 (right).
    Each trial is segmented into six behavioral epochs, and then concatenated
    one trial after the other for each combination (out of four) of task
    grips, leading to 24 datasets per session analyzed separately with SPADE.
    Histograms represent the number of significant patterns detected by SPADE
    in each epoch (start, cue, early-delay, late-delay, movement and hold),
    color code according to the grip type (precision/side grip and low/high
    force).

    Parameters
    ----------
    surrogates: list
        list of strings of surrogate names as saved in the results
    sessions: list
        sessions being analyzed
    trialtypes: list of strings
        trialtypes taken into consideration
    epoch_tags: list
        list of strings of names of epochs of the experiment
    epoch_tags_short: list
        list of abbreviation of epoch tags
    binsize: pq.quantity
        binsize of the spade analysis
    winlen: int
        window length of the spade analysis
    """
    # Plotting the averaged statistics
    # Plotting parameters
    label_size = 8
    title_size = 10
    tick_size = 8
    # Initializing the figure
    fig_mean_stat = plt.figure(figsize=(4, 5.5), dpi=300)
    fig_mean_stat.subplots_adjust(
        left=.15,
        right=.98,
        wspace=0,
        hspace=0.1,
        top=.95,
        bottom=.09)
    tt_colors = ['navy',
                 'dodgerblue',
                 'limegreen',
                 'hotpink']
    num_subplots = 1
    for index, surrogate in enumerate(surrogates):
        results = 'results_' + surrogate
        num_patt = create_dictionaries_statistics(
            surrogate=surrogate,
            results=results,
            sessions=sessions,
            trialtypes=trialtypes,
            epoch_tags=epoch_tags,
            binsize=binsize,
            winlen=winlen)[0]
        for session_index, session_name in enumerate(sessions):
            # Dictionary to keep track of which trial type
            # has already been plotted
            label_already_assigned = {tt: False for tt in trialtypes}

            # Panel A
            # Initializing the axis for number of patterns
            ax_num_patt = fig_mean_stat.add_subplot(len(surrogates),
                                                    len(sessions),
                                                    num_subplots)
            # Loop to collect and plot number of pattern for each epoch and
            # trialtype
            for epoch_id, epoch in enumerate(epoch_tags):
                # Initializing height to start the bar plot
                bar_bottom = 0
                for tt_idx, tt in enumerate(trialtypes):
                    behav_context = epoch + '_' + tt
                    # Keep track that this behavioral context has been plotted
                    if not label_already_assigned[tt]:
                        label = tt
                        label_already_assigned[tt] = True
                    else:
                        label = None
                    # Bar plot of the number of patterns
                    # (stacked bar for the same epoch but different trial type)
                    if session_index == 0:
                        ax_num_patt.bar(epoch_id + 1,
                                        num_patt[session_name][behav_context],
                                        width=0.5, color=tt_colors[tt_idx],
                                        bottom=bar_bottom)
                    else:
                        ax_num_patt.bar(epoch_id + 1,
                                        num_patt[session_name][behav_context],
                                        width=0.5, color=tt_colors[tt_idx],
                                        bottom=bar_bottom, label=label)
                    # Updating height of the bar to stack number of patterns
                    bar_bottom += num_patt[session_name][behav_context]

            if index == 0:
                if session_name == 'l101210-001':
                    ax_num_patt.set_title(
                        'Monkey L\nsession l101210-001', size=title_size)
                    ax_num_patt.legend(loc=2, fontsize=7)
                if session_name == 'i140703-001':
                    ax_num_patt.set_title(
                        'Monkey N\nsession i140703-001', size=title_size)

            num_subplots += 1

            # Fixing axis parameter
            if index == 0:
                ax_num_patt.set_ylim([0, 145])
            else:
                ax_num_patt.set_ylim([0, 17])
            ax_num_patt.set_xlim([0, len(epoch_tags) + 1])
            if session_index == 0:
                ax_num_patt.set_ylabel('Patt count', size=label_size)
                if index != 0:
                    ax_num_patt.set_yticks(np.arange(0, 20, 5))
            else:
                ax_num_patt.set_yticks([])
                ax_num_patt.set_yticklabels('')
            # ax_num_patt.ticklabel_format(fontsize=tick_size)
            if index == len(surrogates) - 1:
                ax_num_patt.set_xticks(range(1, len(epoch_tags_short) + 1))
                ax_num_patt.set_xticklabels(epoch_tags_short, rotation=45,
                                            size=tick_size)
            else:
                ax_num_patt.set_xticks([])
                ax_num_patt.set_xticklabels('')
            ax_num_patt.tick_params(axis='both', which='major',
                                    labelsize=tick_size)
            ax_num_patt.text(0.50, 0.92, tag_surrogates[index],
                             horizontalalignment='center',
                             verticalalignment='top',
                             transform=ax_num_patt.transAxes,
                             fontsize=8)
    fig_mean_stat.align_ylabels()

    fig_mean_stat.savefig('../../figures/'
                          '/fig_experimental_data.eps', dpi=300)
    fig_mean_stat.savefig('../../figures/'
                          '/fig_experimental_data.png', dpi=300)


def demonstrate_loading_all_data():
    """
    Example function showing how to load all saved data components.
    This can be used when you need access to parameters and metadata.
    """
    # Example path
    example_path = '../results/experimental_data/surrogate/session/context/filtered_res.npy'
    
    # Load all components
    patterns, loading_params, filter_params, config_params, spade_params = \
        load_filtered_results(example_path)
    
    print(f"Loaded {len(patterns)} patterns")
    print(f"Configuration parameters: {config_params}")
    print(f"Filter parameters: {filter_params}")
    print(f"SPADE parameters: {spade_params}")
    print(f"Loading parameters: {loading_params}")
    
    return patterns, loading_params, filter_params, config_params, spade_params


if __name__ == "__main__":
    # Loading general parameters
    with open("./configfile.yaml", 'r') as stream:
        param = yaml.load(stream, Loader=Loader)
    binsize = (param['binsize'] * pq.s).rescale(pq.ms)
    winlen = param['winlen']
    # The 5 epochs to analyze
    epoch_tags = param['epochs']
    epoch_tags_short = ['start', 'cue', 'earl-d', 'late-d', 'mov', 'hold']
    # The 4 trial types to analyze
    trialtypes = param['trialtypes']
    # The sessions to analyze
    sessions = param['sessions']
    surrogates = ['dither_spikes',
                  'dither_spikes_with_refractory_period',
                  'joint_isi_dithering',
                  'isi_dithering'
                  'trial_shifting',
                  'bin_shuffling']
    tag_surrogates = ['UD', 'UDD', 'JISI-D', 'ISI-D', 'TR-SHIFT', 'BIN-SHUFF']
    plot_experimental_data_results(surrogates=surrogates,
                                   tag_surrogates=tag_surrogates,
                                   sessions=sessions,
                                   trialtypes=trialtypes,
                                   epoch_tags=epoch_tags,
                                   epoch_tags_short=epoch_tags_short,
                                   binsize=binsize,
                                   winlen=winlen)