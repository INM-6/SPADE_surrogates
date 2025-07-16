import itertools
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import quantities as pq

import yaml

from generate_artificial_data import get_cv2
from rgutils import load_processed_spike_trains
from SPADE_surrogates.analyse_data_utils.filter_results import load_filtered_results


def create_firing_rate_plots(axes, what_to_plot='rate'):
    xlabel = r'$\lambda$ (Hz)' if what_to_plot == 'rate' else 'CV2'
    color_for_combination = {
        'all': 'C0', 'only UD': 'C1',
        'UD & UDD': 'C2', 'other': 'C3'}

    lines = {'all': None, 'only UD': None,
             'UD & UDD': None, 'other': None}

    for row_id, (axes_row, session) \
            in enumerate(zip(axes, config['sessions'])):

        for ax, process in zip(axes_row, config['processes']):
            ax.set_yticks(
                (len(config['trialtypes']) + 1) * np.arange(len(epoch_labels)))
            ax.set_yticklabels(epoch_labels)

            if row_id == 1:
                ax.set_xlabel(xlabel, fontsize=fontsize_title)
            if what_to_plot == 'rate':
                ax.set_xlim(rate_limits)
                ax.set_xticks(rate_ticks)
            else:
                ax.set_xlim(0.4, 1.6)
            ax.set_title(
                f'{monkeys[session]} - {capitalized_processes[process]}',
                fontsize=fontsize_title, y=yloc_title)

            for bev_id, ((epoch_id, epoch), trialtype) in enumerate(
                    itertools.product(
                        enumerate(config['epochs']), config['trialtypes'])):

                bev_id = bev_id + epoch_id

                spiketrains = load_processed_spike_trains(
                    f'{spiketrain_path}{process}/'
                    f'{session}/{process}_{epoch}_{trialtype}.npy')

                for neuron in excluded_neurons[session]:
                    spiketrains.pop(int(neuron))

                if what_to_plot == 'rate':
                    n_trials = int(
                        (spiketrains[0].t_stop.simplified.item()
                         - spiketrains[0].t_start.simplified.item()) /
                        (epoch_length + sep))

                    effective_length = n_trials * epoch_length

                    rates = np.array(
                        [len(spiketrain) / effective_length
                         for spiketrain in spiketrains])
                    quantity_to_plot = rates
                else:
                    cv2s = np.array(
                        [get_cv2(spiketrain, sep=sep * pq.s)
                         for spiketrain in spiketrains])
                    quantity_to_plot = cv2s

                ax.plot(
                    quantity_to_plot, np.repeat(bev_id, len(quantity_to_plot)),
                    color='grey', marker='o', linestyle='', alpha=alpha,
                    markersize=markersize)

                neurons_per_method = {}
                all_neurons = np.array([])
                for surr_method in surr_methods:
                    filepath = \
                        f'{results_path}{surr_method}/{process}/' \
                        f'{session}/{epoch}_{trialtype}/filtered_res.npy'
                    patterns, _, _, _, _ = load_filtered_results(filepath)

                    if len(patterns) == 0:
                        neurons_per_method[surr_method] = np.array([])
                        continue

                    neurons_per_method[surr_method] = np.unique(
                        np.hstack([pattern['neurons']
                                   for pattern in patterns]))
                    all_neurons = np.unique(np.hstack(
                        (all_neurons, neurons_per_method[surr_method])))

                neurons_per_combination = defaultdict(list)
                for neuron in all_neurons:
                    combination = \
                        [surr_method
                         for surr_method in surr_methods
                         if neuron in neurons_per_method[surr_method]]
                    neurons_per_combination[tuple(combination)].append(neuron)

                for combination, neurons in neurons_per_combination.items():
                    neurons = np.array(neurons).astype(int)
                    if len(combination) == len(surr_methods):
                        combination = 'all'
                    elif len(combination) == 1 and combination[0] == 'ud':
                        combination = 'only UD'
                    elif len(combination) == 2 and \
                            combination[1] == 'udrp' and \
                            combination[0] == 'ud':
                        combination = 'UD & UDD'
                    else:
                        combination = 'other'

                    lines[combination], = ax.plot(
                        quantity_to_plot[neurons],
                        np.repeat(bev_id, len(quantity_to_plot[neurons])),
                        marker='o', linestyle='',
                        markersize=markersize,
                        color=color_for_combination[combination])
    return lines


if __name__ == '__main__':
    spiketrain_path = '../data/artificial_data/'
    results_path = '../results/artificial_data/'

    plot_path = '../plots/fp_firing_distribution/'

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    sep = 2. * config['winlen'] * config['binsize']  # in s
    epoch_length = 0.5  # in s

    markersize = 2.0
    figsize = (5.2, 6.4)
    alpha = 0.15
    rate_limits = (-2., 69.)
    rate_ticks = np.arange(0, 61, 10)

    legend_loc = 10
    yloc_title = 0.95
    fontsize_title = 9.

    surr_methods = ['ud', 'udrp', 'jisi', 'isi', 'bin_shuffling', 'tr_shift']

    excluded_neurons = np.load(
        'analysis_artificial_data/excluded_neurons.npy',
        allow_pickle=True).item()

    epoch_labels = []
    for epoch in config['epochs']:
        if epoch == 'movement':
            epoch = 'mov'
        elif epoch == 'latedelay':
            epoch = 'late'
        elif epoch == 'earlydelay':
            epoch = 'early'
        epoch_labels.append(epoch)

    monkeys = {'i140703-001': 'Monkey N', 'l101210-001': 'Monkey L'}
    capitalized_processes = {'ppd': 'PPD', 'gamma': 'Gamma'}

    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(bottom=0.13, top=0.95, right=0.95, hspace=0.35)
    gs = gridspec.GridSpec(ncols=1, nrows=2,
                           figure=fig)

    # firing rates
    gs0 = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=2,
                                           subplot_spec=gs[0], hspace=0.55,
                                           wspace=0.3)
    ax01 = fig.add_subplot(gs0[0])
    ax02 = fig.add_subplot(gs0[1])
    ax03 = fig.add_subplot(gs0[2])
    ax04 = fig.add_subplot(gs0[3])
    axes_fr = ([ax01, ax02], [ax03, ax04])
    legend_lines_fr = create_firing_rate_plots(
        axes_fr,
        what_to_plot='rate')

    # cv2s
    gs1 = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=2,
                                           subplot_spec=gs[1], hspace=0.55,
                                           wspace=0.3)
    ax11 = fig.add_subplot(gs1[0])
    ax12 = fig.add_subplot(gs1[1])
    ax13 = fig.add_subplot(gs1[2])
    ax14 = fig.add_subplot(gs1[3])
    axes_cv2 = ([ax11, ax12], [ax13, ax14])
    legend_lines_cv2 = create_firing_rate_plots(
        axes_cv2,
        what_to_plot='CV2')
    plt.text(x=-0.25, y=1.05, s='A', transform=ax01.transAxes, fontsize=12)
    plt.text(x=-0.25, y=1.05, s='B', transform=ax11.transAxes, fontsize=12)

    fig.legend(
        list(legend_lines_fr.values()),
        list(legend_lines_fr.keys()),
        fontsize=fontsize_title,
        fancybox=True, shadow=True, ncol=5, loc="lower left",
        mode="expand", borderaxespad=1)
    # plt.show()
    plt.savefig('../plots/fig_fps_characteristics.png', dpi=300)
    plt.savefig('../plots/fig_fps_characteristics.pdf', dpi=300)
