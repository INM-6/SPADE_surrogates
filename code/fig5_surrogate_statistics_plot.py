"""
Plot the figure containing the overview over statistical
features of the different surrogate methods.

It is necessary to first run the data generation script.
"""
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import fig5_surrogate_statistics_config as cf


def plot_clipped_firing_rate(axes_clip):
    """
    This function makes a plot for the clipped firing rate.

    Parameters
    ----------
    axes_clip : Generator of matplotlib.axes.Axes

    Returns
    -------
    None
    """
    results = np.load(
        f'{cf.DATA_PATH}/clipped_rates.npy', allow_pickle=True).item()
    rates = results['rates']
    ratio_clipped = results['ratio_clipped']
    ratio_clipped_surr = results['ratio_clipped_surr']

    for type_id, data_type in enumerate(cf.DATA_TYPES):
        axes_clip[type_id].plot(
            rates[data_type], 1. - np.array(ratio_clipped[data_type]),
            label='original',
            linewidth=cf.ORIGINAL_LINEWIDTH,
            color=cf.COLORS['original'],
            linestyle=cf.LINE_STYLES['original']
        )
        for surr_method in cf.SURR_METHODS:
            axes_clip[type_id].plot(
                rates[data_type],
                1. - np.array(ratio_clipped_surr[data_type][surr_method]),
                label=surr_method,
                color=cf.COLORS[surr_method],
                linestyle=cf.LINE_STYLES[surr_method])

        axes_clip[type_id].set_xlabel(r'$\lambda$ (Hz)',
                                      labelpad=cf.XLABELPAD,)


def _plot_isi(axes_isi, axes_insets, data_type, type_id, surr_method):
    results = np.load(
        f'{cf.DATA_PATH}/isi_{data_type}_{surr_method}.npy',
        allow_pickle=True).item()

    bin_edges = results['bin_edges']
    hist = results['hist']

    linewidth = cf.ORIGINAL_LINEWIDTH if surr_method == 'original' else 1.

    label = cf.LABELS[surr_method] if type_id == 0 else None

    axes_isi[type_id].plot(
        bin_edges[:-1] + bin_edges[0] / 2,
        hist,
        linewidth=linewidth,
        label=label,
        color=cf.COLORS[surr_method],
        linestyle=cf.LINE_STYLES[surr_method])

    axes_insets[type_id].plot(
        bin_edges[:-1] + bin_edges[0] / 2,
        hist,
        linewidth=linewidth,
        label=label,
        color=cf.COLORS[surr_method],
        linestyle=cf.LINE_STYLES[surr_method])


def _plot_ac_cc(axes_ac, axes_cc, type_id, data_type,
                surr_method):
    linewidth = cf.ORIGINAL_LINEWIDTH if surr_method == 'original' else 1.
    for axes, corr_type in zip((axes_ac, axes_cc), ('ac', 'cc')):
        results = np.load(
            f'{cf.DATA_PATH}/{corr_type}_{data_type}_{surr_method}.npy',
            allow_pickle=True).item()

        hist_times = results['hist_times']
        hist = results['hist']

        axes[type_id].plot(
            hist_times,
            hist,
            label=surr_method, linewidth=linewidth, color=cf.COLORS[surr_method],
            linestyle=cf.LINE_STYLES[surr_method])


def _label_axes(axes_isi, axes_ac, axes_cc):
    for axis_isi, axis_cc in zip(axes_isi, axes_cc):
        for axis in (axis_isi, axis_cc):
            axis.set_xlabel(r'$\tau$ (ms)',
                            labelpad=cf.XLABELPAD,)

    for axis_ac, axis_cc in zip(axes_ac, axes_cc):
        axis_ac.set_ylim(bottom=cf.AC_BOTTOM * cf.FIRING_RATE,
                         top=cf.AC_TOP * cf.FIRING_RATE)
        axis_ac.set_xlim(left=-cf.AC_CC_XLIM * cf.DITHER,
                         right=cf.AC_CC_XLIM * cf.DITHER)

    for axis_cc in axes_cc:
        axis_cc.set_ylim(bottom=cf.CC_BOTTOM * cf.FIRING_RATE,
                         top=cf.CC_TOP * cf.FIRING_RATE)
        axis_cc.set_xlim(left=-cf.AC_CC_XLIM * cf.DITHER,
                         right=cf.AC_CC_XLIM * cf.DITHER)
        axis_cc.set_xticks([-cf.DITHER.magnitude, 0., cf.DITHER.magnitude])


def plot_statistical_analysis_of_single_rate(
        axes_isi, axes_ac, axes_cc):
    """
    Plot the ISI-distribution, autocorrelation, cross-correlation for original
    data and its surrogates.

    Parameters
    ----------
    axes_isi : Generator of matplotlib.axes.Axes
    axes_ac : Generator of matplotlib.axes.Axes
    axes_cc : Generator of matplotlib.axes.Axes

    Returns
    -------
    None
    """
    axes_insets = []
    for type_id, data_type in enumerate(cf.DATA_TYPES):
        axes_insets.append(
            inset_axes(axes_isi[type_id], 0.7, 0.4))  # (0.7, 0.5)
        axes_insets[type_id].set_xlim(-0.5, 5.5)        # (0., 7.5)
        axes_insets[type_id].set_ylim(
            0.5 * cf.FIRING_RATE, 1.1*cf.FIRING_RATE)
        axes_insets[type_id].set_xticks([1, 3, 5])
        axes_insets[type_id].tick_params(axis='both',
                                         pad=0.5,
                                         labelsize='x-small')

    for type_id, data_type in enumerate(cf.DATA_TYPES):
        _plot_isi(axes_isi, axes_insets,
                  data_type, type_id, surr_method='original')
        _plot_ac_cc(axes_ac, axes_cc, type_id, data_type,
                    surr_method='original')

        for surr_method in cf.SURR_METHODS:
            _plot_isi(
                axes_isi, axes_insets, data_type, type_id, surr_method)

            _plot_ac_cc(axes_ac, axes_cc, type_id, data_type,
                        surr_method)

    _label_axes(axes_isi, axes_ac, axes_cc)


def plot_firing_rate_change(axis):
    """
    This function creates a plot which shows the change in firing rate profile
    after applying a surrogate method. Starting point is a spike train
    that has the first 75 ms a firing rate of 10 Hz and than of
    80 Hz, chosen similarly to Louis et al. (2010).

    Parameters
    ----------
    axis : matplotlib.axes.Axes

    Returns
    -------
    None
    """
    results = np.load(f'{cf.DATA_PATH}/rate_step.npy', allow_pickle=True).item()

    rate_original = results['original']
    axis.plot(rate_original.times, rate_original.simplified.magnitude,
              label='original', linewidth=cf.ORIGINAL_LINEWIDTH,
              color=cf.COLORS['original'], linestyle=cf.LINE_STYLES['original'])

    for surr_method in cf.SURR_METHODS:
        rate_dithered = results[surr_method]

        axis.plot(rate_dithered.times, rate_dithered.simplified.magnitude,
                  label=surr_method, color=cf.COLORS[surr_method],
                  linestyle=cf.LINE_STYLES[surr_method])

    axis.set_xlabel(r't (ms)',
                    labelpad=cf.XLABELPAD,)
    axis.set_ylabel(r'$\lambda(t)$ (Hz)',
                    labelpad=cf.YLABELPAD,)

    t_stop = cf.DURATION_RATES_STEP.rescale(pq.ms).magnitude
    axis.set_xticks([0., 1/3 * t_stop, 2/3 * t_stop, t_stop])


def plot_eff_moved(axis):
    """
    This function makes a plot showing the ratio between moved spikes over the
    spike count depending on the firing rate.

    Parameters
    ----------
    axis: matplotlib.axes.Axes
        The axis to plot onto

    Returns
    -------
    None
    """
    results = np.load(
        f'{cf.DATA_PATH}/clipped_rates.npy', allow_pickle=True).item()

    rates = results['rates']
    ratio_indep_moved = results['ratio_indep_moved']
    ratio_moved = results['ratio_moved']

    axis.plot(
        rates[cf.STEP_DATA_TYPE], ratio_indep_moved[cf.STEP_DATA_TYPE],
        label='indep.',
        linewidth=cf.ORIGINAL_LINEWIDTH,
        color=cf.COLORS['original'],
        linestyle=cf.LINE_STYLES['original'])
    for surr_method in cf.SURR_METHODS:
        axis.plot(
            rates[cf.STEP_DATA_TYPE],
            ratio_moved[cf.STEP_DATA_TYPE][surr_method],
            label=surr_method,
            color=cf.COLORS[surr_method],
            linestyle=cf.LINE_STYLES[surr_method])

    axis.set_xlabel(r'$\lambda$ (Hz)',
                    labelpad=cf.XLABELPAD,)
    axis.set_ylabel(r'$N_{moved}/N$',
                    labelpad=cf.YLABELPAD2,)

    maximal_rate = cf.RATES[-1]
    axis.set_xticks([1/4 * maximal_rate, 2/4 * maximal_rate,
                     3/4 * maximal_rate, 4/4 * maximal_rate])
    axis.set_yticks([0.5, 0.7, 0.9])


def plot_cv_change(axis):
    """
    Function to plot the relationship of the CV from original data to that of
    the surrogates.

    Parameters
    ----------
    axis: matplotlib.axes.Axes
        The axis to plot onto

    Returns
    -------
    None
    """
    results = np.load(f'{cf.DATA_PATH}/cv_change.npy', allow_pickle=True).item()

    cvs_real = results['cvs_real']
    axis.plot(cvs_real, cvs_real, linewidth=cf.ORIGINAL_LINEWIDTH,
              color=cf.COLORS['original'], linestyle=cf.LINE_STYLES['original'])

    for surr_method in cf.SURR_METHODS:
        cvs_dithered = results[surr_method]
        axis.plot(cvs_real, cvs_dithered, label=surr_method,
                  color=cf.COLORS[surr_method],
                  linestyle=cf.LINE_STYLES[surr_method])
    axis.set_xlabel('CV - original',
                    labelpad=cf.XLABELPAD,)
    axis.set_ylabel('CV - surrogate',
                    labelpad=cf.YLABELPAD,)


def _hide_x_ticks(axis):
    axis.set_xticklabels([])
    axis.tick_params(bottom=False)


def _hide_y_ticks(axis):
    axis.set_yticklabels([])
    axis.tick_params(left=False)


def _dash_spines(axis):
    for direction in ('left', 'right', 'bottom', 'top'):
        axis.spines[direction].set_linewidth(.5)
        axis.spines[direction].set_linestyle((0, (8, 8)))


def plot_statistics_overview():
    """
    This function creates and saves the plot with the overview over the
    statistical features of the surrogate methods.

    Returns
    -------
    None
    """
    plt.rcParams.update(
        {'lines.linewidth': cf.SURROGATES_LINEWIDTH,
         'xtick.labelsize': 'small',
         'axes.labelsize': 'small'})

    fig = plt.figure(figsize=cf.FIGSIZE)
    axes = [[
        fig.add_axes(
            rect=[  # left
                  cf.distance_left_border + cf.width_figure * left_to_right_id,
                    # bottom
                  cf.distance_bottom_border
                  - (top_to_bottom_id-3)
                  * (cf.height_figure + cf.distance_vertical_panels)
                  # sign is only zero in the last case, allowing to have zero
                  # distance between the panels
                  + np.sign(top_to_bottom_id-3) * cf.distance_vertical_panels,
                    # width
                  cf.width_figure,
                    # height
                  cf.height_figure])
        for left_to_right_id in range(3)] for top_to_bottom_id in range(4)]

    right_side_axes = \
        [fig.add_axes(
            rect=[cf.distance_left_border + cf.width_figure * 3
                  + cf.distance_horizontal_panels,  # left
                  cf.distance_bottom_border
                  - (top_to_bottom_id - 2)
                  * (cf.height_side_figure + cf.distance_vertical_panels),  # bottom
                  cf.width_figure,  # width
                  cf.height_side_figure  # height
                  ])
         for top_to_bottom_id in range(3)]

    axes_clip, axes_isi, axes_ac, axes_cc = axes
    axis_cv, axis_moved, axis_step = right_side_axes

    axes_clip[0].set_ylabel(r'$1 - N_{clip}/N$',
                            labelpad=cf.YLABELPAD2,
                            )
    axes_isi[0].set_ylabel(r'$p(\tau)$ (1/s)',
                           labelpad=cf.YLABELPAD,
                           )
    axes_ac[0].set_ylabel('   ACH (1/s)',
                          labelpad=cf.YLABELPAD,
                          )
    axes_cc[0].set_ylabel('CCH (1/s)      ',
                          labelpad=cf.YLABELPAD,
                          )

    plot_clipped_firing_rate(axes_clip)

    plot_statistical_analysis_of_single_rate(
        axes_isi, axes_ac, axes_cc)

    plot_firing_rate_change(axis_step)
    plot_eff_moved(axis_moved)
    plot_cv_change(axis_cv)

    for data_id, data_type in enumerate(cf.DATA_TYPES):
        axes[0][data_id].set_title(data_type)

    for axis_id, axis in enumerate(axes):
        letter_pos_x = -0.25
        letter_pos_y = 1.05 if axis_id < 3 else 0.8
        axis[0].text(
            letter_pos_x, letter_pos_y, cf.LETTERS[axis_id],
            transform=axis[0].transAxes, fontsize=15)
        _hide_y_ticks(axis[1])
        _hide_y_ticks(axis[2])

    for axis_id, axis in enumerate(right_side_axes):
        axis.text(
            -0.15, 1.05, cf.LETTERS[len(axes) + axis_id],
            transform=axis.transAxes, fontsize=15)

    for axis in axes_ac:
        _hide_x_ticks(axis)

    for axes_in_row in axes:
        ylims = [axis.get_ylim() for axis in axes_in_row]
        ylim = (min(ylim[0] for ylim in ylims),
                max(ylim[1] for ylim in ylims))

        for axis in axes_in_row:
            axis.set_ylim(ylim)

    handles, labels = axes_isi[0].get_legend_handles_labels()

    legend = fig.legend(handles, labels, fontsize='x-small',
                        bbox_to_anchor=(0.93,  # x
                                        0.98))  # y
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')

    fig.savefig(f'{cf.PLOT_PATH}/{cf.FIG_NAME}', dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_statistics_overview()