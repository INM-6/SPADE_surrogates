"""
Script for plotting statistical features of different surrogate methods.

This script creates comprehensive visualizations comparing original spike train data
with various surrogate methods, including ISI distributions, autocorrelations,
cross-correlations, firing rate changes, and coefficient of variation analysis.

Prerequisites:
    - Data generation script must be run first to generate required data files
    - Configuration file (fig_surrogate_statistics_config) must be available
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import warnings

import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import fig_surrogate_statistics_config as cf


class SurrogateStatisticsPlotter:
    """
    A class for creating comprehensive statistical plots comparing original and surrogate data.
    
    This class handles loading data, creating various statistical visualizations,
    and managing plot aesthetics for surrogate method comparisons.
    """
    
    def __init__(self, data_path: str = None, plot_path: str = None):
        """
        Initialize the plotter with data and output paths.
        
        Parameters
        ----------
        data_path : str, optional
            Path to data directory, defaults to config value
        plot_path : str, optional
            Path for saving plots, defaults to config value
        """
        self.data_path = Path(data_path or cf.DATA_PATH)
        self.plot_path = Path(plot_path or cf.PLOT_PATH)
        self.plot_path.mkdir(parents=True, exist_ok=True)
        
        # Store configuration for easy access
        self.data_types = cf.DATA_TYPES
        self.surr_methods = cf.SURR_METHODS
        self.colors = cf.COLORS
        self.line_styles = cf.LINE_STYLES
        self.labels = cf.LABELS
        
    def _load_data_safely(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Safely load numpy data file with error handling.
        
        Parameters
        ----------
        filename : str
            Name of the data file to load
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Loaded data dictionary, or None if loading fails
        """
        try:
            file_path = self.data_path / filename
            return np.load(file_path, allow_pickle=True).item()
        except FileNotFoundError:
            warnings.warn(f"Data file not found: {filename}")
            return None
        except Exception as e:
            warnings.warn(f"Error loading {filename}: {e}")
            return None
    
    def _setup_plot_style(self) -> None:
        """Set up matplotlib parameters for consistent plot styling."""
        plt.rcParams.update({
            'lines.linewidth': cf.SURROGATES_LINEWIDTH,
            'xtick.labelsize': 'small',
            'ytick.labelsize': 'small',
            'axes.labelsize': 'small'
        })
    
    def _get_plot_params(self, method: str, is_original: bool = False) -> Dict[str, Any]:
        """
        Get standardized plot parameters for a given method.
        
        Parameters
        ----------
        method : str
            Method name (e.g., 'original', 'dithering', etc.)
        is_original : bool, optional
            Whether this is the original data, by default False
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of plot parameters (color, linestyle, linewidth)
        """
        linewidth = cf.ORIGINAL_LINEWIDTH if is_original else cf.SURROGATES_LINEWIDTH
        
        return {
            'color': self.colors[method],
            'linestyle': self.line_styles[method],
            'linewidth': linewidth
        }


class ClippedRatePlotter(SurrogateStatisticsPlotter):
    """Specialized plotter for clipped firing rate analysis."""
    
    def plot_clipped_firing_rate(self, axes: List[plt.Axes]) -> bool:
        """
        Plot the proportion of non-clipped spikes vs firing rate.
        
        Parameters
        ----------
        axes : List[plt.Axes]
            List of axes for plotting different data types
            
        Returns
        -------
        bool
            True if plotting was successful, False otherwise
        """
        results = self._load_data_safely('clipped_rates.npy')
        if results is None:
            return False
        
        rates = results['rates']
        ratio_clipped = results['ratio_clipped']
        ratio_clipped_surr = results['ratio_clipped_surr']
        
        for type_id, data_type in enumerate(self.data_types):
            # Plot original data
            original_params = self._get_plot_params('original', is_original=True)
            axes[type_id].plot(
                rates[data_type], 
                1.0 - np.array(ratio_clipped[data_type]),
                label='original',
                **original_params
            )
            
            # Plot surrogate methods
            for surr_method in self.surr_methods:
                surr_params = self._get_plot_params(surr_method)
                axes[type_id].plot(
                    rates[data_type],
                    1.0 - np.array(ratio_clipped_surr[data_type][surr_method]),
                    label=self.labels[surr_method],
                    **surr_params
                )
            
            axes[type_id].set_xlabel(r'$\lambda$ (Hz)', labelpad=cf.XLABELPAD)
        
        return True
    
    def create_clipped_rate_figure(self) -> None:
        """Create and save the clipped rate figure."""
        fig, axes_clip = plt.subplots(
            1, 3, sharex='all', sharey='all', figsize=(6.5, 2.0),
            gridspec_kw=dict(wspace=0.1, bottom=0.35, top=0.89, right=0.99, left=0.09)
        )
        
        success = self.plot_clipped_firing_rate(axes_clip)
        if not success:
            warnings.warn("Failed to create clipped rate plot - missing data")
            return
        
        # Set labels and formatting
        axes_clip[0].set_ylabel(r'$1 - N_{clip}/N$', labelpad=0.5, fontsize='small')
        
        for data_id, data_type in enumerate(self.data_types):
            axes_clip[data_id].set_title(data_type, fontsize=10)
            axes_clip[data_id].tick_params(axis='both', labelsize='small')
            axes_clip[data_id].set_xlabel(r'$\lambda$ (Hz)', labelpad=-0.5, fontsize='small')
        
        # Add legend
        handles, labels = axes_clip[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize='small', fancybox=True, ncol=7,
                  loc="lower left", mode="expand", borderaxespad=0.5)
        
        # Save figure
        output_files = [
            self.plot_path / f'{cf.FIG_NAME}',
            self.plot_path / 'clipped_rates.eps',
            self.plot_path / 'clipped_rates.svg'
        ]
        
        for output_file in output_files:
            fig.savefig(output_file, dpi=300)
        
        plt.show()


class ISIPlotter(SurrogateStatisticsPlotter):
    """Specialized plotter for Inter-Spike Interval distributions."""
    
    def _plot_isi_for_method(self, axes_isi: List[plt.Axes], axes_insets: List[plt.Axes],
                            data_type: str, type_id: int, surr_method: str) -> bool:
        """
        Plot ISI distribution for a specific method and data type.
        
        Parameters
        ----------
        axes_isi : List[plt.Axes]
            Main ISI plot axes
        axes_insets : List[plt.Axes]
            Inset plot axes
        data_type : str
            Type of data being plotted
        type_id : int
            Index of the data type
        surr_method : str
            Surrogate method name
            
        Returns
        -------
        bool
            True if plotting was successful, False otherwise
        """
        filename = f'isi_{data_type}_{surr_method}.npy'
        results = self._load_data_safely(filename)
        if results is None:
            return False
        
        bin_edges = results['bin_edges']
        hist = results['hist']
        
        # Calculate bin centers
        bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        
        # Get plot parameters
        is_original = (surr_method == 'original')
        plot_params = self._get_plot_params(surr_method, is_original)
        
        # Only show label for first subplot to avoid duplicate legend entries
        label = self.labels[surr_method] if type_id == 0 else None
        plot_params['label'] = label
        
        # Plot on both main and inset axes
        axes_isi[type_id].plot(bin_centers, hist, **plot_params)
        axes_insets[type_id].plot(bin_centers, hist, **plot_params)
        
        return True


class CorrelationPlotter(SurrogateStatisticsPlotter):
    """Specialized plotter for autocorrelation and cross-correlation analysis."""
    
    def _plot_correlation(self, axes_corr: List[plt.Axes], type_id: int, 
                         data_type: str, surr_method: str, corr_type: str = 'ac') -> bool:
        """
        Plot correlation function (autocorrelation or cross-correlation).
        
        Parameters
        ----------
        axes_corr : List[plt.Axes]
            Correlation plot axes
        type_id : int
            Index of the data type
        data_type : str
            Type of data being plotted
        surr_method : str
            Surrogate method name
        corr_type : str, optional
            Type of correlation ('ac' for auto, 'cc' for cross), by default 'ac'
            
        Returns
        -------
        bool
            True if plotting was successful, False otherwise
        """
        filename = f'{corr_type}_{data_type}_{surr_method}.npy'
        results = self._load_data_safely(filename)
        if results is None:
            return False
        
        hist_times = results['hist_times']
        hist = results['hist']
        
        # Get plot parameters
        is_original = (surr_method == 'original')
        plot_params = self._get_plot_params(surr_method, is_original)
        
        axes_corr[type_id].plot(hist_times, hist, **plot_params)
        return True


class RateAnalysisPlotter(SurrogateStatisticsPlotter):
    """Specialized plotter for firing rate analysis."""
    
    def plot_firing_rate_change(self, axis: plt.Axes) -> bool:
        """
        Plot firing rate profile changes after applying surrogate methods.
        
        Parameters
        ----------
        axis : plt.Axes
            Axis for plotting
            
        Returns
        -------
        bool
            True if plotting was successful, False otherwise
        """
        results = self._load_data_safely('rate_step.npy')
        if results is None:
            return False
        
        # Plot original rate
        rate_original = results['original']
        original_params = self._get_plot_params('original', is_original=True)
        axis.plot(
            rate_original.times,
            rate_original.simplified.magnitude,
            label='original',
            **original_params
        )
        
        # Plot surrogate rates
        for surr_method in self.surr_methods:
            if surr_method in results:
                rate_dithered = results[surr_method]
                surr_params = self._get_plot_params(surr_method)
                axis.plot(
                    rate_dithered.times,
                    rate_dithered.simplified.magnitude,
                    label=surr_method,
                    **surr_params
                )
        
        # Set labels and ticks
        axis.set_xlabel(r't (ms)', labelpad=cf.XLABELPAD)
        axis.set_ylabel(r'$\lambda(t)$ (Hz)', labelpad=cf.YLABELPAD)
        
        t_stop = cf.DURATION_RATES_STEP.rescale(pq.ms).magnitude
        axis.set_xticks([0., t_stop/3, 2*t_stop/3, t_stop])
        
        return True
    
    def plot_spike_movement_efficiency(self, axis: plt.Axes) -> bool:
        """
        Plot the ratio of moved spikes vs spike count as function of firing rate.
        
        Parameters
        ----------
        axis : plt.Axes
            Axis for plotting
            
        Returns
        -------
        bool
            True if plotting was successful, False otherwise
        """
        results = self._load_data_safely('clipped_rates.npy')
        if results is None:
            return False
        
        rates = results['rates']
        ratio_indep_moved = results['ratio_indep_moved']
        ratio_moved = results['ratio_moved']
        
        # Plot independent shuffling baseline
        original_params = self._get_plot_params('original', is_original=True)
        axis.plot(
            rates[cf.STEP_DATA_TYPE],
            ratio_indep_moved[cf.STEP_DATA_TYPE],
            label='indep.',
            **original_params
        )
        
        # Plot surrogate methods
        for surr_method in self.surr_methods:
            surr_params = self._get_plot_params(surr_method)
            axis.plot(
                rates[cf.STEP_DATA_TYPE],
                ratio_moved[cf.STEP_DATA_TYPE][surr_method],
                label=surr_method,
                **surr_params
            )
        
        # Set labels and ticks
        axis.set_xlabel(r'$\lambda$ (Hz)', labelpad=cf.XLABELPAD)
        axis.set_ylabel(r'$N_{moved}/N$', labelpad=cf.YLABELPAD2)
        
        max_rate = cf.RATES[-1]
        axis.set_xticks([max_rate/4, max_rate/2, 3*max_rate/4, max_rate])
        axis.set_yticks([0.5, 0.7, 0.9])
        
        return True


class CVAnalysisPlotter(SurrogateStatisticsPlotter):
    """Specialized plotter for Coefficient of Variation analysis."""
    
    def plot_cv_change(self, axis: plt.Axes) -> bool:
        """
        Plot the relationship between original and surrogate CV values.
        
        Parameters
        ----------
        axis : plt.Axes
            Axis for plotting
            
        Returns
        -------
        bool
            True if plotting was successful, False otherwise
        """
        results = self._load_data_safely('cv_change.npy')
        if results is None:
            return False
        
        cvs_real = results['cvs_real']
        
        # Plot identity line (original data)
        original_params = self._get_plot_params('original', is_original=True)
        axis.plot(cvs_real, cvs_real, **original_params)
        
        # Plot surrogate methods
        for surr_method in self.surr_methods:
            if surr_method in results:
                cvs_dithered = results[surr_method]
                surr_params = self._get_plot_params(surr_method)
                axis.plot(cvs_real, cvs_dithered, label=surr_method, **surr_params)
        
        # Set labels
        axis.set_xlabel('CV - original', labelpad=cf.XLABELPAD)
        axis.set_ylabel('CV - surrogate', labelpad=cf.YLABELPAD)
        
        return True


class ComprehensivePlotter(SurrogateStatisticsPlotter):
    """Main plotter class that combines all specialized plotters."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize specialized plotters
        self.isi_plotter = ISIPlotter(self.data_path, self.plot_path)
        self.corr_plotter = CorrelationPlotter(self.data_path, self.plot_path)
        self.rate_plotter = RateAnalysisPlotter(self.data_path, self.plot_path)
        self.cv_plotter = CVAnalysisPlotter(self.data_path, self.plot_path)
    
    def _create_inset_axes(self, axes_isi: List[plt.Axes]) -> List[plt.Axes]:
        """Create inset axes for ISI plots."""
        axes_insets = []
        for type_id, _ in enumerate(self.data_types):
            width, height = 0.7, 0.3
            inset = inset_axes(axes_isi[type_id], width, height)
            inset.set_xlim(-0.5, 5.5)
            inset.set_ylim(0.5 * cf.FIRING_RATE, 1.1 * cf.FIRING_RATE)
            inset.set_xticks([1, 3, 5])
            inset.tick_params(axis='both', pad=0.5, labelsize='x-small')
            axes_insets.append(inset)
        return axes_insets
    
    def _setup_correlation_axes(self, axes_ac: List[plt.Axes], axes_cc: List[plt.Axes]) -> None:
        """Configure autocorrelation and cross-correlation axes."""
        for axis_ac in axes_ac:
            axis_ac.set_ylim(bottom=cf.AC_BOTTOM * cf.FIRING_RATE,
                            top=cf.AC_TOP * cf.FIRING_RATE)
            axis_ac.set_xlim(left=-cf.AC_CC_XLIM * cf.DITHER,
                            right=cf.AC_CC_XLIM * cf.DITHER)
            axis_ac.set_xticks([-40, -20, 0, 20, 40])
        
        for axis_cc in axes_cc:
            axis_cc.set_ylim(bottom=cf.CC_BOTTOM * cf.FIRING_RATE,
                            top=cf.CC_TOP * cf.FIRING_RATE)
            axis_cc.set_xlim(left=-cf.AC_CC_XLIM * cf.DITHER,
                            right=cf.AC_CC_XLIM * cf.DITHER)
            axis_cc.set_xticks([-40, -20, 0, 20, 40])
        
        # Set common x-axis labels
        for axis_isi, axis_ac, axis_cc in zip(axes_ac, axes_ac, axes_cc):
            for axis in (axis_isi, axis_ac, axis_cc):
                axis.set_xlabel(r'$\tau$ (ms)', labelpad=cf.XLABELPAD)
    
    def _hide_axis_labels(self, axes: List[List[plt.Axes]]) -> None:
        """Hide redundant axis labels to clean up the plot."""
        for axis_row in axes:
            for axis in axis_row[1:]:  # Hide y-labels for non-leftmost plots
                axis.set_yticklabels([])
                axis.tick_params(left=False)
    
    def _synchronize_y_limits(self, axes: List[List[plt.Axes]]) -> None:
        """Synchronize y-limits across subplot rows."""
        for axes_row in axes:
            ylims = [axis.get_ylim() for axis in axes_row]
            unified_ylim = (
                min(ylim[0] for ylim in ylims),
                max(ylim[1] for ylim in ylims)
            )
            for axis in axes_row:
                axis.set_ylim(unified_ylim)
    
    def plot_statistical_analysis(self, axes_isi: List[plt.Axes],
                                axes_ac: List[plt.Axes],
                                axes_cc: List[plt.Axes]) -> bool:
        """
        Plot comprehensive statistical analysis including ISI, autocorr, and cross-corr.
        
        Parameters
        ----------
        axes_isi : List[plt.Axes]
            ISI distribution axes
        axes_ac : List[plt.Axes]
            Autocorrelation axes
        axes_cc : List[plt.Axes]
            Cross-correlation axes
            
        Returns
        -------
        bool
            True if all plots were successful, False otherwise
        """
        # Create inset axes for ISI plots
        axes_insets = self._create_inset_axes(axes_isi)
        
        success_count = 0
        total_plots = len(self.data_types) * (1 + len(self.surr_methods)) * 3  # ISI, AC, CC
        
        for type_id, data_type in enumerate(self.data_types):
            # Plot original data
            if self.isi_plotter._plot_isi_for_method(axes_isi, axes_insets, data_type, type_id, 'original'):
                success_count += 1
            if self.corr_plotter._plot_correlation(axes_cc, type_id, data_type, 'original', 'cc'):
                success_count += 1
            if self.corr_plotter._plot_correlation(axes_ac, type_id, data_type, 'original', 'ac'):
                success_count += 1
            
            # Plot surrogate methods
            for surr_method in self.surr_methods:
                if self.isi_plotter._plot_isi_for_method(axes_isi, axes_insets, data_type, type_id, surr_method):
                    success_count += 1
                if self.corr_plotter._plot_correlation(axes_cc, type_id, data_type, surr_method, 'cc'):
                    success_count += 1
                if self.corr_plotter._plot_correlation(axes_ac, type_id, data_type, surr_method, 'ac'):
                    success_count += 1
        
        # Configure axes
        self._setup_correlation_axes(axes_ac, axes_cc)
        
        return success_count == total_plots
    
    def create_comprehensive_overview(self) -> None:
        """Create and save the comprehensive statistical overview figure."""
        self._setup_plot_style()
        
        # Create figure and axes
        fig = plt.figure(figsize=cf.FIGSIZE, dpi=300)
        
        # Create main subplot grid
        axes = []
        for row in range(3):
            row_axes = []
            for col in range(3):
                left = cf.distance_left_border + cf.width_figure * col
                bottom = (cf.distance_bottom_border - 
                         (row - 2) * (cf.height_figure + cf.distance_vertical_panels))
                
                ax = fig.add_axes([left, bottom, cf.width_figure, cf.height_figure])
                row_axes.append(ax)
            axes.append(row_axes)
        
        # Create right-side axes
        right_axes = []
        for row in range(3):
            left = (cf.distance_left_border + cf.width_figure * 3 + 
                   cf.distance_horizontal_panels)
            bottom = (cf.distance_bottom_border - 
                     (row - 2) * (cf.height_figure + cf.distance_vertical_panels))
            
            ax = fig.add_axes([left, bottom, cf.width_figure, cf.height_figure])
            right_axes.append(ax)
        
        axes_isi, axes_cc, axes_ac = axes
        axis_cv, axis_moved, axis_step = right_axes
        
        # Set y-axis labels
        axes_isi[0].set_ylabel(r'$p(\tau)$ (1/s)', labelpad=cf.YLABELPAD)
        axes_ac[0].set_ylabel('   ACH (1/s)', labelpad=cf.YLABELPAD)
        axes_cc[0].set_ylabel('CCH (1/s)      ', labelpad=cf.YLABELPAD)
        
        # Generate all plots
        analysis_success = self.plot_statistical_analysis(axes_isi, axes_ac, axes_cc)
        rate_success = self.rate_plotter.plot_firing_rate_change(axis_step)
        moved_success = self.rate_plotter.plot_spike_movement_efficiency(axis_moved)
        cv_success = self.cv_plotter.plot_cv_change(axis_cv)
        
        if not all([analysis_success, rate_success, moved_success, cv_success]):
            warnings.warn("Some plots failed due to missing data")
        
        # Add titles and labels
        for data_id, data_type in enumerate(self.data_types):
            axes[0][data_id].set_title(data_type, fontsize=10)
        
        # Add subplot labels (A, B, C, etc.)
        for axis_id, axis_row in enumerate(axes):
            axis_row[0].text(-0.25, 1.05, cf.LETTERS[axis_id],
                           transform=axis_row[0].transAxes, fontsize=12)
        
        for axis_id, axis in enumerate(right_axes):
            axis.text(-0.15, 1.05, cf.LETTERS[len(axes) + axis_id],
                     transform=axis.transAxes, fontsize=12)
        
        # Clean up axes
        self._hide_axis_labels(axes)
        self._synchronize_y_limits(axes)
        
        # Add legend
        handles, labels = axes_isi[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize='small', fancybox=True, shadow=True,
                  ncol=7, loc="lower left", mode="expand", borderaxespad=1)
        
        # Save figure
        output_path = self.plot_path / cf.FIG_NAME
        fig.savefig(output_path, dpi=300)
        plt.show()


def main():
    """
    Main function to create all surrogate statistics plots.
    """
    try:
        # Create comprehensive plotter
        plotter = ComprehensivePlotter()
        
        # Create clipped rate figure
        print("Creating clipped rate figure...")
        clipped_plotter = ClippedRatePlotter(plotter.data_path, plotter.plot_path)
        clipped_plotter.create_clipped_rate_figure()
        
        # Create comprehensive overview
        print("Creating comprehensive statistical overview...")
        plotter.create_comprehensive_overview()
        
        print("All plots created successfully!")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        raise


if __name__ == '__main__':
    main()