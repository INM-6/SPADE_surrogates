"""
Modern Script for plotting statistical features of different surrogate methods.

This script creates comprehensive visualizations comparing original spike train data
with various surrogate methods, using modern matplotlib styling optimized for DIN A4.

Prerequisites:
    - Data generation script must be run first to generate required data files
    - Configuration file (fig_surrogate_statistics_config) must be available
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import quantities as pq
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import fig_surrogate_statistics_config as cf

# Configure matplotlib for modern appearance and avoid Type 3 fonts
plt.style.use('seaborn-v0_8-whitegrid')

# Comprehensive Type 3 font avoidance
mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts in PDF
mpl.rcParams['ps.fonttype'] = 42   # TrueType fonts in PostScript
mpl.rcParams['svg.fonttype'] = 'none'  # Embed fonts in SVG

# Font configuration
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['mathtext.fontset'] = 'dejavusans'  # Use DejaVu for math text
mpl.rcParams['mathtext.default'] = 'regular'

# Axes and grid styling
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linewidth'] = 0.5

# Text rendering
mpl.rcParams['text.usetex'] = False  # Don't use LaTeX (can cause Type 3 fonts)
mpl.rcParams['axes.unicode_minus'] = False  # Use ASCII minus sign

# DIN A4 landscape image dimensions (maintain original aspect ratio)
DIN_A4_WIDTH = 10.5   # inches (11.7" - margins)
DIN_A4_HEIGHT = 7.0   # inches (maintain ~1.5:1 aspect ratio)


class CleanModernSurrogateStatisticsPlotter:
    """
    A clean, modern class for creating comprehensive statistical plots comparing 
    original and surrogate data with contemporary styling.
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
        self.surr_methods = list(cf.SURR_METHODS)  # Convert to list immediately
        
        # Use original colors and styles from config
        self.colors = getattr(cf, 'COLORS', {})
        self.line_styles = getattr(cf, 'LINE_STYLES', {})
        self.labels = getattr(cf, 'LABELS', {})
        
        # Set up modern styling
        self._setup_modern_style()
        
    def _setup_modern_style(self) -> None:
        """Set up modern matplotlib parameters for consistent plot styling."""
        plt.rcParams.update({
            'lines.linewidth': getattr(cf, 'SURROGATES_LINEWIDTH', 1.5),
            'lines.markersize': 4,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'figure.titlesize': 13,
            'axes.linewidth': 1.0,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'axes.edgecolor': '#2E3440',
            'text.color': '#2E3440',
            'axes.labelcolor': '#2E3440',
            'xtick.color': '#2E3440',
            'ytick.color': '#2E3440'
        })
    
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
    
    def _get_plot_params(self, method: str, is_original: bool = False) -> Dict[str, Any]:
        """
        Get standardized plot parameters for a given method with stronger colors.
        
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
        linewidth = getattr(cf, 'ORIGINAL_LINEWIDTH', 2.0) if is_original else getattr(cf, 'SURROGATES_LINEWIDTH', 1.5)
        
        # Get color and enhance it for better visibility
        color = self.colors.get(method, '#2E3440')
        if not is_original and color != '#2E3440':
            # Make colors stronger/more saturated for better distinction
            try:
                rgb = mcolors.to_rgb(color)
                hsv = mcolors.rgb_to_hsv(rgb)
                # Increase saturation and adjust brightness
                hsv = (hsv[0], min(1.0, hsv[1] * 1.3), max(0.3, hsv[2] * 0.9))
                color = mcolors.hsv_to_rgb(hsv)
            except:
                pass  # Keep original color if conversion fails
        
        return {
            'color': color,
            'linestyle': self.line_styles.get(method, '-'),
            'linewidth': linewidth
        }
    
    def _apply_modern_axis_styling(self, axis: plt.Axes) -> None:
        """Apply consistent modern styling to an axis."""
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_color('#2E3440')
        axis.spines['bottom'].set_color('#2E3440')
        axis.grid(True, alpha=0.3, linewidth=0.5)
    
    def _create_inset_axes(self, axes_isi: List[plt.Axes]) -> List[plt.Axes]:
        """Create inset axes for ISI plots exactly like original."""
        axes_insets = []
        for type_id, _ in enumerate(self.data_types):
            width, height = 1.5*0.7, 2*0.3
            inset = inset_axes(axes_isi[type_id], width, height)
            inset.set_xlim(-0.5, 5.5)
            firing_rate = getattr(cf, 'FIRING_RATE', 50)
            inset.set_ylim(0.5 * firing_rate, 1.1 * firing_rate)
            inset.set_xticks([1, 3, 5])
            inset.tick_params(axis='both', pad=0.5, labelsize='x-small')
            # Apply modern styling to insets
            self._apply_modern_axis_styling(inset)
            axes_insets.append(inset)
        return axes_insets
    
    def _plot_isi_for_method(self, axes_isi: List[plt.Axes], axes_insets: List[plt.Axes],
                            data_type: str, type_id: int, method: str) -> bool:
        """Plot ISI distribution for a specific method."""
        filename = f'isi_{data_type}_{method}.npy'
        results = self._load_data_safely(filename)
        if results is None:
            return False
        
        bin_edges = results['bin_edges']
        hist = results['hist']
        bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        
        is_original = (method == 'original')
        plot_params = self._get_plot_params(method, is_original)
        
        # Only show label for first subplot to avoid duplicate legend entries
        if type_id == 0:
            plot_params['label'] = self.labels.get(method, method)
        
        # Plot on both main and inset axes
        axes_isi[type_id].plot(bin_centers, hist, **plot_params)
        axes_insets[type_id].plot(bin_centers, hist, **plot_params)
        
        return True
    
    def _plot_correlation_for_method(self, axes_corr: List[plt.Axes], type_id: int,
                                   data_type: str, method: str, corr_type: str) -> bool:
        """Plot correlation function for a specific method."""
        filename = f'{corr_type}_{data_type}_{method}.npy'
        results = self._load_data_safely(filename)
        if results is None:
            return False
        
        hist_times = results['hist_times']
        hist = results['hist']
        
        is_original = (method == 'original')
        plot_params = self._get_plot_params(method, is_original)
        
        axes_corr[type_id].plot(hist_times, hist, **plot_params)
        return True
    
    def _setup_correlation_axes_limits(self, axes_ac: List[plt.Axes], axes_cc: List[plt.Axes]) -> None:
        """Configure autocorrelation and cross-correlation axes with original limits."""
        firing_rate = getattr(cf, 'FIRING_RATE', 50)
        dither = getattr(cf, 'DITHER', 5)
        
        for axis_ac in axes_ac:
            axis_ac.set_ylim(bottom=getattr(cf, 'AC_BOTTOM', 0.7) * firing_rate,
                            top=getattr(cf, 'AC_TOP', 1.3) * firing_rate)
            axis_ac.set_xlim(left=-getattr(cf, 'AC_CC_XLIM', 10) * dither,
                            right=getattr(cf, 'AC_CC_XLIM', 10) * dither)
            axis_ac.set_xticks([-40, -20, 0, 20, 40])
        
        for axis_cc in axes_cc:
            axis_cc.set_ylim(bottom=getattr(cf, 'CC_BOTTOM', 0.7) * firing_rate,
                            top=getattr(cf, 'CC_TOP', 1.3) * firing_rate)
            axis_cc.set_xlim(left=-getattr(cf, 'AC_CC_XLIM', 10) * dither,
                            right=getattr(cf, 'AC_CC_XLIM', 10) * dither)
            axis_cc.set_xticks([-40, -20, 0, 20, 40])
    
    def _plot_firing_rate_change(self, axis: plt.Axes) -> bool:
        """Plot firing rate profile changes after applying surrogate methods."""
        results = self._load_data_safely('rate_step.npy')
        if results is None:
            return False
        
        # Plot original rate
        if 'original' in results:
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
                rate_surr = results[surr_method]
                surr_params = self._get_plot_params(surr_method)
                axis.plot(
                    rate_surr.times,
                    rate_surr.simplified.magnitude,
                    label=self.labels.get(surr_method, surr_method),
                    **surr_params
                )
        
        # Set labels and ticks
        axis.set_xlabel('t (ms)', labelpad=getattr(cf, 'XLABELPAD', 5))
        axis.set_ylabel('Firing rate (Hz)', labelpad=getattr(cf, 'YLABELPAD', 5))
        
        if hasattr(cf, 'DURATION_RATES_STEP'):
            t_stop = cf.DURATION_RATES_STEP.rescale(pq.ms).magnitude
            axis.set_xticks([0., t_stop/3, 2*t_stop/3, t_stop])
        
        return True
    
    def _plot_spike_movement_efficiency(self, axis: plt.Axes) -> bool:
        """Plot the ratio of moved spikes vs spike count as function of firing rate."""
        results = self._load_data_safely('clipped_rates.npy')
        if results is None:
            return False
        
        rates = results.get('rates', {})
        ratio_indep_moved = results.get('ratio_indep_moved', {})
        ratio_moved = results.get('ratio_moved', {})
        
        step_data_type = getattr(cf, 'STEP_DATA_TYPE', list(self.data_types)[0])
        
        # Plot independent shuffling baseline (original)
        if (step_data_type in rates and 
            step_data_type in ratio_indep_moved and 
            rates[step_data_type] is not None and 
            ratio_indep_moved[step_data_type] is not None):
            original_params = self._get_plot_params('original', is_original=True)
            axis.plot(
                rates[step_data_type],
                ratio_indep_moved[step_data_type],
                label='original',  # Changed from 'indep.' to 'original'
                **original_params
            )
        
        # Plot surrogate methods
        if step_data_type in ratio_moved:
            for surr_method in self.surr_methods:
                if surr_method in ratio_moved[step_data_type]:
                    surr_params = self._get_plot_params(surr_method)
                    axis.plot(
                        rates[step_data_type],
                        ratio_moved[step_data_type][surr_method],
                        label=self.labels.get(surr_method, surr_method),
                        **surr_params
                    )
        
        # Set labels and ticks
        axis.set_xlabel('λ (Hz)', labelpad=getattr(cf, 'XLABELPAD', 5))
        axis.set_ylabel('Moved Spikes Ratio', labelpad=getattr(cf, 'YLABELPAD2', 5))
        
        if hasattr(cf, 'RATES'):
            max_rate = cf.RATES[-1]
            axis.set_xticks([max_rate/4, max_rate/2, 3*max_rate/4, max_rate])
            axis.set_yticks([0.5, 0.7, 0.9])
        
        return True
    
    def _plot_cv_change(self, axis: plt.Axes) -> bool:
        """Plot the relationship between original and surrogate CV values."""
        results = self._load_data_safely('cv_change.npy')
        if results is None:
            return False
        
        if 'cvs_real' not in results:
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
                axis.plot(cvs_real, cvs_dithered, 
                         label=self.labels.get(surr_method, surr_method), 
                         **surr_params)
        
        # Set labels
        axis.set_xlabel('Original CV', labelpad=getattr(cf, 'XLABELPAD', 5))
        axis.set_ylabel('Surrogate CV', labelpad=getattr(cf, 'YLABELPAD', 5))
        
        return True
    
    def create_comprehensive_figure(self) -> None:
        """Create and save the comprehensive statistical overview figure."""
        # Create figure with DIN A4 width but original aspect ratio
        fig = plt.figure(figsize=(DIN_A4_WIDTH, DIN_A4_HEIGHT), dpi=300)
        fig.patch.set_facecolor('white')
        
        # Create main subplot grid (3x3 for ISI, AC, CC) using original layout
        axes = []
        for row in range(3):
            row_axes = []
            for col in range(3):
                left = cf.distance_left_border + cf.width_figure * col
                bottom = (cf.distance_bottom_border - 
                         (row - 2) * (cf.height_figure + cf.distance_vertical_panels))
                
                ax = fig.add_axes([left, bottom, cf.width_figure, cf.height_figure])
                self._apply_modern_axis_styling(ax)
                row_axes.append(ax)
            axes.append(row_axes)
        
        # Create right-side axes (like original)
        right_axes = []
        for row in range(3):
            left = (cf.distance_left_border + cf.width_figure * 3 + 
                   cf.distance_horizontal_panels)
            bottom = (cf.distance_bottom_border - 
                     (row - 2) * (cf.height_figure + cf.distance_vertical_panels))
            
            ax = fig.add_axes([left, bottom, cf.width_figure, cf.height_figure])
            self._apply_modern_axis_styling(ax)
            right_axes.append(ax)
        
        axes_isi, axes_cc, axes_ac = axes
        axis_cv, axis_moved, axis_step = right_axes
        
        # Set y-axis labels
        axes_isi[0].set_ylabel('ISI-distribution', labelpad=getattr(cf, 'YLABELPAD', 5))
        axes_ac[0].set_ylabel('Autocorrelation', labelpad=getattr(cf, 'YLABELPAD', 5))
        axes_cc[0].set_ylabel('Cross-correlation', labelpad=getattr(cf, 'YLABELPAD', 5))
        
        # Create inset axes for ISI plots
        axes_insets = self._create_inset_axes(axes_isi)
        
        # Plot main statistical analysis
        success_count = 0
        for type_id, data_type in enumerate(self.data_types):
            # Plot original data first
            methods_to_plot = ['original'] + self.surr_methods
            
            for method in methods_to_plot:
                if self._plot_isi_for_method(axes_isi, axes_insets, data_type, type_id, method):
                    success_count += 1
                if self._plot_correlation_for_method(axes_cc, type_id, data_type, method, 'cc'):
                    success_count += 1
                if self._plot_correlation_for_method(axes_ac, type_id, data_type, method, 'ac'):
                    success_count += 1
        
        # Configure correlation axes limits
        self._setup_correlation_axes_limits(axes_ac, axes_cc)
        
        # Plot additional analyses
        self._plot_firing_rate_change(axis_step)
        self._plot_spike_movement_efficiency(axis_moved)
        self._plot_cv_change(axis_cv)
        
        # Set x-axis labels
        for axis_isi in axes_isi:
            axis_isi.set_xlabel('t (ms)', labelpad=getattr(cf, 'XLABELPAD', 5))
        for axis_ac in axes_ac:
            axis_ac.set_xlabel('τ (ms)', labelpad=getattr(cf, 'XLABELPAD', 5))
        for axis_cc in axes_cc:
            axis_cc.set_xlabel('τ (ms)', labelpad=getattr(cf, 'XLABELPAD', 5))
        
        # Add titles only for main plots (A, B, C - not D, E, F)
        for data_id, data_type in enumerate(self.data_types):
            axes[0][data_id].set_title(data_type, fontsize=10, fontweight='bold')
        
        # Add subplot labels (A, B, C, etc.) - make them bold
        letters = getattr(cf, 'LETTERS', ['A', 'B', 'C', 'D', 'E', 'F'])
        for axis_id, axis_row in enumerate(axes):
            if axis_id < len(letters):
                axis_row[0].text(-0.25, 1.05, letters[axis_id],
                               transform=axis_row[0].transAxes, fontsize=12, fontweight='bold')
        
        for axis_id, axis in enumerate(right_axes):
            letter_id = len(axes) + axis_id
            if letter_id < len(letters):
                axis.text(-0.15, 1.05, letters[letter_id],
                         transform=axis.transAxes, fontsize=12, fontweight='bold')
        
        # Hide redundant labels
        for axis_row in axes:
            for axis in axis_row[1:]:
                axis.set_yticklabels([])
                axis.tick_params(left=False)
        
        # Synchronize y-limits
        for axes_row in axes:
            ylims = [axis.get_ylim() for axis in axes_row]
            unified_ylim = (
                min(ylim[0] for ylim in ylims),
                max(ylim[1] for ylim in ylims)
            )
            for axis in axes_row:
                axis.set_ylim(unified_ylim)
        
        # Add legend with less space at bottom
        if success_count > 0:
            handles, labels = axes_isi[0].get_legend_handles_labels()
            fig.legend(handles, labels, fontsize='small', fancybox=True, shadow=True,
                      ncol=7, loc="lower left", mode="expand", borderaxespad=0.3)
        
        # Save figure
        output_path = self.plot_path / getattr(cf, 'FIG_NAME', 'modern_surrogate_statistics')
        output_path_no_fmt = output_path.with_suffix('')
        for fmt in ('pdf', 'eps', 'png'):
            fig.savefig(output_path_no_fmt.with_suffix(f'.{fmt}'), dpi=300)
        
        plt.show()
        
        if success_count == 0:
            warnings.warn("No plots were successfully created - check data files")


def main():
    """
    Main function to create modern surrogate statistics plots.
    """
    try:
        # Create modern plotter
        plotter = CleanModernSurrogateStatisticsPlotter()
        
        # Create comprehensive figure
        print("Creating modern comprehensive statistical overview...")
        plotter.create_comprehensive_figure()
        
        print("Modern plots created successfully!")
        
    except Exception as e:
        print(f"Error creating modern plots: {e}")
        raise


if __name__ == '__main__':
    main()