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

# ANU thesis textwidth dimensions (4 rows x 3 columns)
ANU_TEXTWIDTH_MM = 150   # mm (exact from anuthesis.sty)
ANU_TEXTWIDTH = ANU_TEXTWIDTH_MM / 25.4  # inches (150mm = 5.906 inches)
FIGURE_HEIGHT = ANU_TEXTWIDTH * 4/3      # inches (4:3 ratio = 7.874 inches)


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
            width, height = 0.7, 0.7
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
        axis.set_ylabel('λ(t) (Hz)', labelpad=getattr(cf, 'YLABELPAD', 5))
        
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
        axis.set_xlabel('CV - original', labelpad=getattr(cf, 'XLABELPAD', 5))
        axis.set_ylabel('CV - surrogate', labelpad=getattr(cf, 'YLABELPAD', 5))
        
        # Set specific ticks for CV plot
        axis.set_xticks([0.6, 0.8, 1.0])
        axis.set_yticks([0.6, 0.8, 1.0])
        axis.set_xlim(0.55, 1.05)
        axis.set_ylim(0.55, 1.05)
        
        return True
    
    def create_comprehensive_figure(self) -> None:
        """Create and save the comprehensive statistical overview figure."""
        # Create figure with exact 150mm width and 4:3 aspect ratio
        fig = plt.figure(figsize=(ANU_TEXTWIDTH, FIGURE_HEIGHT), dpi=300)
        fig.patch.set_facecolor('white')
        
        # Calculate subplot dimensions with extra spacing before last row
        margin_left = 0.08
        margin_right = 0.02
        margin_top = 0.05    # Reduced top margin to ensure upper row is visible
        margin_bottom = 0.08  # Updated margin
        
        spacing_horizontal = 0.08  # Increased for better y-tick visibility
        spacing_vertical = 0.03
        spacing_before_last_row = 0.12  # Updated spacing
        
        # Calculate available space for 3 main rows + 1 additional row
        available_height = 1 - margin_top - margin_bottom - 2 * spacing_vertical - spacing_before_last_row
        main_panel_height = available_height * 0.75 / 3  # 75% for main 3 rows
        additional_panel_height = available_height * 0.25  # 25% for additional row
        
        panel_width = (1 - margin_left - margin_right - 2 * spacing_horizontal) / 3
        
        # Create main subplot grid (4 rows x 3 columns) with better spacing
        axes = []
        for row in range(4):  # 4 rows: Data types 1-3, Additional
            row_axes = []
            for col in range(3):  # 3 columns: 3 analysis types
                left = margin_left + col * (panel_width + spacing_horizontal)
                
                if row < 3:  # First 3 rows (main data)
                    panel_height = main_panel_height
                    bottom = (margin_bottom + additional_panel_height + spacing_before_last_row + 
                             (2 - row) * (main_panel_height + spacing_vertical))
                else:  # Last row (additional analyses)
                    panel_height = additional_panel_height
                    bottom = margin_bottom
                
                ax = fig.add_axes([left, bottom, panel_width, panel_height])
                self._apply_modern_axis_styling(ax)
                row_axes.append(ax)
            axes.append(row_axes)
        
        # Add a subtle separating line before the last row
        line_y = margin_bottom + additional_panel_height + spacing_before_last_row / 2
        fig.add_artist(plt.Line2D([margin_left, 1 - margin_right], [line_y, line_y], 
                                 color='gray', linewidth=0.8, alpha=0.5))
        
        # Assign axes to meaningful names (TRANSPOSED)
        # Now each data type gets a row, and each analysis type gets a column
        axes_data_type_1 = axes[0]  # Row 1: Data type 1 (ISI, AC, CC)
        axes_data_type_2 = axes[1]  # Row 2: Data type 2 (ISI, AC, CC)  
        axes_data_type_3 = axes[2]  # Row 3: Data type 3 (ISI, AC, CC)
        axes_additional = axes[3]   # Row 4: Additional analyses
        
        # For compatibility with existing plotting code, create column-wise access
        axes_isi = [axes_data_type_1[0], axes_data_type_2[0], axes_data_type_3[0]]  # Col 1: ISI for all data types
        axes_ac = [axes_data_type_1[1], axes_data_type_2[1], axes_data_type_3[1]]   # Col 2: AC for all data types
        axes_cc = [axes_data_type_1[2], axes_data_type_2[2], axes_data_type_3[2]]   # Col 3: CC for all data types
        
        # Individual additional analysis axes
        axis_cv = axes_additional[0]      # Panel J: CV analysis
        axis_moved = axes_additional[1]   # Panel K: Spike movement  
        axis_step = axes_additional[2]    # Panel L: Firing rate
        
        # Set y-axis labels with simplified data type names
        data_type_names = ['Poisson', 'PPD', 'Gamma']  # Simplified names
        axes_data_type_1[0].set_ylabel(f'{data_type_names[0]}', labelpad=getattr(cf, 'YLABELPAD', 5))
        axes_data_type_2[0].set_ylabel(f'{data_type_names[1]}', labelpad=getattr(cf, 'YLABELPAD', 5))
        axes_data_type_3[0].set_ylabel(f'{data_type_names[2]}', labelpad=getattr(cf, 'YLABELPAD', 5))
        
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
        
        # Set x-axis labels (now different due to transpose)
        # Bottom row of main 3x3 grid gets x-labels
        axes_data_type_3[0].set_xlabel('ISI: t (ms)', labelpad=getattr(cf, 'XLABELPAD', 5))
        axes_data_type_3[1].set_xlabel('time lag: τ (ms)', labelpad=getattr(cf, 'XLABELPAD', 5))
        axes_data_type_3[2].set_xlabel('time lag: τ (ms)', labelpad=getattr(cf, 'XLABELPAD', 5))
        
        # Additional analyses also get x-labels and y-labels
        for axis in axes_additional:
            if axis == axis_cv:
                axis.set_xlabel('Original CV', labelpad=getattr(cf, 'XLABELPAD', 5))
                axis.set_ylabel('Surrogate CV', labelpad=getattr(cf, 'YLABELPAD', 5))
            elif axis == axis_moved:
                axis.set_xlabel('λ (Hz)', labelpad=getattr(cf, 'XLABELPAD', 5))
                axis.set_ylabel('Moved Spikes Ratio', labelpad=getattr(cf, 'YLABELPAD', 5))
            elif axis == axis_step:
                axis.set_xlabel('t (ms)', labelpad=getattr(cf, 'XLABELPAD', 5))
                axis.set_ylabel('λ(t) (Hz)', labelpad=getattr(cf, 'YLABELPAD', 5))
        
        # Add titles only for top row (now analysis types instead of data types)
        axes_data_type_1[0].set_title('ISI-distribution', fontsize=10)
        axes_data_type_1[1].set_title('Autocorrelation', fontsize=10)
        axes_data_type_1[2].set_title('Cross-correlation', fontsize=10)
        
        # Add subplot labels (A through L) for 4x3 grid - ensure all 12 are positioned
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        
        # Label each position in the 4x3 grid (row by row)
        label_index = 0
        for row in range(4):
            for col in range(3):
                if label_index < len(letters):
                    # Position labels more carefully to ensure they're visible
                    axes[row][col].text(-0.14, 1.08, letters[label_index],
                                       transform=axes[row][col].transAxes, 
                                       fontsize=12, fontweight='bold',
                                       ha='center', va='bottom')
                    label_index += 1
        
        # Hide redundant labels for cleaner appearance (adapted for transpose)
        # Hide y-labels for columns 2 and 3 in first 3 rows (keep only leftmost column)
        for row_id in range(3):  # First 3 rows
            for col_id in range(1, 3):  # Columns 2 and 3
                axes[row_id][col_id].set_yticklabels([])
                axes[row_id][col_id].tick_params(left=False)
        
        # For additional analyses row: keep all y-ticks visible (don't hide any)
        # This ensures all three bottom plots have y-ticks
        
        # Synchronize y-limits within each row (each data type)
        for row_id in range(3):  # First 3 rows (data types)
            ylims = [axes[row_id][col].get_ylim() for col in range(3)]
            unified_ylim = (
                min(ylim[0] for ylim in ylims),
                max(ylim[1] for ylim in ylims)
            )
            for col in range(3):
                axes[row_id][col].set_ylim(unified_ylim)
        
        # Don't synchronize the additional analyses row - they have different scales
        
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