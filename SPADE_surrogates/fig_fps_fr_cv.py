"""
Firing Rate Distribution Plotter
==============================

A publication-ready plotting script for visualizing firing rate and CV2 distributions
from neurophysiological data with pattern detection results.

Features:
- Seaborn whitegrid styling for professional appearance
- Half A4 page sizing (8.27" x 5.83") for publications
- Guaranteed Type 3 font avoidance for journal compliance
- No LaTeX dependencies - uses Unicode symbols only
- Modern scatter plot visualization with categorical highlighting

Requirements:
- matplotlib >= 3.0
- seaborn >= 0.11
- numpy
- quantities
- yaml

Author: [Your Name]
Date: [Current Date]
"""

import itertools
import os
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import quantities as pq
import yaml

from generate_artificial_data import get_cv2
from rgutils import load_processed_spike_trains
from SPADE_surrogates.analyse_data_utils.filter_results import load_filtered_results


class FiringRateDistributionPlotter:
    """
    Creates publication-ready plots of firing rate and CV2 distributions.
    
    This class handles the visualization of neuronal firing characteristics
    with overlay of pattern detection results from multiple surrogate methods.
    """
    
    def __init__(self, config, file_paths, plot_parameters):
        """
        Initialize the plotter with configuration and parameters.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary from YAML file containing:
            - epochs: list of epoch names
            - processes: list of process types
            - trialtypes: list of trial types
            - reduced_sessions: list of session identifiers
            - winlen, binsize: analysis parameters
            
        file_paths : dict
            Dictionary containing file paths:
            - spiketrain: path to spike train data
            - results: path to pattern detection results
            - plot: path for output figures
            
        plot_parameters : dict
            Dictionary containing plotting parameters:
            - figsize, fontsize_*, markersize, alpha, etc.
        """
        self.config = config
        self.paths = file_paths
        self.params = plot_parameters
        
        # Analysis parameters
        self.separation_time = 2.0 * config['winlen'] * config['binsize']  # seconds
        self.epoch_duration = 0.5  # seconds
        
        # Color scheme (seaborn-compatible)
        self.colors = {
            'all': '#1f77b4',        # Seaborn blue
            'only UD': '#ff7f0e',    # Seaborn orange  
            'UD & UDD': '#2ca02c',   # Seaborn green
            'other': '#d62728'       # Seaborn red
        }
        
        # Load excluded neurons data
        self.excluded_neurons = np.load(
            'analysis_artificial_data/excluded_neurons.npy',
            allow_pickle=True
        ).item()
        
        # Create epoch labels for display
        self.epoch_labels = self._create_epoch_labels()
        
        # Session and process name mappings
        self.session_names = {'i140703s001': 'Monkey N', 'l101210s001': 'Monkey L'}
        self.process_names = {'ppd': 'PPD', 'gamma': 'Gamma'}
        
        # Surrogate method configurations
        self.surrogate_methods = ['ud', 'udrp', 'jisi', 'isi', 'bin_shuffling', 'tr_shift']
        self.surrogate_method_mapping = {
            'ud': 'dither_spikes',
            'udrp': 'dither_spikes_with_refractory_period', 
            'jisi': 'joint_isi_dithering',
            'isi': 'isi_dithering',
            'tr_shift': 'trial_shifting',
            'bin_shuffling': 'bin_shuffling'
        }
    
    def _create_epoch_labels(self):
        """
        Create abbreviated epoch labels for y-axis display.
        
        Returns
        -------
        list
            List of abbreviated epoch names
        """
        abbreviations = {
            'movement': 'mov',
            'latedelay': 'late', 
            'earlydelay': 'early',
            'cue1': 'cue'
        }
        return [abbreviations.get(epoch, epoch) for epoch in self.config['epochs']]
    
    def _load_spike_data(self, session, process, epoch, trialtype):
        """
        Load spike train data for specified experimental condition.
        
        Parameters
        ----------
        session : str
            Session identifier
        process : str
            Process type (e.g., 'ppd', 'gamma')
        epoch : str
            Epoch name
        trialtype : str
            Trial type identifier
            
        Returns
        -------
        list
            List of spike train objects with excluded neurons removed
        """
        filepath = f"{self.paths['spiketrain']}{process}/{session}/{process}_{epoch}_{trialtype}.npy"
        spike_trains = load_processed_spike_trains(filepath)
        
        # Remove excluded neurons
        for neuron_id in self.excluded_neurons[session]:
            spike_trains.pop(int(neuron_id))
            
        return spike_trains
    
    def _calculate_firing_rates(self, spike_trains):
        """
        Calculate firing rates from spike trains.
        
        Parameters
        ----------
        spike_trains : list
            List of spike train objects
            
        Returns
        -------
        numpy.ndarray
            Array of firing rates in Hz
        """
        first_train = spike_trains[0]
        total_duration = (first_train.t_stop.simplified.item() - 
                         first_train.t_start.simplified.item())
        
        n_trials = int(total_duration / (self.epoch_duration + self.separation_time))
        effective_duration = n_trials * self.epoch_duration
        
        return np.array([len(train) / effective_duration for train in spike_trains])
    
    def _calculate_cv2_values(self, spike_trains):
        """
        Calculate CV2 values from spike trains.
        
        Parameters
        ----------
        spike_trains : list
            List of spike train objects
            
        Returns
        -------
        numpy.ndarray
            Array of CV2 values
        """
        return np.array([
            get_cv2(train, sep=self.separation_time * pq.s) 
            for train in spike_trains
        ])
    
    def _load_pattern_detection_results(self, session, process, epoch, trialtype):
        """
        Load pattern detection results for all surrogate methods.
        
        Parameters
        ----------
        session, process, epoch, trialtype : str
            Experimental condition identifiers
            
        Returns
        -------
        tuple
            (neurons_per_method dict, all_neurons array)
        """
        neurons_per_method = {}
        all_detected_neurons = np.array([])
        
        for method in self.surrogate_methods:
            full_method_name = self.surrogate_method_mapping[method]
            result_file = (f"{self.paths['results']}{full_method_name}/{process}/"
                          f"{session}/{epoch}_{trialtype}/filtered_res.npy")
            
            patterns, _, _, _, _ = load_filtered_results(result_file)
            
            if len(patterns) == 0:
                neurons_per_method[method] = np.array([])
                continue
                
            method_neurons = np.unique(
                np.hstack([pattern['neurons'] for pattern in patterns])
            )
            neurons_per_method[method] = method_neurons
            all_detected_neurons = np.unique(
                np.hstack((all_detected_neurons, method_neurons))
            )
        
        return neurons_per_method, all_detected_neurons
    
    def _categorize_neurons_by_detection(self, neurons_per_method, all_neurons):
        """
        Categorize neurons based on which surrogate methods detected them.
        
        Parameters
        ----------
        neurons_per_method : dict
            Mapping of method names to detected neuron arrays
        all_neurons : numpy.ndarray
            Array of all detected neurons
            
        Returns
        -------
        dict
            Mapping of category names to neuron arrays
        """
        neuron_combinations = defaultdict(list)
        
        for neuron in all_neurons:
            detecting_methods = [
                method for method in self.surrogate_methods
                if neuron in neurons_per_method[method]
            ]
            neuron_combinations[tuple(detecting_methods)].append(neuron)
        
        # Convert to readable category names
        categorized = {}
        for method_tuple, neuron_list in neuron_combinations.items():
            neurons = np.array(neuron_list, dtype=int)
            category = self._get_category_name(method_tuple)
            categorized[category] = neurons
            
        return categorized
    
    def _get_category_name(self, method_combination):
        """
        Convert method combination to readable category name.
        
        Parameters
        ----------
        method_combination : tuple
            Tuple of detecting method names
            
        Returns
        -------
        str
            Human-readable category name
        """
        n_methods = len(method_combination)
        
        if n_methods == len(self.surrogate_methods):
            return 'all'
        elif n_methods == 1 and method_combination[0] == 'ud':
            return 'only UD'
        elif (n_methods == 2 and 
              'ud' in method_combination and 
              'udrp' in method_combination):
            return 'UD & UDD'
        else:
            return 'other'
    
    def _configure_axis(self, ax, plot_type, session, process, is_leftmost_column):
        """
        Configure individual axis appearance and labels.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis object to configure
        plot_type : str
            Either 'rate' or 'CV2'
        session : str
            Session identifier for row label
        process : str
            Process type for column
        is_leftmost_column : bool
            Whether this is the leftmost column (for y-label)
        """
        # Set axis labels (Unicode only - no LaTeX)
        if plot_type == 'rate':
            xlabel = 'λ (Hz)'    # Unicode lambda U+03BB
            ax.set_xlim(self.params['rate_limits'])
            ax.set_xticks(self.params['rate_ticks'])
        else:
            xlabel = 'CV2'       # Just CV2, not CV_2
            ax.set_xlim(0.4, 1.6)
            ax.set_xticks(np.arange(0.4, 1.7, 0.2))
        
        # Configure y-axis
        n_conditions = len(self.config['trialtypes']) + 1
        y_positions = n_conditions * np.arange(len(self.epoch_labels))
        ax.set_yticks(y_positions)
        
        # Only show y-labels on leftmost column
        if is_leftmost_column:
            ax.set_yticklabels(self.epoch_labels, fontsize=self.params['fontsize_ticks'])
        else:
            ax.set_yticklabels([])
        
        # INVERT Y-AXIS
        ax.invert_yaxis()
        
        # Set x-axis label for all subplots
        ax.set_xlabel(xlabel, fontsize=self.params['fontsize_axis'], fontweight='bold')
        
        # Configure tick appearance
        ax.tick_params(axis='both', which='major', labelsize=self.params['fontsize_ticks'])
        
        # Set column title (process + plot type) - use "Firing Rate" instead of "RATE"
        if session == self.config['reduced_sessions'][0]:  # First session only
            plot_type_display = "Firing Rate" if plot_type == 'rate' else plot_type.upper()
            title = f"{self.process_names[process]} - {plot_type_display}"
            ax.set_title(title, fontsize=self.params['fontsize_title'], 
                        fontweight='bold', y=self.params['yloc_title'], pad=15)
        
        # Enable grid (seaborn whitegrid style)
        ax.grid(True)
        ax.set_axisbelow(True)
    
    def create_distribution_plots(self, axes, plot_types):
        """
        Create the main distribution plots in the new 2x4 layout.
        
        Parameters
        ----------
        axes : list of lists
            2D list of matplotlib axes objects (2 rows x 4 columns)
        plot_types : list
            List of plot types for each column ['rate', 'CV2', 'rate', 'CV2']
            
        Returns
        -------
        dict
            Dictionary mapping category names to plot handles for legend
        """
        legend_handles = {category: None for category in self.colors.keys()}
        
        # Layout: rows = sessions, columns = [PPD rate, Gamma rate, PPD CV2, Gamma CV2]
        # Swapped columns 2 and 3 as requested
        processes = ['ppd', 'gamma', 'ppd', 'gamma']
        
        # Z-order mapping for proper layering (highest to lowest)
        zorder_map = {
            'all': 5,        # Blue dots - highest z-order
            'other': 4,      # Red dots - second highest
            'UD & UDD': 3,   # Green dots - third
            'only UD': 2     # Orange dots - lowest of highlighted
        }
        
        for row_idx, (axes_row, session) in enumerate(zip(axes, self.config['reduced_sessions'])):
            # Add row label (monkey name) with better positioning
            axes_row[0].text(-0.35, 0.5, self.session_names[session], 
                           transform=axes_row[0].transAxes, rotation=90, 
                           verticalalignment='center', fontsize=self.params['fontsize_title'],
                           fontweight='bold')
            
            for col_idx, (ax, process, plot_type) in enumerate(zip(axes_row, processes, plot_types)):
                is_leftmost_column = (col_idx == 0)
                self._configure_axis(ax, plot_type, session, process, is_leftmost_column)
                
                # Process each experimental condition
                for condition_idx, ((epoch_idx, epoch), trialtype) in enumerate(
                    itertools.product(enumerate(self.config['epochs']), self.config['trialtypes'])
                ):
                    y_position = condition_idx + epoch_idx
                    
                    # Load and process data
                    spike_trains = self._load_spike_data(session, process, epoch, trialtype)
                    
                    if plot_type == 'rate':
                        values = self._calculate_firing_rates(spike_trains)
                    else:
                        values = self._calculate_cv2_values(spike_trains)
                    
                    # Plot all neurons as background (less transparent, smaller)
                    ax.scatter(values, np.full(len(values), y_position),
                             c='#CCCCCC', s=self.params['markersize_background']**2, 
                             alpha=self.params['alpha_background'], 
                             linewidths=0, zorder=1)
                    
                    # Load pattern detection results
                    neurons_per_method, all_neurons = self._load_pattern_detection_results(
                        session, process, epoch, trialtype
                    )
                    categorized_neurons = self._categorize_neurons_by_detection(
                        neurons_per_method, all_neurons
                    )
                    
                    # Highlight categorized neurons with proper z-order
                    for category, neuron_indices in categorized_neurons.items():
                        if len(neuron_indices) > 0:
                            legend_handles[category] = ax.scatter(
                                values[neuron_indices], 
                                np.full(len(neuron_indices), y_position),
                                c=self.colors[category], 
                                s=self.params['markersize_highlight']**2,
                                alpha=self.params['alpha_highlight'], 
                                linewidths=0, zorder=zorder_map[category]
                            )
        
        return legend_handles


def setup_matplotlib_for_publication():
    """Configure matplotlib to avoid Type 3 fonts and LaTeX dependencies."""
    # CRITICAL: Set these before any plotting operations
    matplotlib.rcParams.update({
        # Type 3 font prevention (MANDATORY for publication)
        'pdf.fonttype': 42,          # Embed TrueType fonts in PDF
        'ps.fonttype': 42,           # Embed TrueType fonts in PostScript
        'svg.fonttype': 'none',      # Don't convert text to paths in SVG
        
        # LaTeX prevention (MANDATORY for compatibility)
        'text.usetex': False,        # Disable LaTeX completely
        'mathtext.default': 'regular', # No math italics
        'mathtext.fontset': 'dejavusans', # System fonts for math
        'text.latex.preamble': '',   # Clear LaTeX preamble
        'pgf.rcfonts': False,        # Don't use PGF fonts
        
        # Safe font configuration
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica']
    })


def get_plot_parameters():
    """
    Define plotting parameters optimized for publication.
    
    Returns
    -------
    dict
        Dictionary of plotting parameters
    """
    return {
        # Figure dimensions (top half of A4 page at 300 DPI with margins)
        'figsize': (11.0, 5.5),    # A4 width with margins, half height
        
        # Typography (sized for readability)
        'fontsize_title': 12.0,
        'fontsize_axis': 10.0,
        'fontsize_ticks': 9.0,
        'fontsize_legend': 10.0,
        
        # Visual elements (smaller dots, adjusted transparency)
        'markersize_background': 2.5,    # Smaller background dots
        'markersize_highlight': 3.0,     # Smaller highlighted dots
        'alpha_background': 0.4,         # Less transparent background
        'alpha_highlight': 0.6,          # More transparent highlights
        'yloc_title': 1.05,              # Title vertical position
        
        # Axis configuration
        'rate_limits': (-2.0, 69.0),
        'rate_ticks': np.arange(0, 61, 10),
        'spine_width': 0.8
    }


def get_file_paths():
    """
    Define file paths for data and output.
    
    Returns
    -------
    dict
        Dictionary of file paths
    """
    return {
        'spiketrain': '../data/artificial_data/',
        'results': '../results/artificial_data/',
        'plot': '../figures/fp_firing_distribution/',
        'config': 'configfile.yaml'
    }


def create_figure_layout(plot_params):
    """
    Create figure with 2x4 layout for top half of A4 page.
    
    Parameters
    ----------
    plot_params : dict
        Plotting parameters dictionary
        
    Returns
    -------
    tuple
        (figure, axes_2x4)
    """
    # Apply seaborn style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Override with publication-safe settings
    plt.rcParams.update({
        'axes.labelsize': plot_params['fontsize_axis'],
        'axes.titlesize': plot_params['fontsize_title'],
        'xtick.labelsize': plot_params['fontsize_ticks'],
        'ytick.labelsize': plot_params['fontsize_ticks'],
        'legend.fontsize': plot_params['fontsize_legend'],
        'axes.linewidth': plot_params['spine_width'],
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7
    })
    
    # Create figure
    fig = plt.figure(figsize=plot_params['figsize'], facecolor='white')
    
    # Adjust layout - move plots to right, reduce spacing between plots
    plt.subplots_adjust(bottom=0.08, top=0.85, left=0.15, right=0.98, 
                       hspace=0.35, wspace=0.3)
    
    # Create 2x4 subplot grid
    axes = []
    for row in range(2):
        axes_row = []
        for col in range(4):
            ax = fig.add_subplot(2, 4, row * 4 + col + 1)
            axes_row.append(ax)
        axes.append(axes_row)
    
    return fig, axes


def add_panel_labels(axes, plot_params):
    """
    Add panel labels (A, B, C, D) to the figure.
    
    Parameters
    ----------
    axes : list of lists
        2x4 axes array
    plot_params : dict
        Plotting parameters
    """
    label_style = {
        'fontsize': plot_params['fontsize_title'] + 2,
        'fontweight': 'bold',
        'color': '#333333'
    }
    
    # Add labels only to top row
    labels = ['A', 'B', 'C', 'D']
    for i, label in enumerate(labels):
        plt.text(-0.15, 1.15, label, transform=axes[0][i].transAxes, **label_style)


def create_legend(fig, color_mapping, plot_params):
    """
    Create publication-quality legend.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object
    color_mapping : dict
        Category to color mapping
    plot_params : dict
        Plotting parameters
    """
    legend_elements = []
    legend_labels = []
    
    for category, color in color_mapping.items():
        element = plt.scatter([], [], c=color, s=40, alpha=0.8, linewidths=0)
        legend_elements.append(element)
        legend_labels.append(category)
    
    fig.legend(legend_elements, legend_labels,
              fontsize=plot_params['fontsize_legend'],
              loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05),
              frameon=True, fancybox=False, shadow=False, framealpha=0.95,
              edgecolor='#DDDDDD', facecolor='white')


def save_publication_figure(file_paths, dpi=300):
    """
    Save figure with guaranteed Type 3 font avoidance.
    
    Parameters
    ----------
    file_paths : dict
        File path dictionary
    dpi : int, optional
        Resolution in dots per inch (default: 300)
        
    Returns
    -------
    list
        List of saved file paths
    """
    # Final safety check for Type 3 fonts
    with plt.rc_context({
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'text.usetex': False, 'font.family': 'sans-serif'
    }):
        output_files = [
            f"{file_paths['plot']}firing_rate_distribution.png",
            f"{file_paths['plot']}firing_rate_distribution.pdf", 
            f"{file_paths['plot']}firing_rate_distribution.svg"
        ]
        
        for filepath in output_files:
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
    
    return output_files


def main():
    """
    Main function to generate publication-ready firing rate distribution plots.
    
    This function creates a 2x4 layout figure:
    - Rows: Monkey N (top), Monkey L (bottom)  
    - Columns: PPD Firing Rate, Gamma Firing Rate, PPD CV2, Gamma CV2
    """
    # Essential: Configure matplotlib before any plotting
    setup_matplotlib_for_publication()
    
    # Setup configuration
    file_paths = get_file_paths()
    plot_params = get_plot_parameters()
    
    # Ensure output directory exists
    os.makedirs(file_paths['plot'], exist_ok=True)
    
    # Load experimental configuration
    with open(file_paths['config'], 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # Initialize plotter
    plotter = FiringRateDistributionPlotter(config, file_paths, plot_params)
    
    # Create figure layout (2 rows x 4 columns)
    fig, axes = create_figure_layout(plot_params)
    
    # Define plot types for columns: [PPD rate, Gamma rate, PPD CV2, Gamma CV2]
    # Swapped columns 2 and 3 as requested
    plot_types = ['rate', 'rate', 'CV2', 'CV2']
    
    # Generate plots
    legend_handles = plotter.create_distribution_plots(axes, plot_types)
    
    # Add labels and legend
    add_panel_labels(axes, plot_params)
    create_legend(fig, plotter.colors, plot_params)
    
    # Save publication-ready files
    saved_files = save_publication_figure(file_paths)
    
    # Success message
    print("✅ Publication-ready figure generated successfully!")
    print(f"📁 Files saved: {', '.join(os.path.basename(f) for f in saved_files)}")
    print(f"📏 Dimensions: {plot_params['figsize'][0]:.1f}\" × {plot_params['figsize'][1]:.1f}\" (top half A4)")
    print("🎨 Layout: 2 rows (monkeys) × 4 columns (PPD rate, Gamma rate, PPD CV2, Gamma CV2)")
    print("🔵 Z-order: Blue (all) → Red (other) → Green (UD & UDD) → Orange (only UD)")
    print("🔍 Smaller dots with adjusted transparency")
    print("✅ Type 3 fonts avoided - journal compliant")
    print("✅ No LaTeX dependencies - cross-platform compatible")
    print("\n📋 Verification command:")
    print("   pdffonts firing_rate_distribution.pdf")
    print("   (Should show only 'TrueType' fonts)")


if __name__ == '__main__':
    main()