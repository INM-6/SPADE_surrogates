"""
Script to create modern, readable figures showing patterns in experimental data.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
import yaml
from yaml import Loader

from SPADE_surrogates.analyse_data_utils.filter_results import load_filtered_results


class ExperimentalDataPlotter:
    """Class to handle plotting of experimental data results."""
    
    def __init__(self, config_path: str = "./configfile.yaml"):
        """Initialize plotter with configuration."""
        self.config = self._load_config(config_path)
        self.setup_plotting_style()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as stream:
            return yaml.load(stream, Loader=Loader)
    
    def setup_plotting_style(self):
        """Set up modern matplotlib style with readable text sizes for 4x3 layout."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.facecolor': 'white',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'font.family': 'sans-serif',
            'pdf.fonttype': 42,  # Avoid Type 3 fonts
            'ps.fonttype': 42,   # Avoid Type 3 fonts
            'svg.fonttype': 'none'  # Avoid Type 3 fonts in SVG
        })
    
    def load_pattern_results(self, surrogate: str, session_name: str, 
                           epoch: str, trial_type: str) -> List[Dict]:
        """Load pattern results from file."""
        directory = Path(f'../results/experimental_data/{surrogate}/{session_name}/{epoch}_{trial_type}')
        filepath = directory / 'filtered_res.npy'
        
        try:
            patterns, _, _, _, _ = load_filtered_results(filepath)
            return patterns
        except FileNotFoundError:
            print(f"Warning: No results found for {surrogate}/{session_name}/{epoch}_{trial_type}")
            return []
    
    def calculate_statistics(self, surrogate: str, sessions: List[str], 
                           trialtypes: List[str], epoch_tags: List[str],
                           binsize: pq.Quantity, winlen: int) -> Tuple[Dict, int]:
        """Calculate pattern statistics across sessions, epochs, and trial types."""
        pattern_counts = {}
        max_patterns_per_epoch = 0
        
        for session_name in sessions:
            pattern_counts[session_name] = self._initialize_session_stats(
                epoch_tags, trialtypes
            )
            
            for epoch_id, epoch in enumerate(epoch_tags):
                epoch_total = 0
                
                for trial_type in trialtypes:
                    # Load patterns for this behavioral context
                    patterns = self.load_pattern_results(
                        surrogate, session_name, epoch, trial_type
                    )
                    
                    # Store pattern count
                    behavioral_context = f"{epoch}_{trial_type}"
                    pattern_counts[session_name][behavioral_context] = len(patterns)
                    epoch_total += len(patterns)
                
                # Track maximum for dynamic y-axis scaling
                max_patterns_per_epoch = max(max_patterns_per_epoch, epoch_total)
        
        return pattern_counts, max_patterns_per_epoch
    
    def _initialize_session_stats(self, epoch_tags: List[str], 
                                trialtypes: List[str]) -> Dict:
        """Initialize statistics dictionary for a session."""
        stats = {}
        
        # Initialize for each behavioral context (epoch_trialtype combination)
        for epoch in epoch_tags:
            for trial_type in trialtypes:
                behavioral_context = f"{epoch}_{trial_type}"
                stats[behavioral_context] = 0
        
        return stats
    
    def _calculate_dynamic_ylims(self, surrogates: List[str], sessions: List[str],
                               trialtypes: List[str], epoch_tags: List[str],
                               binsize: pq.Quantity, winlen: int) -> Dict[int, int]:
        """Calculate dynamic y-limits: unique for top plots, shared for others."""
        ylims = {}
        
        # Calculate ylim for top plots (first surrogate - index 0)
        top_surrogate = surrogates[0]
        max_count_top = 0
        
        for session_name in sessions:
            pattern_counts, _ = self.calculate_statistics(
                top_surrogate, [session_name], trialtypes, epoch_tags, binsize, winlen
            )
            
            # Find maximum total count per epoch for top surrogate
            for epoch in epoch_tags:
                epoch_total = sum(
                    pattern_counts[session_name][f"{epoch}_{tt}"] 
                    for tt in trialtypes
                )
                max_count_top = max(max_count_top, epoch_total)
        
        # Set ylim for top plots (surrogate index 0)
        ylims[0] = int(np.ceil(max_count_top * 1.1 / 5) * 5)
        
        # Calculate shared ylim for all other plots (surrogate indices 1+)
        if len(surrogates) > 1:
            max_count_others = 0
            
            for surrogate in surrogates[1:]:  # Skip first surrogate
                for session_name in sessions:
                    pattern_counts, _ = self.calculate_statistics(
                        surrogate, [session_name], trialtypes, epoch_tags, binsize, winlen
                    )
                    
                    # Find maximum total count per epoch
                    for epoch in epoch_tags:
                        epoch_total = sum(
                            pattern_counts[session_name][f"{epoch}_{tt}"] 
                            for tt in trialtypes
                        )
                        max_count_others = max(max_count_others, epoch_total)
            
            # Set shared ylim for all other surrogates (indices 1+)
            shared_ylim = int(np.ceil(max_count_others * 1.1 / 5) * 5)
            # x1.5 this shared ylim that the difference is clearer
            shared_ylim = int(1.5 * shared_ylim)
            for surrogate_idx in range(1, len(surrogates)):
                ylims[surrogate_idx] = shared_ylim
        
        return ylims
    
    def create_modern_figure(self, surrogates: List[str], tag_surrogates: List[str],
                           sessions: List[str], trialtypes: List[str],
                           epoch_tags: List[str], epoch_tags_short: List[str],
                           binsize: pq.Quantity, winlen: int):
        """Create a modern-looking figure with 4 rows x 3 columns for DIN A4 portrait."""
        
        # Modern color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # DIN A4 portrait: 210mm x 297mm (8.27" x 11.69")
        # Use most of the page: approximately 8.0" x 10.5"
        fig = plt.figure(figsize=(8.0, 10.5), dpi=300)
        
        # Create 4 rows x 3 columns grid
        n_rows = 4
        n_cols = 3
        
        # Adjust subplot spacing for portrait layout with custom spacing
        fig.subplots_adjust(
            left=0.15,      # Left margin for y-labels
            right=0.95,     # Right margin
            bottom=0.15,    # Space for x-labels and legend
            top=0.92,       # Space for titles
            wspace=0.25,    # Space between columns
            hspace=0.35     # Reduced base spacing between rows
        )
        
        # Manually adjust row spacing for better grouping
        # We'll adjust individual subplot positions after creation
        
        # Calculate dynamic y-limits
        ylims = self._calculate_dynamic_ylims(
            surrogates, sessions, trialtypes, epoch_tags, binsize, winlen
        )
        
        # Create subplots with custom positioning for better grouping
        axes = []
        
        # Define custom positions for better row grouping
        row_positions = [0.73, 0.58, 0.28, 0.13]  # Moved all plots slightly down
        col_positions = [0.15, 0.45, 0.75]  # x-positions for 3 columns
        subplot_width = 0.25
        subplot_height = 0.12
        
        # Initialize axes array: axes[row][col]
        for row in range(n_rows):
            axes_row = []
            for col in range(n_cols):
                # Calculate position for this subplot
                left = col_positions[col]
                bottom = row_positions[row]
                
                # Create subplot with custom position
                ax = fig.add_axes([left, bottom, subplot_width, subplot_height])
                axes_row.append(ax)
            axes.append(axes_row)
        
        # Plot data - all 6 surrogate methods
        print("Pattern counts for each combination:")
        print("=" * 80)
        
        for surrogate_idx, (surrogate, tag_surrogate) in enumerate(zip(surrogates, tag_surrogates)):
            col = surrogate_idx % 3  # Column position (0, 1, 2)
            row_offset = 0 if surrogate_idx < 3 else 2  # First 3 methods: rows 0-1, Last 3 methods: rows 2-3
            
            print(f"\n{tag_surrogate} ({surrogate}) - Column {col + 1}:")
            print("-" * 60)
            
            # Plot for both sessions (N and L)
            for session_idx, session_name in enumerate(sessions):
                row = row_offset + session_idx  # Row position
                
                monkey_label = "Monkey N" if session_name == 'i140703s001' else "Monkey L"
                print(f"\n  {monkey_label} ({session_name}) - Row {row + 1}:")
                
                ax = axes[row][col]
                
                # Calculate statistics for this surrogate
                pattern_counts, _ = self.calculate_statistics(
                    surrogate, [session_name], trialtypes, epoch_tags, binsize, winlen
                )
                
                # Print pattern counts for each epoch and trial type
                for epoch_idx, epoch in enumerate(epoch_tags):
                    epoch_total = 0
                    epoch_details = []
                    
                    for trial_type in trialtypes:
                        behavioral_context = f"{epoch}_{trial_type}"
                        count = pattern_counts[session_name][behavioral_context]
                        epoch_total += count
                        epoch_details.append(f"{trial_type}: {count}")
                    
                    print(f"    {epoch_tags_short[epoch_idx]}: {epoch_total} total ({', '.join(epoch_details)})")
                
                # Plot stacked bars for each epoch
                self._plot_stacked_bars(
                    ax, pattern_counts[session_name], epoch_tags, 
                    trialtypes, colors, is_first_column=(surrogate_idx == 0)
                )
                
                # Customize axis
                self._customize_axis_4x3(
                    ax, session_name, tag_surrogate, epoch_tags_short,
                    ylims[surrogate_idx], row, col, surrogate_idx,
                    n_rows, n_cols
                )
        
        print("\n" + "=" * 80)
        
        # Add figure-level legend
        self._add_figure_legend(fig, trialtypes, colors)
        
        # Save figures
        self._save_figures(fig)
        
        return fig
    
    def _plot_stacked_bars(self, ax, pattern_counts: Dict, epoch_tags: List[str],
                          trialtypes: List[str], colors: List[str], 
                          is_first_column: bool = False):
        """Plot stacked bars for pattern counts."""
        bar_width = 0.7  # Increased for better visibility
        
        # Use smaller edge line width everywhere for nicer appearance
        edge_linewidth = 0.3
        
        for epoch_idx, epoch in enumerate(epoch_tags):
            bottom = 0
            
            for tt_idx, trial_type in enumerate(trialtypes):
                behavioral_context = f"{epoch}_{trial_type}"
                count = pattern_counts[behavioral_context]
                
                ax.bar(
                    epoch_idx + 1, count, bar_width,
                    bottom=bottom, color=colors[tt_idx],
                    alpha=0.8,
                    edgecolor='white', linewidth=edge_linewidth
                )
                
                bottom += count
    
    def _customize_axis_4x3(self, ax, session_name: str, tag_surrogate: str,
                           epoch_tags_short: List[str], ylim: int,
                           row: int, col: int, surrogate_idx: int,
                           n_rows: int, n_cols: int):
        """Customize individual axis appearance for 4x3 layout showing all 6 surrogate methods."""
        
        # Panel labels (A, B, C, D, E, F) for top rows of each section
        panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        if row == 0 or row == 2:  # Top row of each section
            panel_letter = panel_labels[surrogate_idx]
            ax.text(-0.08, 1.08, panel_letter, transform=ax.transAxes,
                   fontsize=16, fontweight='bold', 
                   verticalalignment='bottom', horizontalalignment='center')
        
        # Set column titles (surrogate methods) for rows 0 and 2 (top of each section)
        if row == 0 or row == 2:  # Top row of each section
            if surrogate_idx == 0:  # First column has different scale
                ax.set_title(f'{tag_surrogate}*', fontweight='bold', pad=25, fontsize=14)
                # Add note about different scale below UD title
                if row == 0:  # Only for the first UD title
                    ax.text(0.5, 0.88, '* Different y-axis scale', transform=ax.transAxes,
                           fontsize=9, style='italic', color='gray',
                           horizontalalignment='center', verticalalignment='top')
            else:
                ax.set_title(tag_surrogate, fontweight='bold', pad=25, fontsize=14)
        
        # Set y-axis
        ax.set_ylim(0, ylim)
        ax.set_xlim(0.5, len(epoch_tags_short) + 0.5)
        
        # Subtle visual marker for first column (different scale)
        if surrogate_idx == 0:
            # Add subtle colored background to indicate different scale
            ax.set_facecolor('#fff5f5')  # Very light red background
        
        # Get monkey information
        if session_name == 'l101210s001':
            monkey_name = 'Monkey L'
            session_id = 'l101210-001'
        elif session_name == 'i140703s001':
            monkey_name = 'Monkey N'
            session_id = 'i140703-001'
        else:
            monkey_name = 'Session'
            session_id = session_name
        
        # Y-label only for leftmost column (all rows)
        if col == 0:  # First column only
            ax.set_ylabel(f'{monkey_name}\n{session_id}\nPattern Count', 
                         fontweight='bold', fontsize=11)
        # Keep y-tick labels for all plots to show pattern counts
        # (removed the else clause that was hiding y-tick labels)
        
        # X-axis labels for bottom rows of each section (rows 1 and 3)
        if row == 1 or row == 3:  # Bottom row of each section
            ax.set_xticks(range(1, len(epoch_tags_short) + 1))
            ax.set_xticklabels(epoch_tags_short, rotation=45, ha='right', fontsize=11)
        else:
            ax.set_xticks(range(1, len(epoch_tags_short) + 1))
            ax.set_xticklabels([])
        
        # Improve tick parameters
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='x', which='major', pad=6)
        ax.tick_params(axis='y', which='major', pad=4)
    
    def _add_figure_legend(self, fig, trialtypes: List[str], colors: List[str]):
        """Add a figure-level legend at the bottom for 4x3 layout."""
        # Create legend handles
        legend_handles = []
        for tt_idx, trial_type in enumerate(trialtypes):
            handle = plt.Rectangle((0, 0), 1, 1, facecolor=colors[tt_idx], 
                                 alpha=0.8, edgecolor='white', linewidth=1.0)
            legend_handles.append(handle)
        
        # Add legend at bottom center of figure
        fig.legend(legend_handles, trialtypes, 
                  loc='lower center', 
                  bbox_to_anchor=(0.5, 0.02),
                  ncol=len(trialtypes), 
                  fontsize=12,
                  frameon=True, 
                  fancybox=True, 
                  shadow=True,
                  columnspacing=2.0,
                  handlelength=1.5,
                  handletextpad=0.8)
        
        # Remove the separate note since it's now integrated into the title
    
    def _save_figures(self, fig):
        """Save figures in multiple formats."""
        figures_dir = Path('../figures/')
        figures_dir.mkdir(exist_ok=True)
        
        # Save in multiple formats
        for fmt in ['png', 'pdf', 'svg']:
            fig.savefig(
                figures_dir / f'fig_experimental_data.{fmt}',
                dpi=300, bbox_inches='tight', facecolor='white'
            )
        
        print(f"Figures saved to {figures_dir}")


def main():
    """
    Main function to create the experimental data figure.
    
    Figure Caption:
    **Pattern detection results from experimental data analysis.**
    SPADE analysis results for two experimental sessions: monkey N (session i140703-001) 
    and monkey L (session l101210-001). Bar plots show the number of statistically 
    significant patterns detected in each behavioral epoch (start, cue, early-delay, 
    late-delay, movement, and hold) across different experimental conditions. Colors 
    indicate grip type and force combinations: precision grip with low force (PGLF), 
    precision grip with high force (PGHF), side grip with low force (SGLF), and side 
    grip with high force (SGHF). The figure uses a 4×3 layout where each column 
    represents a different surrogate method: (A) uniform dithering (UD), (B) uniform 
    dithering with dead-time (UDD), (C) joint ISI dithering (JISI-D), (D) 
    ISI dithering (ISI-D), (E) trial shifting (TR-SHIFT), and (F) window shuffling 
    (WIN-SHUFF). Within each surrogate method, the top row shows Monkey N results and 
    the bottom row shows Monkey L results. Note that uniform dithering (UD, panel A) 
    uses a different y-axis scale due to substantially higher pattern counts compared 
    to other methods.
    """
    # Initialize plotter
    plotter = ExperimentalDataPlotter()
    
    # Extract parameters from config
    config = plotter.config
    binsize = (config['binsize'] * pq.s).rescale(pq.ms)
    winlen = config['winlen']
    epoch_tags = config['epochs']
    epoch_tags_short = ['Start', 'Cue', 'Early-D', 'Late-D', 'Movement', 'Hold']
    trialtypes = config['trialtypes']
    sessions = config['reduced_sessions']
    
    # Surrogate methods
    surrogates = [
        'dither_spikes',
        'dither_spikes_with_refractory_period',
        'joint_isi_dithering',
        'isi_dithering',
        'trial_shifting',
        'bin_shuffling'
    ]
    
    tag_surrogates = ['UD', 'UDD', 'JISI-D', 'ISI-D', 'TR-SHIFT', 'WIN-SHUFF']
    
    # Create the figure
    fig = plotter.create_modern_figure(
        surrogates=surrogates,
        tag_surrogates=tag_surrogates,
        sessions=sessions,
        trialtypes=trialtypes,
        epoch_tags=epoch_tags,
        epoch_tags_short=epoch_tags_short,
        binsize=binsize,
        winlen=winlen
    )
    
    plt.show()


if __name__ == "__main__":
    main()