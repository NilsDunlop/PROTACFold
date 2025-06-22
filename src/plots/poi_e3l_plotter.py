import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from base_plotter import BasePlotter
from config import PlotConfig
from utils import save_figure

class POI_E3LPlotter(BasePlotter):
    """
    Class for plotting POI and E3 ligase metrics in grid layouts.
    
    This class provides methods for creating grid-style plots showing both 
    Protein of Interest (POI) and E3 Ligase metrics in organized layouts:
    - Single model grids (POI and E3L for one model)
    - Combined grids (POI and E3L for multiple models)
    
    The original separate plotting methods are deprecated in favor of 
    cleaner grid layouts.
    """
    
    # --- Protein Group and Color Map Constants ---
    PROTAC_POI_GROUPS = {
        "Kinases": ["BTK", "PTK2", "WEE1"],
        "Nuclear_Regulators": ["SMARCA2", "SMARCA4", "STAT5A", "STAT6", "BRD2", "BRD4", "BRD9", "WDR5"],
        "Signaling_Modulators": ["FKBP1A", "FKBP5", "KRAS", "PTPN2"],
        "Apoptosis_Regulators": ["BCL2", "BCL2L1"],
        "Diverse_Enzymes": ["CA2", "EPHX2", "R1AB"]
    }
    PROTAC_E3_GROUPS = {
        "CRBN": ["CRBN"], "VHL": ["VHL"], "BIRC2": ["BIRC2"], "DCAF1": ["DCAF1"]
    }
    MG_POI_GROUPS = {
        "Kinases": ["CDK12", "CSNK1A1", "PAK6"],
        "Nuclear_Regulators": ["BRD4", "HDAC1", "WIZ"],
        "Transcription_Factors": ["IKZF1", "IKZF2", "SALL4", "ZNFN1A2", "ZNF692"],
        "RNA_Translation_Regulators": ["RBM39", "GSPT1"],
        "Signaling_Metabolism": ["CTNNB1", "CDO1"]
    }
    MG_E3_GROUPS = {
        "CRBN": ["CRBN", "MGR_0879"], "VHL": ["VHL"], "TRIM_Ligase": ["TRIM21"],
        "DCAF_Receptors": ["DCAF15", "DCAF16"], "Others": ["BTRC", "DDB1", "KBTBD4"],
    }
    PROTAC_POI_COLOR_MAP = {
        "Kinases": PlotConfig.SMILES_PRIMARY, "Nuclear_Regulators": PlotConfig.SMILES_TERTIARY,
        "Signaling_Modulators": PlotConfig.SMILES_SECONDARY, "Apoptosis_Regulators": PlotConfig.CCD_PRIMARY,
        "Diverse_Enzymes": PlotConfig.CCD_SECONDARY,
    }
    PROTAC_E3_COLOR_MAP = {
        "CRBN": PlotConfig.CCD_PRIMARY, "VHL": PlotConfig.SMILES_TERTIARY,
        "BIRC2": PlotConfig.SMILES_PRIMARY, "DCAF1": PlotConfig.CCD_SECONDARY,
    }
    MG_POI_COLOR_MAP = {
        "Kinases": PlotConfig.SMILES_PRIMARY, "Nuclear_Regulators": PlotConfig.SMILES_TERTIARY,
        "Transcription_Factors": PlotConfig.SMILES_SECONDARY, "RNA_Translation_Regulators": PlotConfig.CCD_PRIMARY,
        "Signaling_Metabolism": PlotConfig.CCD_SECONDARY,
    }
    MG_E3_COLOR_MAP = {
        "CRBN": PlotConfig.CCD_PRIMARY, "VHL": PlotConfig.SMILES_TERTIARY,
        "TRIM_Ligase": PlotConfig.SMILES_PRIMARY, "DCAF_Receptors": PlotConfig.CCD_SECONDARY,
        "Others": PlotConfig.GRAY,
    }

    # --- Default Styling and Layout Constants ---
    # General Plotting Style - REFINED for cleaner look
    CLS_DEFAULT_BAR_WIDTH = 0.7
    CLS_BAR_ALPHA = 1
    CLS_BAR_EDGE_COLOR = 'black'
    CLS_BAR_LINEWIDTH = 0.5
    CLS_ERROR_BAR_CAPSIZE = 3              # Reduced from 5 to 3
    CLS_ERROR_BAR_THICKNESS = 0.8          # Added for consistency
    CLS_ERROR_BAR_ALPHA = 0.7              # Added for consistency
    CLS_THRESHOLD_LINE_ALPHA = 1
    CLS_THRESHOLD_LINE_WIDTH = 1.0         # Added for consistency

    # Font Sizes - INCREASED for better readability in smaller plots
    CLS_PLOT_TITLE_FONTSIZE = 15      # Increased from 14 to 15
    CLS_AXIS_LABEL_FONTSIZE_INCREMENT_GRID = 2  # Increment for PlotConfig.AXIS_LABEL_SIZE in grid plots
    CLS_YTICK_FONTSIZE = 12
    CLS_TICK_FONTSIZE_GRID = 13  # For combined grid plots
    CLS_LEGEND_FONTSIZE_RMSD = 11      # Increased from 9.5 to 11
    CLS_LEGEND_FONTSIZE_GRID = 13      # For combined grid plots
    CLS_LEGEND_FONTSIZE_LABEL = 15     # Increased from 14 to 15 for legend labels

    # Legend Styling
    CLS_LEGEND_FRAME_ON = False
    CLS_LEGEND_FRAME_ALPHA = 0.7
    CLS_LEGEND_COLUMN_SPACING = 1.0
    CLS_LEGEND_HANDLE_TEXT_PAD = 0.5
    CLS_LEGEND_NCOL_E3L_MAX = 2
    CLS_LEGEND_NCOL_POI_MAX = 1
    CLS_LEGEND_NCOL_GRID_MAX = 1

    # Default Sizing for plot_poi_e3l_rmsd and _create_rmsd_plot - REDUCED for compact plots
    CLS_PLOT_DEFAULT_WIDTH_RMSD = 4.0  # Reduced from 6.0 to 4.0
    CLS_HEIGHT_CALC_MIN_RMSD = 8.0
    CLS_HEIGHT_CALC_FACTOR_RMSD = 0.35
    CLS_HEIGHT_CALC_PADDING_RMSD = 2.0
    CLS_E3L_HEIGHT_MIN_RMSD = 4.0

    # Default Sizing for plot_combined_grid - REDUCED for compact plots
    CLS_GRID_DEFAULT_WIDTH = 8.0          # Reduced from 12.0 to 8.0
    CLS_GRID_HEIGHT_CALC_FACTOR = 0.4
    CLS_GRID_HEIGHT_CALC_PADDING = 3.0
    CLS_GRID_DEFAULT_OVERALL_HEIGHT = 12.0
    CLS_GRID_X_AXIS_PADDING_FACTOR = 0.05
    CLS_GRID_HSPACE = 0.05
    CLS_GRID_YLABEL_PAD = 20
    CLS_GRID_YLABEL_X_COORD = -0.45
    CLS_GRID_YLABEL_Y_COORD = 0.5
    CLS_GRID_ALPHA = 0.2
    
    # --- New constant for single model grid plots ---
    CLS_SINGLE_MODEL_GRID_WIDTH = CLS_GRID_DEFAULT_WIDTH / 2  # Half of the combined grid width
    
    def __init__(self, debug=False):
        """Initialize the plotter."""
        super().__init__()
        self.debug = debug

    def _debug_print(self, message):
        """Print debug information if debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] POI_E3LPlotter: {message}")
    
    def plot_poi_e3l_rmsd(
        self,
        df,
        model_type='AlphaFold3',
        metric_type='RMSD',
        add_threshold=True,
        threshold_value=4.0,
        width=CLS_PLOT_DEFAULT_WIDTH_RMSD,
        height=None,
        bar_width=CLS_DEFAULT_BAR_WIDTH,
        save=True,
        legend_position=None,
        molecule_type="PROTAC",
        debug=False
    ):
        """
        DEPRECATED: This method is deprecated. Use plot_single_model_grid() instead for a cleaner grid layout.
        
        This method now redirects to plot_single_model_grid() to maintain backwards compatibility.
        """
        print("Warning: plot_poi_e3l_rmsd() is deprecated. Using plot_single_model_grid() instead.")
        
        # Redirect to the grid method
        fig = self.plot_single_model_grid(
            df=df,
            model_type=model_type,
            metric_type=metric_type,
            add_threshold=add_threshold,
            threshold_value=threshold_value,
            width=width / 2,  # Adjust width since grid is different layout
            height=height,
            bar_width=bar_width,
            save=save,
            legend_position=legend_position,
            molecule_type=molecule_type,
            debug=debug
        )
        
        # Return in the old format (list of POI figs, E3L fig) for compatibility
        # Since grid returns a single figure, we return it as the E3L figure
        return [], fig
    
    def _get_sorted_e3l_results(self, molecule_type, e3l_results, metric_type=None):
        """
        Sort E3L results based on metric value with highest values at the top.
        
        Args:
            molecule_type: Type of molecule ("PROTAC" or "MOLECULAR GLUE")
            e3l_results: List of processed E3L data
            metric_type: Type of metric ('RMSD' or 'DockQ') for determining sort direction
            
        Returns:
            List of sorted E3L results with highest metric values at the top
        """
        # Sort proteins by metric value (highest values at top)
        sorted_e3l_results = sorted(e3l_results, key=lambda x: x['mean_metric'], reverse=False)
        return sorted_e3l_results
    
    def _sort_results_by_group_order(self, results, group_order, metric_type=None):
        """
        Sort results by metric value with highest values at the top.
        
        Args:
            results: List of protein result dictionaries
            group_order: List of groups (used only for color mapping, not for ordering)
            metric_type: Type of metric ('RMSD' or 'DockQ')
            
        Returns:
            List of sorted results with highest metric values at the top
        """
        # Sort proteins by metric value (highest values at top)
        sorted_results = sorted(results, key=lambda x: x['mean_metric'], reverse=False)
        return sorted_results
        
    def _process_protein_data(self, df, name_column, groups, metric_type='RMSD'):
        """
        Process protein data to calculate mean metric values.
        
        Args:
            df: DataFrame with data
            name_column: Column with protein names ('SIMPLE_POI_NAME' or 'SIMPLE_E3_NAME')
            groups: Dictionary with protein groups
            metric_type: 'RMSD' or 'DockQ'
            
        Returns:
            list: List of dictionaries with processed data
        """
        results = []
        
        # Determine column names based on metric type
        if metric_type == 'RMSD':
            smiles_col = 'SMILES_RMSD'
            ccd_col = 'CCD_RMSD'
        else:  # DockQ
            smiles_col = 'SMILES_DOCKQ_SCORE'
            ccd_col = 'CCD_DOCKQ_SCORE'
        
        # Create a flat mapping of protein names to their groups
        protein_to_group = {}
        for group, proteins in groups.items():
            for protein in proteins:
                protein_to_group[protein] = group
        
        # Get unique protein names
        unique_proteins = df[name_column].unique()
        
        for protein in unique_proteins:
            # Skip empty or NaN values
            if pd.isna(protein) or protein == '':
                continue
                
            protein_df = df[df[name_column] == protein]
            
            # Calculate mean metric for each seed (combining SMILES and CCD)
            seed_mean_metric = []
            seed_values = protein_df['SEED'].unique()
            
            for seed in seed_values:
                seed_df = protein_df[protein_df['SEED'] == seed]
                
                # Calculate mean of SMILES and CCD metrics for this seed
                smiles_metric = seed_df[smiles_col].mean()
                ccd_metric = seed_df[ccd_col].mean()
                
                # Skip if both values are NaN
                if pd.isna(smiles_metric) and pd.isna(ccd_metric):
                    continue
                
                # Calculate combined mean
                metric_values = [val for val in [smiles_metric, ccd_metric] if not pd.isna(val)]
                if metric_values:
                    combined_mean = np.nanmean(metric_values)
                    seed_mean_metric.append(combined_mean)
            
            # Skip if no valid metric values
            if not seed_mean_metric:
                continue
                
            # Get overall mean and std across seeds
            overall_mean = np.nanmean(seed_mean_metric)
            overall_std = np.nanstd(seed_mean_metric) if len(seed_mean_metric) > 1 else 0
            
            # Determine the group this protein belongs to
            group = protein_to_group.get(protein, 'Others')
            
            results.append({
                'name': protein,
                'group': group,
                'mean_metric': overall_mean,
                'std_metric': overall_std,
                'count': len(protein_df)
            })
        
        return results
        
    def _create_split_poi_plots(
        self, 
        data, 
        color_map,
        model_type, 
        metric_type,
        add_threshold, 
        threshold_value, 
        width, 
        height, 
        bar_width, 
        save,
        legend_position=None,
        molecule_type="PROTAC"
    ):
        """
        DEPRECATED: This method is deprecated and only kept for backwards compatibility.
        Use plot_single_model_grid() or plot_combined_grid() instead.
        
        Returns:
            list: Empty list to maintain compatibility
        """
        print("Warning: _create_split_poi_plots() is deprecated. Use grid plotting methods instead.")
        return []
        
    def _create_rmsd_plot(
        self, 
        data, 
        protein_type, 
        color_map,
        model_type,
        metric_type,
        add_threshold, 
        threshold_value, 
        width, 
        height, 
        bar_width, 
        save,
        suffix="",
        custom_legend_order=None,
        legend_position=None,
        molecule_type="PROTAC"
    ):
        """
        Create plot for either POI or E3L with RMSD or DockQ metrics.
        
        Args:
            data: Processed data from _process_protein_data
            protein_type: 'POI' or 'E3L'
            color_map: Dictionary mapping groups to colors
            model_type: 'AlphaFold3' or 'Boltz-1'
            metric_type: 'RMSD' or 'DockQ'
            add_threshold: Whether to add a threshold line
            threshold_value: Value for the threshold line
            width: Figure width
            height: Figure height
            bar_width: Width of the bars. Now passed as an argument.
            save: Whether to save the figure
            suffix: Optional suffix to add to the filename
            custom_legend_order: Optional custom order for legend groups
            legend_position: Fixed position for the legend ('upper center', 'upper right', etc.)
                            If None, position will be determined automatically
            molecule_type: Type of molecule being plotted
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if not data:
            return None
            
        # Calculate appropriate height if not provided
        if height is None:
            height = max(self.CLS_HEIGHT_CALC_MIN_RMSD, len(data) * self.CLS_HEIGHT_CALC_FACTOR_RMSD + self.CLS_HEIGHT_CALC_PADDING_RMSD)
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Set up positions for bars
        positions = np.arange(len(data))
        
        # Prepare data for plotting
        names = [item['name'] for item in data]
        means = [item['mean_metric'] for item in data]
        stds = [item['std_metric'] for item in data]
        groups = [item['group'] for item in data]
        
        # Fallback color for any group not in the color map
        default_color = PlotConfig.GRAY
        
        # Assign colors with fallback to default for any missing group
        colors = []
        for group in groups:
            if group in color_map:
                colors.append(color_map[group])
            else:
                self._debug_print(f"Warning: Group '{group}' not found in color map. Using default color.")
                colors.append(default_color)
        
        # Plot bars
        bars = ax.barh(positions, means, bar_width, xerr=stds, 
                      color=colors, alpha=self.CLS_BAR_ALPHA, 
                      edgecolor=self.CLS_BAR_EDGE_COLOR, linewidth=self.CLS_BAR_LINEWIDTH,
                      error_kw={'ecolor': 'black', 'capsize': self.CLS_ERROR_BAR_CAPSIZE, 
                                'capthick': self.CLS_ERROR_BAR_THICKNESS, 'alpha': self.CLS_ERROR_BAR_ALPHA})
        
        # Add threshold line if requested
        if add_threshold and threshold_value is not None:
            ax.axvline(x=threshold_value, color='gray', linestyle='--', 
                      alpha=self.CLS_THRESHOLD_LINE_ALPHA, linewidth=self.CLS_THRESHOLD_LINE_WIDTH, label='Threshold')
            
        # Customize the plot
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=self.CLS_TICK_FONTSIZE_GRID)
        
        # Set appropriate x-axis label based on metric type
        if metric_type == 'RMSD':
            ax.set_xlabel('RMSD (Å)', fontsize=self.CLS_LEGEND_FONTSIZE_LABEL)
        else:  # DockQ
            ax.set_xlabel('DockQ Score', fontsize=self.CLS_LEGEND_FONTSIZE_LABEL)
        
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=self.CLS_GRID_ALPHA)
        
        # Set y-axis limits to reduce empty space for POI plots
        if protein_type == 'POI':
            y_margin = 0.6  # Small margin around the bars for POI
            ax.set_ylim(positions[0] - y_margin, positions[-1] + y_margin)
        
        # Create legend elements
        legend_elements = self._create_legend_elements(groups, color_map, custom_legend_order, add_threshold)
        
        # Determine legend position
        legend_loc = self._determine_legend_position(
            legend_position, protein_type, metric_type, means
        )
        
        # Create the legend - COMMENTED OUT FOR NOW
        # legend = ax.legend(
        #     handles=legend_elements,
        #     loc=legend_loc,
        #     ncol=1 if protein_type == 'POI' else min(self.CLS_LEGEND_NCOL_E3L_MAX, len(legend_elements)),
        #     fontsize=self.CLS_LEGEND_FONTSIZE_RMSD,
        #     framealpha=self.CLS_LEGEND_FRAME_ALPHA,
        #     frameon=self.CLS_LEGEND_FRAME_ON,
        #     facecolor='white',
        #     columnspacing=self.CLS_LEGEND_COLUMN_SPACING,
        #     handletextpad=self.CLS_LEGEND_HANDLE_TEXT_PAD
        # )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            metric_abbr = metric_type.lower()
            filename = f"{protein_type.lower().split()[0]}_{metric_abbr}_{model_type.lower().replace('-', '_')}{suffix}"
            self.save_plot(fig, filename)
            
        return fig

    def _create_legend_elements(self, groups, color_map, custom_legend_order=None, add_threshold=False):
        """
        Create legend elements based on groups and colors.
        
        Args:
            groups: List of group names for each bar
            color_map: Dictionary mapping groups to colors
            custom_legend_order: Optional custom order for legend groups
            add_threshold: Whether to add threshold line to legend
            
        Returns:
            list: Legend elements
        """
        legend_elements = []
        
        # Get the list of groups to include in the legend
        if custom_legend_order:
            # Use custom order for legend groups
            legend_groups = custom_legend_order
        else:
            # Use unique groups from data in reverse order to match visual appearance
            legend_groups = []
            for group in groups:
                if group not in legend_groups:
                    legend_groups.append(group)
            
            # Reverse the order to match visual appearance in plot (top to bottom)
            legend_groups.reverse()
        
        # Create a legend element for each group
        for group in legend_groups:
            # Only add groups that exist in the data
            if group in groups:
                display_name = group.replace('_', ' ')
                # Use the color from the color map, or gray if not found
                color = color_map.get(group, PlotConfig.GRAY)
                legend_elements.append(Patch(facecolor=color, label=display_name,
                                             edgecolor=self.CLS_BAR_EDGE_COLOR, 
                                             linewidth=self.CLS_BAR_LINEWIDTH))
        
        # Add threshold to legend if it exists
        if add_threshold:
            from matplotlib.lines import Line2D
            threshold_line = Line2D([0], [0], color='gray', linestyle='--',
                                   label='Threshold')
            legend_elements.append(threshold_line)
        
        return legend_elements
    
    def _determine_legend_position(self, legend_position, protein_type, metric_type, means):
        """
        Determine legend position for the plot.
        
        Args:
            legend_position: User-specified position (if any)
            protein_type: 'POI' or 'E3L' (not used in current implementation)
            metric_type: 'RMSD' or 'DockQ' (not used in current implementation)
            means: List of mean values (not used in current implementation)
            
        Returns:
            str: Legend position
        """
        # If legend position is specified, use that
        if legend_position is not None:
            return legend_position
        
        # Default legend position is bottom left
        return 'lower right'

    def plot_combined_grid(
        self,
        df,
        model_types=['AlphaFold3', 'Boltz-1'],
        metric_type='RMSD',
        add_threshold=True,
        threshold_value=4.0,
        width=CLS_GRID_DEFAULT_WIDTH,
        height=None,
        bar_width=CLS_DEFAULT_BAR_WIDTH,
        save=False,
        legend_position='lower right',
        molecule_type="PROTAC",
        debug=False
    ):
        """
        Plot RMSD or DockQ for POIs and E3Ls in a 2x2 grid with POI plots on top and E3L plots at the bottom.
        This maintains the same plotting logic as plot_combined_models but arranges the plots in a grid.
        
        Args:
            df: DataFrame with RMSD data
            model_types: List of model types to include, e.g. ['AlphaFold3', 'Boltz-1']
            metric_type: 'RMSD' or 'DockQ'
            add_threshold: Whether to add a threshold line
            threshold_value: Value for the threshold line
            width: Figure width
            height: Figure height (calculated automatically if None)
            bar_width: Width of the bars
            save: Whether to save the figure
            legend_position: Position for the legend (defaults to lower right)
            molecule_type: Type of molecule to filter by ('PROTAC' or 'MOLECULAR GLUE')
            debug: Enable debugging output
            
        Returns:
            matplotlib.figure.Figure: The created combined figure
        """
        # Enable debugging for this call if requested
        self.debug = debug or self.debug
        
        # First, generate the POI and E3L data using the same logic as plot_combined_models
        poi_data, e3l_data = self._prepare_combined_plot_data(
            df=df,
            model_types=model_types,
            metric_type=metric_type,
            molecule_type=molecule_type
        )
        
        if not poi_data or not e3l_data:
            print(f"Error: No valid POI or E3L data found after filtering")
            return None
        
        # Find max number of proteins across all models
        max_poi_count = max([len(poi_data[model_type]) for model_type in model_types])
        max_e3l_count = max([len(e3l_data[model_type]) for model_type in model_types])
        
        # Calculate proportional heights to maintain same bar thickness between POI and E3L plots
        if max_poi_count > 0 and max_e3l_count > 0:
            # Calculate the height ratio based strictly on the data point counts
            height_ratio = [max_poi_count, max_e3l_count]
            
            # Calculate base figure height with tighter margins
            total_data_points = max_poi_count + max_e3l_count
            base_height = total_data_points * self.CLS_GRID_HEIGHT_CALC_FACTOR + self.CLS_GRID_HEIGHT_CALC_PADDING
            
            if height is None:
                height = base_height
        else:
            height_ratio = [2, 1]  # Default ratio if we don't have data
            if height is None:
                height = self.CLS_GRID_DEFAULT_OVERALL_HEIGHT
        
        # Find the global min and max values across all datasets for consistent x-axis
        global_min = float('inf')
        global_max = float('-inf')
        
        # Get min/max from POI data
        for model_type in model_types:
            for item in poi_data[model_type]:
                metric_value = item['mean_metric']
                error_value = item['std_metric']
                global_min = min(global_min, metric_value - error_value if error_value else metric_value)
                global_max = max(global_max, metric_value + error_value if error_value else metric_value)
                
        # Get min/max from E3L data
        for model_type in model_types:
            for item in e3l_data[model_type]:
                metric_value = item['mean_metric']
                error_value = item['std_metric']
                global_min = min(global_min, metric_value - error_value if error_value else metric_value)
                global_max = max(global_max, metric_value + error_value if error_value else metric_value)
        
        # Add a small padding to the global range
        range_padding = (global_max - global_min) * self.CLS_GRID_X_AXIS_PADDING_FACTOR
        global_min = max(0, global_min - range_padding)
        global_max = global_max + range_padding
        
        # Create figure with 2x2 grid
        fig = plt.figure(figsize=(width, height), constrained_layout=True)
        
        # Create GridSpec with proper height ratios and minimal spacing
        gs = fig.add_gridspec(2, 2, height_ratios=height_ratio, hspace=self.CLS_GRID_HSPACE)
        
        # Create a list to store all axes for later adjustments
        all_axes = []
        left_axes = []  # Store left column axes for label alignment
        
        # Process each model in the grid
        for i, model_type in enumerate(model_types):
            # --- TOP ROW: POI PLOTS ---
            ax_poi = fig.add_subplot(gs[0, i])
            all_axes.append(ax_poi)
            if i == 0:
                left_axes.append(ax_poi)
            
            # Get POI data for this model
            poi_results = poi_data[model_type]
            sorted_poi_results = sorted(poi_results, key=lambda x: x['mean_metric'], reverse=False)
            
            # Plot POI data
            self._plot_grid_data(
                ax=ax_poi,
                data=sorted_poi_results,
                color_map=self._get_color_map('POI', molecule_type),
                add_threshold=add_threshold,
                threshold_value=threshold_value,
                bar_width=bar_width,
                title=None,
                legend_position=legend_position,
                show_legend=True,
                custom_legend_order=self._get_legend_order('POI', molecule_type)
            )
            
            # Set y-label only for the first column
            if i == 0:
                # Don't set the label yet, we'll set it with consistent parameters later
                pass
            
            # No x-axis label for top row
            ax_poi.set_xlabel('')
            
            # --- BOTTOM ROW: E3L PLOTS ---
            ax_e3l = fig.add_subplot(gs[1, i])
            all_axes.append(ax_e3l)
            if i == 0:
                left_axes.append(ax_e3l)
            
            # Get E3L data for this model
            e3l_results = e3l_data[model_type]
            sorted_e3l_results = sorted(e3l_results, key=lambda x: x['mean_metric'], reverse=False)
            
            # Plot E3L data
            self._plot_grid_data(
                ax=ax_e3l,
                data=sorted_e3l_results,
                color_map=self._get_color_map('E3L', molecule_type),
                add_threshold=add_threshold,
                threshold_value=threshold_value,
                bar_width=bar_width,
                title=None,
                legend_position=legend_position,
                show_legend=True,
                custom_legend_order=self._get_legend_order('E3L', molecule_type)
            )
            
            # Set labels for bottom row
            if i == 0:
                # Don't set the label yet, we'll set it with consistent parameters later
                pass
                
            # Set x-axis label based on metric type
            if metric_type == 'RMSD':
                ax_e3l.set_xlabel('RMSD (Å)', fontsize=self.CLS_LEGEND_FONTSIZE_LABEL)
            else:  # DockQ
                ax_e3l.set_xlabel('DockQ Score', fontsize=self.CLS_LEGEND_FONTSIZE_LABEL)
        
        # Set consistent x-axis limits for all subplots
        for ax in all_axes:
            ax.set_xlim(global_min, global_max)
        
        # Get largest tick label width for both axes to ensure proper alignment
        fig.canvas.draw()  # Force draw to get text bounds
        max_tick_width = 0
        for ax in left_axes:
            tick_labels = ax.get_yticklabels()
            if tick_labels:
                for label in tick_labels:
                    bbox = label.get_window_extent()
                    width_inches = bbox.width / fig.dpi
                    max_tick_width = max(max_tick_width, width_inches)
        
        # Now set the y-axis labels with consistent positioning
        label_pad_val = 0.5 # Default padding if max_tick_width is not available or zero
        if max_tick_width > 0 : # Ensure max_tick_width is positive before adding
             label_pad_val = max_tick_width + 0.5 # This 0.5 could be a constant too, e.g. CLS_GRID_YLABEL_TICK_PADDING
        
        # Set the labels with the consistent pad and increased font size
        left_axes[0].set_ylabel('Protein of Interest', fontsize=self.CLS_LEGEND_FONTSIZE_LABEL, labelpad=self.CLS_GRID_YLABEL_PAD)
        left_axes[1].set_ylabel('E3 Ligase', fontsize=self.CLS_LEGEND_FONTSIZE_LABEL)
        
        # Further alignment adjustments - moved closer to the axis
        left_axes[0].yaxis.set_label_coords(self.CLS_GRID_YLABEL_X_COORD, self.CLS_GRID_YLABEL_Y_COORD)
        left_axes[1].yaxis.set_label_coords(self.CLS_GRID_YLABEL_X_COORD, self.CLS_GRID_YLABEL_Y_COORD)
        
        # Save figure if requested
        if save:
            metric_abbr = metric_type.lower()
            models_abbr = "_".join([m.lower().replace('-', '_') for m in model_types])
            filename = f"poi_e3l_grid_{metric_abbr}_{models_abbr}"
            self.save_plot(fig, filename)
            
        return fig
    
    def _prepare_combined_plot_data(self, df, model_types, metric_type, molecule_type):
        """Helper method to prepare data for combined plots (used by both combined methods)."""
        # Create two separate filtered dataframes for each model type
        filtered_dfs = {}
        for model_type in model_types:
            # Filter data based on model type - accept both 'Boltz1' and 'Boltz-1'
            if model_type == 'Boltz-1':
                model_variants = ['Boltz-1', 'Boltz1']
                model_df = df[df['MODEL_TYPE'].isin(model_variants)].copy()
            else:
                model_df = df[df['MODEL_TYPE'] == model_type].copy()
            
            if len(model_df) == 0:
                self._debug_print(f"Error: No data available for model type '{model_type}'")
                return None, None
            
            # Filter by molecule type
            # Check if we have MOLECULE_TYPE column (newer datasets) or TYPE column (older datasets)
            molecule_type_col = 'MOLECULE_TYPE' if 'MOLECULE_TYPE' in model_df.columns else 'TYPE'
            
            if molecule_type_col in model_df.columns:
                model_df = model_df[model_df[molecule_type_col] == molecule_type].copy()
                
                if model_df.empty:
                    self._debug_print(f"Error: No data available for molecule type '{molecule_type}' with model '{model_type}'")
                    return None, None
            else:
                self._debug_print(f"Warning: No '{molecule_type_col}' column found in data, skipping molecule type filtering")
            
            # Determine valid seeds based on model type
            if model_type == 'AlphaFold3':
                valid_seeds = [24, 37, 42]
            else:  # Boltz-1
                valid_seeds = [24, 37, 42]
            # Filter by seeds
            model_df = model_df[model_df['SEED'].isin(valid_seeds)]
            
            if len(model_df) == 0:
                self._debug_print(f"Error: No data available for selected seeds with model '{model_type}'")
                return None, None
                
            # Store the filtered dataframe
            filtered_dfs[model_type] = model_df
        
        # Select appropriate groups based on molecule type
        if molecule_type == "PROTAC":
            poi_groups = self.PROTAC_POI_GROUPS
            e3_groups = self.PROTAC_E3_GROUPS
        else:  # MOLECULAR GLUE
            poi_groups = self.MG_POI_GROUPS
            e3_groups = self.MG_E3_GROUPS
        
        # Process POI data for each model
        poi_results = {}
        for model_type, model_df in filtered_dfs.items():
            poi_results[model_type] = self._process_protein_data(model_df, 'SIMPLE_POI_NAME', poi_groups, metric_type)
        
        # Process E3L data for each model
        e3l_results = {}
        for model_type, model_df in filtered_dfs.items():
            e3l_results[model_type] = self._process_protein_data(model_df, 'SIMPLE_E3_NAME', e3_groups, metric_type)
        
        # Check if we have valid data
        for model_type in model_types:
            if not poi_results[model_type] or not e3l_results[model_type]:
                self._debug_print(f"Warning: Missing data for model '{model_type}'")
        
        return poi_results, e3l_results
    
    def _get_color_map(self, protein_type, molecule_type):
        """Helper method to get the appropriate color map."""
        if protein_type == 'POI':
            if molecule_type == "PROTAC":
                return self.PROTAC_POI_COLOR_MAP
            else:  # MOLECULAR GLUE
                return self.MG_POI_COLOR_MAP
        else:  # E3L
            if molecule_type == "PROTAC":
                return self.PROTAC_E3_COLOR_MAP
            else:  # MOLECULAR GLUE
                return self.MG_E3_COLOR_MAP
    
    def _get_legend_order(self, protein_type, molecule_type):
        """Helper method to get the appropriate legend order."""
        if protein_type == 'POI':
            if molecule_type == "PROTAC":
                return ["Kinases", "Nuclear_Regulators", "Signaling_Modulators", 
                        "Apoptosis_Regulators", "Diverse_Enzymes"]
            else:  # MOLECULAR GLUE
                return ["Kinases", "Nuclear_Regulators", "Transcription_Factors",
                        "RNA_Translation_Regulators", "Signaling_Metabolism"]
        else:  # E3L
            if molecule_type == "PROTAC":
                return ["CRBN", "VHL", "BIRC2", "DCAF1"]
            else:  # MOLECULAR GLUE
                return ["CRBN", "VHL", "TRIM_Ligase", "DCAF_Receptors", "Others"]
    
    def _plot_grid_data(self, ax, data, color_map, add_threshold, threshold_value, bar_width,
                        title=None, legend_position='lower right', show_legend=True, custom_legend_order=None):
        """Helper method to plot data on the given axis for the combined grid."""
        # Set up positions for bars
        positions = np.arange(len(data))
        
        # Prepare data for plotting
        names = [item['name'] for item in data]
        means = [item['mean_metric'] for item in data]
        stds = [item['std_metric'] for item in data]
        groups = [item['group'] for item in data]
        
        # Fallback color for any group not in the color map
        default_color = PlotConfig.GRAY
        
        # Assign colors with fallback to default for any missing group
        colors = []
        for group in groups:
            if group in color_map:
                colors.append(color_map[group])
            else:
                self._debug_print(f"Warning: Group '{group}' not found in color map. Using default color.")
                colors.append(default_color)
        
        # Plot bars
        bars = ax.barh(positions, means, bar_width, xerr=stds, 
                     color=colors, alpha=self.CLS_BAR_ALPHA, 
                     edgecolor=self.CLS_BAR_EDGE_COLOR, linewidth=self.CLS_BAR_LINEWIDTH,
                     error_kw={'ecolor': 'black', 'capsize': self.CLS_ERROR_BAR_CAPSIZE, 
                               'capthick': self.CLS_ERROR_BAR_THICKNESS, 'alpha': self.CLS_ERROR_BAR_ALPHA})
        
        # Add threshold line if requested
        if add_threshold and threshold_value is not None:
            ax.axvline(x=threshold_value, color='gray', linestyle='--', 
                      alpha=self.CLS_THRESHOLD_LINE_ALPHA, linewidth=self.CLS_THRESHOLD_LINE_WIDTH, label='Threshold')
            
        # Set title if provided
        if title:
            ax.set_title(title, fontsize=self.CLS_PLOT_TITLE_FONTSIZE)
            
        # Customize the plot
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=self.CLS_LEGEND_FONTSIZE_LABEL)
        
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=self.CLS_GRID_ALPHA)
        
        # Set y-axis limits to reduce empty space for POI plots
        if any(n in " ".join(names) for n in ["BTK", "SMARCA", "BRD4", "CDK12", "IKZF1"]):  # POI indicators
            y_margin = 0.6  # Small margin around the bars for POI
            ax.set_ylim(positions[0] - y_margin, positions[-1] + y_margin)
        
        # Create and show legend if requested
        if show_legend:
            legend_elements = self._create_legend_elements(groups, color_map, custom_legend_order, add_threshold)
            
            # Determine if this is a POI or E3L plot based on the titles or names
            is_poi_plot = any(n in " ".join(names) for n in ["BTK", "SMARCA", "BRD4"])
            
            ncol = 1 if is_poi_plot else min(self.CLS_LEGEND_NCOL_E3L_MAX, len(legend_elements))
            
            # ax.legend(
            #     handles=legend_elements,
            #     loc=legend_position,
            #     ncol=ncol,
            #     fontsize=self.CLS_LEGEND_FONTSIZE_GRID,
            #     framealpha=self.CLS_LEGEND_FRAME_ALPHA,
            #     frameon=self.CLS_LEGEND_FRAME_ON,
            #     facecolor='white',
            #     columnspacing=self.CLS_LEGEND_COLUMN_SPACING,
            #     handletextpad=self.CLS_LEGEND_HANDLE_TEXT_PAD
            # )

    def plot_single_model_grid(
        self,
        df,
        model_type,
        metric_type='RMSD',
        add_threshold=True,
        threshold_value=4.0,
        width=CLS_SINGLE_MODEL_GRID_WIDTH,
        height=None,
        bar_width=CLS_DEFAULT_BAR_WIDTH,
        save=False,
        legend_position='lower right',
        molecule_type="PROTAC",
        debug=False,
        x_lim=None
    ):
        """
        Plot RMSD or DockQ for a single model's POIs and E3Ls in a 1x2 grid.
        This creates a single model plot that is half the width of the combined plot.
        
        Args:
            df: DataFrame with RMSD data
            model_type: Model type to plot, e.g. 'AlphaFold3' or 'Boltz-1'
            metric_type: 'RMSD' or 'DockQ'
            add_threshold: Whether to add a threshold line
            threshold_value: Value for the threshold line
            width: Figure width (default is half of the combined grid width)
            height: Figure height (calculated automatically if None)
            bar_width: Width of the bars
            save: Whether to save the figure
            legend_position: Position for the legend
            molecule_type: Type of molecule to filter by ('PROTAC' or 'MOLECULAR GLUE')
            debug: Enable debugging output
            x_lim: Optional tuple (min, max) to set fixed x-axis limits
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Enable debugging for this call if requested
        self.debug = debug or self.debug
        
        # Prepare data for this model - using same helper as the combined grid
        poi_data, e3l_data = self._prepare_combined_plot_data(
            df=df,
            model_types=[model_type],
            metric_type=metric_type,
            molecule_type=molecule_type
        )
        
        if not poi_data or not e3l_data:
            print(f"Error: No valid POI or E3L data found for model {model_type} after filtering")
            return None
        
        # Find number of proteins for this model
        poi_count = len(poi_data[model_type])
        e3l_count = len(e3l_data[model_type])
        
        # Calculate proportional heights to maintain same bar thickness between POI and E3L plots
        if poi_count > 0 and e3l_count > 0:
            # Calculate the height ratio based strictly on the data point counts
            height_ratio = [poi_count, e3l_count]
            
            # Calculate base figure height with tighter margins
            total_data_points = poi_count + e3l_count
            base_height = total_data_points * self.CLS_GRID_HEIGHT_CALC_FACTOR + self.CLS_GRID_HEIGHT_CALC_PADDING
            
            if height is None:
                height = base_height
        else:
            height_ratio = [2, 1]  # Default ratio if we don't have data
            if height is None:
                height = self.CLS_GRID_DEFAULT_OVERALL_HEIGHT
        
        # If no x_lim provided, calculate from the data
        if x_lim is None:
            # Find data min and max
            global_min = float('inf')
            global_max = float('-inf')
            
            # Get min/max from POI data
            for item in poi_data[model_type]:
                metric_value = item['mean_metric']
                error_value = item['std_metric']
                global_min = min(global_min, metric_value - error_value if error_value else metric_value)
                global_max = max(global_max, metric_value + error_value if error_value else metric_value)
                
            # Get min/max from E3L data
            for item in e3l_data[model_type]:
                metric_value = item['mean_metric']
                error_value = item['std_metric']
                global_min = min(global_min, metric_value - error_value if error_value else metric_value)
                global_max = max(global_max, metric_value + error_value if error_value else metric_value)
            
            # Add a small padding to the global range
            range_padding = (global_max - global_min) * self.CLS_GRID_X_AXIS_PADDING_FACTOR
            global_min = max(0, global_min - range_padding)
            global_max = global_max + range_padding
            
            x_lim = (global_min, global_max)
        
        # Create figure with 1x2 grid (POI on top, E3L on bottom)
        fig = plt.figure(figsize=(width, height), constrained_layout=True)
        
        # Create GridSpec with proper height ratios and minimal spacing
        gs = fig.add_gridspec(2, 1, height_ratios=height_ratio, hspace=self.CLS_GRID_HSPACE)
        
        # --- TOP ROW: POI PLOT ---
        ax_poi = fig.add_subplot(gs[0, 0])
        
        # Get POI data for this model
        poi_results = poi_data[model_type]
        sorted_poi_results = sorted(poi_results, key=lambda x: x['mean_metric'], reverse=False)
        
        # Plot POI data
        self._plot_grid_data(
            ax=ax_poi,
            data=sorted_poi_results,
            color_map=self._get_color_map('POI', molecule_type),
            add_threshold=add_threshold,
            threshold_value=threshold_value,
            bar_width=bar_width,
            title=None,
            legend_position=legend_position,
            show_legend=True,
            custom_legend_order=self._get_legend_order('POI', molecule_type)
        )
        
        # Set x limits
        ax_poi.set_xlim(x_lim)
        
        # Set y-label
        ax_poi.set_ylabel('Protein of Interest', fontsize=self.CLS_LEGEND_FONTSIZE_LABEL, 
                          labelpad=self.CLS_GRID_YLABEL_PAD)
        ax_poi.yaxis.set_label_coords(self.CLS_GRID_YLABEL_X_COORD, self.CLS_GRID_YLABEL_Y_COORD)
        
        # No x-axis label for top row
        ax_poi.set_xlabel('')
        
        # --- BOTTOM ROW: E3L PLOT ---
        ax_e3l = fig.add_subplot(gs[1, 0])
        
        # Get E3L data for this model
        e3l_results = e3l_data[model_type]
        sorted_e3l_results = sorted(e3l_results, key=lambda x: x['mean_metric'], reverse=False)
        
        # Plot E3L data
        self._plot_grid_data(
            ax=ax_e3l,
            data=sorted_e3l_results,
            color_map=self._get_color_map('E3L', molecule_type),
            add_threshold=add_threshold,
            threshold_value=threshold_value,
            bar_width=bar_width,
            title=None,
            legend_position=legend_position,
            show_legend=True,
            custom_legend_order=self._get_legend_order('E3L', molecule_type)
        )
        
        # Set x limits
        ax_e3l.set_xlim(x_lim)
        
        # Set y-label
        ax_e3l.set_ylabel('E3 Ligase', fontsize=self.CLS_LEGEND_FONTSIZE_LABEL, 
                          labelpad=self.CLS_GRID_YLABEL_PAD)
        ax_e3l.yaxis.set_label_coords(self.CLS_GRID_YLABEL_X_COORD, self.CLS_GRID_YLABEL_Y_COORD)
        
        # Set x-axis label based on metric type
        if metric_type == 'RMSD':
            ax_e3l.set_xlabel('RMSD (Å)', fontsize=self.CLS_LEGEND_FONTSIZE_LABEL)
        else:  # DockQ
            ax_e3l.set_xlabel('DockQ Score', fontsize=self.CLS_LEGEND_FONTSIZE_LABEL)
        
        # Save figure if requested
        if save:
            metric_abbr = metric_type.lower()
            model_abbr = model_type.lower().replace('-', '_')
            filename = f"poi_e3l_single_{metric_abbr}_{model_abbr}"
            self.save_plot(fig, filename)
            
        return fig
        
    def plot_all_model_grids(
        self,
        df,
        model_types=['AlphaFold3', 'Boltz-1'],
        metric_type='RMSD',
        add_threshold=True,
        threshold_value=4.0,
        save=True,
        legend_position='lower right',
        molecule_type="PROTAC",
        debug=False
    ):
        """
        Create both a combined grid plot and individual plots for each model.
        This method calls both plot_combined_grid and plot_single_model_grid.
        
        Args:
            df: DataFrame with RMSD data
            model_types: List of model types to include
            metric_type: 'RMSD' or 'DockQ'
            add_threshold: Whether to add a threshold line
            threshold_value: Value for the threshold line
            save: Whether to save the figures
            legend_position: Position for the legend
            molecule_type: Type of molecule to filter by
            debug: Enable debugging output
            
        Returns:
            tuple: (combined_fig, single_figs) - The created figures
        """
        # Enable debugging for this call if requested
        self.debug = debug or self.debug
        
        # First create the combined plot to get the global x limits
        combined_fig = self.plot_combined_grid(
            df=df,
            model_types=model_types,
            metric_type=metric_type,
            add_threshold=add_threshold,
            threshold_value=threshold_value,
            save=save,
            legend_position=legend_position,
            molecule_type=molecule_type,
            debug=debug
        )
        
        if combined_fig is None:
            print("Error: Failed to create combined grid plot")
            return None, []
        
        # Get the global x limits from the first subplot of the combined figure
        axes = combined_fig.get_axes()
        if axes:
            global_xlim = axes[0].get_xlim()
        else:
            global_xlim = None
        
        # Create individual model plots using the same x limits
        single_figs = []
        for model_type in model_types:
            single_fig = self.plot_single_model_grid(
                df=df,
                model_type=model_type,
                metric_type=metric_type,
                add_threshold=add_threshold,
                threshold_value=threshold_value,
                save=save,
                legend_position=legend_position,
                molecule_type=molecule_type,
                debug=debug,
                x_lim=global_xlim
            )
            
            if single_fig:
                single_figs.append(single_fig)
        
        return combined_fig, single_figs
    
    def create_vertical_legend(
        self, 
        protein_type='POI',
        molecule_type="PROTAC", 
        add_threshold=True,
        width=2, 
        height=6, 
        save=True, 
        filename=None
    ):
        """
        Create a standalone vertical legend figure for POI/E3L plots.
        
        Args:
            protein_type (str): 'POI' or 'E3L' to determine which legend to create
            molecule_type (str): 'PROTAC' or 'MOLECULAR GLUE' to determine color mapping
            add_threshold (bool): Whether to include threshold line in legend
            width (float): Width of the legend figure
            height (float): Height of the legend figure 
            save (bool): Whether to save the figure
            filename (str): Filename for saving (auto-generated if None)
            
        Returns:
            fig: The created legend figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Get appropriate groups and color maps based on protein type and molecule type
        if protein_type == 'POI':
            if molecule_type == "PROTAC":
                groups = self.PROTAC_POI_GROUPS
                color_map = self.PROTAC_POI_COLOR_MAP
                legend_order = ["Kinases", "Nuclear_Regulators", "Signaling_Modulators", 
                               "Apoptosis_Regulators", "Diverse_Enzymes"]
            else:  # MOLECULAR GLUE
                groups = self.MG_POI_GROUPS
                color_map = self.MG_POI_COLOR_MAP
                legend_order = ["Kinases", "Nuclear_Regulators", "Transcription_Factors",
                               "RNA_Translation_Regulators", "Signaling_Metabolism"]
        else:  # E3L
            if molecule_type == "PROTAC":
                groups = self.PROTAC_E3_GROUPS
                color_map = self.PROTAC_E3_COLOR_MAP
                legend_order = ["CRBN", "VHL", "BIRC2", "DCAF1"]
            else:  # MOLECULAR GLUE
                groups = self.MG_E3_GROUPS
                color_map = self.MG_E3_COLOR_MAP
                legend_order = ["CRBN", "VHL", "TRIM_Ligase", "DCAF_Receptors", "Others"]
        
        # Create legend patches for groups
        legend_handles = []
        for group in legend_order:
            display_name = group.replace('_', ' ')
            color = color_map.get(group, 'gray')
            patch = plt.Rectangle(
                (0, 0), 1, 1,
                facecolor=color,
                edgecolor=self.CLS_BAR_EDGE_COLOR,
                linewidth=self.CLS_BAR_LINEWIDTH,
                label=display_name
            )
            legend_handles.append(patch)
        
        # Add threshold line to legend if requested
        if add_threshold:
            threshold_line = plt.Line2D(
                [0, 1], [0, 0],
                color='gray',
                linestyle='--',
                linewidth=self.CLS_THRESHOLD_LINE_WIDTH,
                label='Threshold'
            )
            legend_handles.append(threshold_line)
        
        # Create vertical legend (ncol=1 for vertical arrangement)
        legend = ax.legend(
            handles=legend_handles,
            loc='center',
            ncol=1,  # Single column for vertical layout
            frameon=False,
            fontsize=self.CLS_LEGEND_FONTSIZE_GRID,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=1.0
        )
        
        # Remove all axes and spines
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Adjust layout to center the legend
        plt.tight_layout()
        
        # Generate filename if not provided
        if filename is None:
            filename = f"poi_e3l_legend_vertical_{protein_type.lower()}_{molecule_type.lower().replace(' ', '_')}"
        
        # Save if requested
        if save:
            self.save_plot(fig, filename)
            
        return fig
    
    def create_horizontal_legend(
        self, 
        protein_type='POI',
        molecule_type="PROTAC", 
        add_threshold=True,
        width=6, 
        height=1, 
        save=True, 
        filename=None
    ):
        """
        Create a standalone horizontal legend figure for POI/E3L plots.
        
        Args:
            protein_type (str): 'POI' or 'E3L' to determine which legend to create
            molecule_type (str): 'PROTAC' or 'MOLECULAR GLUE' to determine color mapping
            add_threshold (bool): Whether to include threshold line in legend
            width (float): Width of the legend figure
            height (float): Height of the legend figure 
            save (bool): Whether to save the figure
            filename (str): Filename for saving (auto-generated if None)
            
        Returns:
            fig: The created legend figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Get appropriate groups and color maps based on protein type and molecule type
        if protein_type == 'POI':
            if molecule_type == "PROTAC":
                groups = self.PROTAC_POI_GROUPS
                color_map = self.PROTAC_POI_COLOR_MAP
                legend_order = ["Kinases", "Nuclear_Regulators", "Signaling_Modulators", 
                               "Apoptosis_Regulators", "Diverse_Enzymes"]
            else:  # MOLECULAR GLUE
                groups = self.MG_POI_GROUPS
                color_map = self.MG_POI_COLOR_MAP
                legend_order = ["Kinases", "Nuclear_Regulators", "Transcription_Factors",
                               "RNA_Translation_Regulators", "Signaling_Metabolism"]
        else:  # E3L
            if molecule_type == "PROTAC":
                groups = self.PROTAC_E3_GROUPS
                color_map = self.PROTAC_E3_COLOR_MAP
                legend_order = ["CRBN", "VHL", "BIRC2", "DCAF1"]
            else:  # MOLECULAR GLUE
                groups = self.MG_E3_GROUPS
                color_map = self.MG_E3_COLOR_MAP
                legend_order = ["CRBN", "VHL", "TRIM_Ligase", "DCAF_Receptors", "Others"]
        
        # Create legend patches for groups
        legend_handles = []
        for group in legend_order:
            display_name = group.replace('_', ' ')
            color = color_map.get(group, PlotConfig.GRAY)
            patch = plt.Rectangle(
                (0, 0), 1, 1,
                facecolor=color,
                edgecolor=self.CLS_BAR_EDGE_COLOR,
                linewidth=self.CLS_BAR_LINEWIDTH,
                label=display_name
            )
            legend_handles.append(patch)
        
        # Add threshold line to legend if requested
        if add_threshold:
            threshold_line = plt.Line2D(
                [0, 1], [0, 0],
                color='gray',
                linestyle='--',
                linewidth=self.CLS_THRESHOLD_LINE_WIDTH,
                label='Threshold'
            )
            legend_handles.append(threshold_line)
        
        # Create horizontal legend
        ncol = len(legend_handles)  # All items in one row
        legend = ax.legend(
            handles=legend_handles,
            loc='center',
            ncol=ncol,
            frameon=False,
            fontsize=self.CLS_LEGEND_FONTSIZE_GRID,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=1.0
        )
        
        # Remove all axes and spines
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Adjust layout to center the legend
        plt.tight_layout()
        
        # Generate filename if not provided
        if filename is None:
            filename = f"poi_e3l_legend_{protein_type.lower()}_{molecule_type.lower().replace(' ', '_')}"
        
        # Save if requested
        if save:
            self.save_plot(fig, filename)
            
        return fig