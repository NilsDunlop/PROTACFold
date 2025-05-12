import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from base_plotter import BasePlotter
from config import PlotConfig
from utils import save_figure

class POI_E3LPlotter(BasePlotter):
    """Class for plotting POI and E3 ligase metrics."""
    
    def __init__(self, debug=False):
        """Initialize the plotter with color maps and protein groupings."""
        super().__init__()
        self.debug = debug
        
        # PROTAC-specific POI groups
        self.PROTAC_POI_GROUPS = {
            "Kinases": ["BTK", "PTK2", "WEE1"],
            "Nuclear_Regulators": ["SMARCA2", "SMARCA4", "STAT5A", "STAT6", "BRD2", "BRD4", "BRD9", "WDR5"],
            "Signaling_Modulators": ["FKBP1A", "FKBP5", "KRAS", "PTPN2"],
            "Apoptosis_Regulators": ["BCL2", "BCL2L1"],
            "Diverse_Enzymes": ["CA2", "EPHX2", "R1AB"]
        }
        
        # PROTAC-specific E3 ligase groups
        self.PROTAC_E3_GROUPS = {
            "CRBN": ["CRBN"],
            "VHL": ["VHL"],
            "BIRC2": ["BIRC2"],
            "DCAF1": ["DCAF1"]
        }
        
        # Molecular glue POI groups
        self.MG_POI_GROUPS = {
            "Kinases": ["CDK12", "CSNK1A1", "PAK6"],
            "Nuclear_Regulators": ["BRD4", "HDAC1", "WIZ"],
            "Transcription_Factors": ["IKZF1", "IKZF2", "SALL4", "ZNFN1A2", "ZNF692"],
            "RNA_Translation_Regulators": ["RBM39", "GSPT1"],
            "Signaling_Metabolism": ["CTNNB1", "CDO1"]
        }
        
        # Molecular glue E3 ligase groups
        self.MG_E3_GROUPS = {
            "CRBN": ["CRBN", "MGR_0879"],
            "VHL": ["VHL"],
            "TRIM_Ligase": ["TRIM21"],
            "DCAF_Receptors": ["DCAF15", "DCAF16"],
            "Others": ["BTRC", "DDB1", "KBTBD4"],
        }
        
        # PROTAC-specific color maps
        self.PROTAC_POI_COLOR_MAP = {
            "Kinases": PlotConfig.SMILES_PRIMARY,
            "Nuclear_Regulators": PlotConfig.SMILES_TERTIARY,
            "Signaling_Modulators": PlotConfig.SMILES_SECONDARY,
            "Apoptosis_Regulators": PlotConfig.CCD_PRIMARY,
            "Diverse_Enzymes": PlotConfig.CCD_SECONDARY,
        }
        
        self.PROTAC_E3_COLOR_MAP = {
            "CRBN": PlotConfig.CCD_PRIMARY,
            "VHL": PlotConfig.SMILES_TERTIARY,
            "BIRC2": PlotConfig.SMILES_PRIMARY,
            "DCAF1": PlotConfig.CCD_SECONDARY,
        }
        
        # Molecular glue color maps
        self.MG_POI_COLOR_MAP = {
            "Kinases": PlotConfig.SMILES_PRIMARY,
            "Nuclear_Regulators": PlotConfig.SMILES_TERTIARY,
            "Transcription_Factors": PlotConfig.SMILES_SECONDARY,
            "RNA_Translation_Regulators": PlotConfig.CCD_PRIMARY,
            "Signaling_Metabolism": PlotConfig.CCD_SECONDARY,
        }
        
        self.MG_E3_COLOR_MAP = {
            "CRBN": PlotConfig.CCD_PRIMARY,
            "VHL": PlotConfig.SMILES_TERTIARY,
            "TRIM_Ligase": PlotConfig.SMILES_PRIMARY,
            "DCAF_Receptors": PlotConfig.CCD_SECONDARY,
            "Others": PlotConfig.GRAY, 
        }

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
        width=12,
        height=None,
        bar_width=0.7,
        save=True,
        legend_position=None,
        molecule_type="PROTAC",
        debug=False
    ):
        """
        Plot RMSD or DockQ for POIs and E3Ls.
        
        Args:
            df: DataFrame with RMSD data
            model_type: 'AlphaFold3' or 'Boltz-1'
            metric_type: 'RMSD' or 'DockQ'
            add_threshold: Whether to add a threshold line
            threshold_value: Value for the threshold line
            width: Figure width
            height: Figure height (calculated automatically if None)
            bar_width: Width of the bars
            save: Whether to save the figure
            legend_position: Position for the legend ('upper center', 'upper right', etc.)
            molecule_type: Type of molecule to filter by ('PROTAC' or 'MOLECULAR GLUE')
            debug: Enable debugging output
            
        Returns:
            tuple: (fig_poi_list, fig_e3l) - the created figures, where fig_poi_list is a list of POI figures
        """
        # Enable debugging for this call if requested
        self.debug = debug or self.debug
        
        # Verify that the dataframe is not empty
        if df.empty:
            print(f"Error: Input dataframe is empty")
            return [], None
        
        # Filter data based on model type - accept both 'Boltz1' and 'Boltz-1'
        if model_type == 'Boltz-1':
            model_variants = ['Boltz-1', 'Boltz1']
            filtered_df = df[df['MODEL_TYPE'].isin(model_variants)].copy()
        else:
            filtered_df = df[df['MODEL_TYPE'] == model_type].copy()
        
        if len(filtered_df) == 0:
            print(f"Error: No data available for model type '{model_type}'")
            return [], None
        
        # Filter by molecule type
        # Check if we have MOLECULE_TYPE column (newer datasets) or TYPE column (older datasets)
        molecule_type_col = 'MOLECULE_TYPE' if 'MOLECULE_TYPE' in filtered_df.columns else 'TYPE'
        
        if molecule_type_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[molecule_type_col] == molecule_type].copy()
            
            if filtered_df.empty:
                print(f"Error: No data available for molecule type '{molecule_type}'")
                available_types = df[df['MODEL_TYPE'] == model_type][molecule_type_col].unique()
                print(f"Available molecule types: {available_types}")
                return [], None
        else:
            print(f"Warning: No '{molecule_type_col}' column found in data, skipping molecule type filtering")
        
        # Determine valid seeds based on model type
        if model_type == 'AlphaFold3':
            valid_seeds = [24, 37, 42]
        else:  # Boltz-1
            valid_seeds = [42]
            
        # Filter by seeds
        filtered_df = filtered_df[filtered_df['SEED'].isin(valid_seeds)]
        
        if len(filtered_df) == 0:
            print(f"Error: No data available for selected seeds")
            return [], None
            
        # Select appropriate groups and color maps based on molecule type
        if molecule_type == "PROTAC":
            poi_groups = self.PROTAC_POI_GROUPS
            e3_groups = self.PROTAC_E3_GROUPS
            poi_color_map = self.PROTAC_POI_COLOR_MAP
            e3_color_map = self.PROTAC_E3_COLOR_MAP
            
            # For PROTAC, we want to default legend to upper center
            # but only do this if user hasn't specified a position
            e3_legend_position = "upper center" if legend_position is None else legend_position
        else:  # MOLECULAR GLUE
            poi_groups = self.MG_POI_GROUPS
            e3_groups = self.MG_E3_GROUPS
            poi_color_map = self.MG_POI_COLOR_MAP
            e3_color_map = self.MG_E3_COLOR_MAP
            e3_legend_position = legend_position
        
        # Process POI data
        poi_results = self._process_protein_data(filtered_df, 'SIMPLE_POI_NAME', poi_groups, metric_type)
        
        # Process E3L data
        e3l_results = self._process_protein_data(filtered_df, 'SIMPLE_E3_NAME', e3_groups, metric_type)
        
        # If no results, return early
        if not poi_results and not e3l_results:
            print("No valid POI or E3L data found after filtering")
            return [], None
        
        # Get sorted E3L results based on molecule type
        sorted_e3l_results = self._get_sorted_e3l_results(molecule_type, e3l_results)
        
        # Create POI plots
        poi_fig_list = []
        if poi_results:
            # Define ordering for POI groups
            if molecule_type == "PROTAC":
                poi_order = ["Kinases", "Nuclear_Regulators", "Signaling_Modulators",
                             "Apoptosis_Regulators", "Diverse_Enzymes"]
            else:  # MOLECULAR GLUE
                poi_order = ["Kinases", "Nuclear_Regulators", "Transcription_Factors",
                            "RNA_Translation_Regulators", "Signaling_Metabolism"]
            
            # Sort POI results by group order
            sorted_poi_results = self._sort_results_by_group_order(poi_results, poi_order)
            
            # Create the POI plots
            poi_fig_list = self._create_split_poi_plots(
                sorted_poi_results, 
                poi_color_map,
                model_type, 
                metric_type,
                add_threshold, 
                threshold_value, 
                width, 
                height, 
                bar_width, 
                save,
                legend_position=legend_position,
                molecule_type=molecule_type
            )
        
        # Create E3L plot
        fig_e3l = None
        if sorted_e3l_results:
            # Define the visual order for legend groups (top to bottom on the plot)
            if molecule_type == "PROTAC":
                legend_order = ["CRBN", "VHL", "BIRC2", "DCAF1"]
            else:  # MOLECULAR GLUE
                legend_order = ["CRBN", "VHL", "TRIM_Ligase", "DCAF_Receptors", "Others"]
                
            fig_e3l = self._create_rmsd_plot(
                sorted_e3l_results, 
                'E3L', 
                e3_color_map,
                model_type,
                metric_type,
                add_threshold, 
                threshold_value, 
                width, 
                height, 
                bar_width, 
                save,
                custom_legend_order=legend_order,
                legend_position=e3_legend_position,
                molecule_type=molecule_type
            )
        
        return poi_fig_list, fig_e3l
    
    def _get_sorted_e3l_results(self, molecule_type, e3l_results):
        """
        Sort E3L results based on molecule type for consistent visual presentation.
        
        Args:
            molecule_type: Type of molecule ("PROTAC" or "MOLECULAR GLUE")
            e3l_results: List of processed E3L data
            
        Returns:
            List of sorted E3L results
        """
        # Define E3L visual order (top to bottom on the plot)
        if molecule_type == "PROTAC":
            visual_order = ["CRBN", "VHL", "BIRC2", "DCAF1"]
        else:  # MOLECULAR GLUE
            visual_order = ["CRBN", "DCAF_Receptors", "VHL", "CRL_Others", "TRIM_Ligase"]
        
        # Sort E3L results to match the visual order (top to bottom)
        # We need to reverse this for plotting since matplotlib plots from bottom to top
        plot_order = list(reversed(visual_order))
        
        # Sort the E3L results to match the plot order (bottom to top)
        sorted_e3l_results = []
        for group in plot_order:
            group_proteins = [item for item in e3l_results if item['group'] == group]
            group_proteins.sort(key=lambda x: x['name'])
            sorted_e3l_results.extend(group_proteins)
        
        # Add any proteins not in the predefined order at the end
        remaining_proteins = [item for item in e3l_results if item['group'] not in visual_order]
        if remaining_proteins:
            remaining_proteins.sort(key=lambda x: x['name'])
            sorted_e3l_results.extend(remaining_proteins)
            
        return sorted_e3l_results
    
    def _sort_results_by_group_order(self, results, group_order):
        """
        Sort results according to defined group order.
        
        Args:
            results: List of protein result dictionaries
            group_order: List of groups in desired order
            
        Returns:
            List of sorted results
        """
        sorted_results = []
        for group in group_order:
            group_proteins = [item for item in results if item['group'] == group]
            group_proteins.sort(key=lambda x: x['name'])
            sorted_results.extend(group_proteins)
        
        # Add any proteins not in the predefined groups at the end
        remaining_proteins = [item for item in results if item['group'] not in group_order]
        if remaining_proteins:
            remaining_proteins.sort(key=lambda x: x['name'])
            sorted_results.extend(remaining_proteins)
            
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
            group = protein_to_group.get(protein, 'Other')
            
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
        Create multiple POI metric plots with at most 25 structures each.
        
        Args:
            data: Processed data from _process_protein_data
            color_map: Dictionary mapping groups to colors
            model_type: 'AlphaFold3' or 'Boltz-1'
            metric_type: 'RMSD' or 'DockQ'
            add_threshold: Whether to add a threshold line
            threshold_value: Value for the threshold line
            width: Figure width
            height: Figure height
            bar_width: Width of the bars
            save: Whether to save the figure
            legend_position: Position for the legend
            molecule_type: Type of molecule being plotted
            
        Returns:
            list: List of matplotlib figures
        """
        if not data:
            return []
            
        # Create a single plot with all POIs
        fig = self._create_rmsd_plot(
            data, 
            'POI', 
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
            legend_position=legend_position,
            molecule_type=molecule_type
        )
        return [fig] if fig else []
        
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
            bar_width: Width of the bars
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
            height = max(8, len(data) * 0.3 + 2)
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Set up positions for bars
        positions = np.arange(len(data))
        
        # Prepare data for plotting
        names = [item['name'] for item in data]
        means = [item['mean_metric'] for item in data]
        stds = [item['std_metric'] for item in data]
        groups = [item['group'] for item in data]
        colors = [color_map[group] for group in groups]
        
        # Plot bars
        bars = ax.barh(positions, means, bar_width, xerr=stds, 
                      color=colors, alpha=0.8, 
                      error_kw={'ecolor': 'black', 'capsize': 5})
        
        # Add threshold line if requested
        if add_threshold and threshold_value is not None:
            ax.axvline(x=threshold_value, color=PlotConfig.GRAY, linestyle='--', 
                      alpha=0.7, label='Threshold')
            
        # Customize the plot
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        
        # Set appropriate x-axis label based on metric type
        if metric_type == 'RMSD':
            ax.set_xlabel('Mean RMSD (Å)', fontsize=PlotConfig.AXIS_LABEL_SIZE)
        else:  # DockQ
            ax.set_xlabel('Mean DockQ Score', fontsize=PlotConfig.AXIS_LABEL_SIZE)
        
        # Add grid
        ax.grid(axis='x', alpha=PlotConfig.GRID_ALPHA)
        
        # Create legend elements
        legend_elements = self._create_legend_elements(groups, color_map, custom_legend_order, add_threshold)
        
        # Determine legend position
        legend_loc = self._determine_legend_position(
            legend_position, protein_type, metric_type, means
        )
        
        # Create the legend
        legend = ax.legend(
            handles=legend_elements,
            loc=legend_loc,
            ncol=min(3, len(legend_elements)) if protein_type == 'E3L' else min(2, len(legend_elements)),
            fontsize=12,
            framealpha=0.7,
            facecolor='white',
            columnspacing=1.0,
            handletextpad=0.5
        )
        
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
        
        if custom_legend_order:
            # Use custom order for legend groups
            for group in custom_legend_order:
                if group in groups:
                    display_name = group.replace('_', ' ')
                    legend_elements.append(Patch(facecolor=color_map[group], label=display_name))
        else:
            # Use unique groups from data in reverse order to match visual appearance
            unique_groups = []
            for group in groups:
                if group not in unique_groups:
                    unique_groups.append(group)
            
            # Reverse the order to match visual appearance in plot (top to bottom)
            unique_groups.reverse()
            for group in unique_groups:
                display_name = group.replace('_', ' ')
                legend_elements.append(Patch(facecolor=color_map[group], label=display_name))
        
        # Add threshold to legend if it exists
        if add_threshold:
            from matplotlib.lines import Line2D
            threshold_line = Line2D([0], [0], color=PlotConfig.GRAY, linestyle='--',
                                   label='Threshold')
            legend_elements.append(threshold_line)
        
        return legend_elements
    
    def _determine_legend_position(self, legend_position, protein_type, metric_type, means):
        """
        Determine best legend position based on data and plot type.
        
        Args:
            legend_position: User-specified position (if any)
            protein_type: 'POI' or 'E3L'
            metric_type: 'RMSD' or 'DockQ'
            means: List of mean values to check for positioning
            
        Returns:
            str: Legend position
        """
        # If legend position is specified, use that
        if legend_position is not None:
            return legend_position
        
        # For E3L plots, use upper center as default
        if protein_type == 'E3L':
            return 'upper center'
        
        # For other plots, determine based on data
        xmax = max(means) if means else 0
        
        # Default legend position is upper center
        legend_loc = 'upper center'
        
        # For DockQ plots, check the topmost bar
        if metric_type == 'DOCKQ':
            if len(means) > 0:
                topmost_bar = means[-1]
                threshold = 0.5 * xmax
                if topmost_bar > threshold:
                    legend_loc = 'upper right'
        else:
            # For other metrics, check top 3 bars
            top_bars = means[-3:] if len(means) >= 3 else means
            threshold = 0.4 * xmax
            
            for bar_value in top_bars:
                if bar_value > threshold:
                    legend_loc = 'upper right'
                    break
        
        return legend_loc