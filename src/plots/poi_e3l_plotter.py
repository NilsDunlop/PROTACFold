import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from base_plotter import BasePlotter
from config import PlotConfig
from utils import save_figure

class POI_E3LPlotter(BasePlotter):
    """Plotter for POI and E3L RMSD analysis."""

    def __init__(self):
        """Initialize the POI_E3L plotter."""
        super().__init__()
        
        # Define POI and E3L groups
        self.POI_GROUPS = {
            "Kinase": ["CSNK1A1", "PAK6", "CDK12", "BRAF", "BTK", "JAK2", "CAMK1D", "DYRK1A", "PLK1", "WEE1", "PTK2"],
            "Transcription_Factor": ["BCL6", "IKZF1", "IKZF2", "SALL4", "STAT5A", "STAT6", "ZNF692", "ZNFN1A2", "SMARCA2", "SMARCA4", "WIZ"],
            "Bromodomain": ["BRD2", "BRD4", "BRD9"],
            "Ubiquitin_Proteasome": ["ADRM1", "DDB1", "WDR5", "BCL2", "BCL2L1"],
            "RNA_Binding": ["RBM39", "EIF4E"],
            "Other": ["GSPT1", "CTNNB1", "CA2", "R1AB", "FKBP5", "FKBP1A", "KRAS", "POLG_CXB3N", "EPHX2", "PTPN2", "IRAK4", "CDO1", "HDAC1", "PRDX1", "NR1I2"]
        }

        self.E3_GROUPS = {
            "CRBN_Complex": ["CRBN", "MGR_0879"],
            "VHL_Complex": ["VHL"],
            "F-box_SCF_Family": ["BTRC"],
            "DDB1_CUL4_Complex": ["DDB1", "DCAF1", "DCAF15", "DCAF16"],
            "TRIM_Family": ["TRIM33", "TRIM21"],
            "Other": ["BIRC2", "KBTBD4"]
        }

        # Define color maps
        self.POI_COLOR_MAP = {
            "Kinase": PlotConfig.SMILES_PRIMARY,
            "Transcription_Factor": PlotConfig.SMILES_TERTIARY,
            "Bromodomain": PlotConfig.SMILES_SECONDARY,
            "Ubiquitin_Proteasome": PlotConfig.CCD_SECONDARY,
            "RNA_Binding": PlotConfig.CCD_PRIMARY,
            "Other": PlotConfig.GRAY
        }

        self.E3_COLOR_MAP = {
            "CRBN_Complex": PlotConfig.CCD_PRIMARY,
            "VHL_Complex": PlotConfig.SMILES_TERTIARY,
            "F-box_SCF_Family": PlotConfig.SMILES_SECONDARY,
            "DDB1_CUL4_Complex": PlotConfig.SMILES_PRIMARY,
            "TRIM_Family": PlotConfig.CCD_SECONDARY,
            "Other": PlotConfig.GRAY
        }

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
        legend_position=None
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
            
        Returns:
            tuple: (fig_poi_list, fig_e3l) - the created figures, where fig_poi_list is a list of POI figures
        """
        # Filter data based on model type - accept both 'Boltz1' and 'Boltz-1'
        if model_type == 'Boltz-1':
            model_variants = ['Boltz-1', 'Boltz1']
            filtered_df = df[df['MODEL_TYPE'].isin(model_variants)].copy()
        else:
            filtered_df = df[df['MODEL_TYPE'] == model_type].copy()
        
        if len(filtered_df) == 0:
            return [], None
        
        # Determine valid seeds based on model type
        if model_type == 'AlphaFold3':
            valid_seeds = [24, 37, 42]
        else:  # Boltz-1
            valid_seeds = [42]
            
        # Filter by seeds
        filtered_df = filtered_df[filtered_df['SEED'].isin(valid_seeds)]
        
        if len(filtered_df) == 0:
            return [], None
        
        # Process POI data
        poi_results = self._process_protein_data(filtered_df, 'SIMPLE_POI_NAME', self.POI_GROUPS, metric_type)
        
        # Process E3L data
        e3l_results = self._process_protein_data(filtered_df, 'SIMPLE_E3_NAME', self.E3_GROUPS, metric_type)
        
        # If no results, return early
        if not poi_results and not e3l_results:
            return [], None
        
        # Reorder E3L results in the desired order
        e3l_order = ["Other", "TRIM_Family", "DDB1_CUL4_Complex", "F-box_SCF_Family", "VHL_Complex", "CRBN_Complex"]
        
        # Sort the E3L results to match this order
        sorted_e3l_results = []
        for group in e3l_order:
            group_proteins = [item for item in e3l_results if item['group'] == group]
            group_proteins.sort(key=lambda x: x['name'])
            sorted_e3l_results.extend(group_proteins)
        
        legend_order = list(reversed(e3l_order))
        
        # Split POI results into multiple plots with at most 25 structures each
        poi_fig_list = []
        if poi_results:
            poi_fig_list = self._create_split_poi_plots(
                poi_results, 
                self.POI_COLOR_MAP,
                model_type, 
                metric_type,
                add_threshold, 
                threshold_value, 
                width, 
                height, 
                bar_width, 
                save,
                legend_position=legend_position
            )
        
        # Create E3L plot
        fig_e3l = None
        if sorted_e3l_results:
            fig_e3l = self._create_rmsd_plot(
                sorted_e3l_results, 
                'E3L', 
                self.E3_COLOR_MAP,
                model_type,
                metric_type,
                add_threshold, 
                threshold_value, 
                width, 
                height, 
                bar_width, 
                save,
                custom_legend_order=legend_order,
                legend_position=legend_position
            )
        
        return poi_fig_list, fig_e3l
        
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
        
        # Sort results by group and then by mean metric
        results.sort(key=lambda x: (list(groups.keys()).index(x['group']), x['mean_metric']))
        
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
        legend_position=None
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
            
        Returns:
            list: List of matplotlib figures
        """
        if not data:
            return []
            
        # Define the priority order of groups for the first plot
        priority_groups = ["Bromodomain", "Transcription_Factor", "Kinase"]
        
        # Define the order for the second plot
        secondary_groups = ["RNA_Binding", "Ubiquitin_Proteasome", "Other"]
        
        # Separate data into priority groups and other groups
        priority_data = []
        for group in priority_groups:
            group_data = [item for item in data if item['group'] == group]
            group_data.sort(key=lambda x: x['name'])
            priority_data.extend(group_data)
            
        # Order the other data by the secondary groups order
        other_data = []
        for group in secondary_groups:
            group_data = [item for item in data if item['group'] == group]
            group_data.sort(key=lambda x: x['name'])
            other_data.extend(group_data)
        
        # Calculate how many structures to include in the first plot
        if len(priority_data) <= 25:
            first_plot_data = priority_data
            second_plot_data = other_data
        else:
            # Need to limit first plot to 25 structures
            # Find a good split point that doesn't break a group
            current_group = priority_data[0]['group']
            split_index = 0
            
            for i, item in enumerate(priority_data):
                if item['group'] != current_group:
                    current_group = item['group']
                    if i >= 25:
                        split_index = i
                        break
                
            # If we didn't find a good split point, just use 25
            if split_index == 0:
                split_index = 25
                
            first_plot_data = priority_data[:split_index]
            second_plot_data = priority_data[split_index:] + other_data
        
        # Further split the second plot if necessary
        plot_data_sets = [first_plot_data]
        
        while second_plot_data:
            current_batch = second_plot_data[:25]
            plot_data_sets.append(current_batch)
            second_plot_data = second_plot_data[25:]
        
        # Create plots for each data set
        figures = []
        for i, plot_data in enumerate(plot_data_sets):
            fig = self._create_rmsd_plot(
                plot_data, 
                f'POI (Plot {i+1} of {len(plot_data_sets)})', 
                color_map,
                model_type,
                metric_type,
                add_threshold, 
                threshold_value, 
                width, 
                height, 
                bar_width, 
                save,
                suffix=f"_part{i+1}",
                legend_position=legend_position
            )
            figures.append(fig)
            
        return figures
        
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
        legend_position=None
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
        
        # Create legend for groups
        legend_elements = []
        
        if custom_legend_order:
            for group in custom_legend_order:
                if group in groups:
                    display_name = group.replace('_', ' ')
                    legend_elements.append(Patch(facecolor=color_map[group], label=display_name))
        else:
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
        if add_threshold and threshold_value is not None:
            from matplotlib.lines import Line2D
            threshold_line = Line2D([0], [0], color=PlotConfig.GRAY, linestyle='--',
                                   label='Threshold')
            legend_elements.append(threshold_line)
        
        # Calculate number of rows and columns for the legend
        num_elements = len(legend_elements)
        ncol = min(2, num_elements)
        
        # Determine legend position based on data distribution or use provided position
        if legend_position is None:
            xmax = max(means) if means else 0
            
            # Default legend position
            legend_loc = 'upper center'
            
            # For DockQ plots, check the topmost bar
            if metric_type == 'DOCKQ':
                if len(means) > 0:
                    topmost_bar = means[-1]
                    if topmost_bar > 0.5 * xmax:
                        legend_loc = 'upper right'
            else:
                top_bars = means[-3:] if len(means) >= 3 else means
                for bar_value in top_bars:
                    if bar_value > 0.4 * xmax:
                        legend_loc = 'upper right'
                        break
        else:
            legend_loc = legend_position
        
        # Create the legend
        legend = ax.legend(
            handles=legend_elements,
            loc=legend_loc,
            ncol=ncol,
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