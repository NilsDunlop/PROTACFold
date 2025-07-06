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
    CLS_DEFAULT_BAR_WIDTH = 0.6
    CLS_BAR_ALPHA = 1
    CLS_BAR_EDGE_COLOR = 'black'
    CLS_BAR_LINEWIDTH = 0.5
    CLS_ERROR_BAR_CAPSIZE = 3    
    CLS_ERROR_BAR_THICKNESS = 0.8         
    CLS_ERROR_BAR_ALPHA = 0.7             
    CLS_THRESHOLD_LINE_ALPHA = 1
    CLS_THRESHOLD_LINE_WIDTH = 1.0         

    CLS_PLOT_TITLE_FONTSIZE = 15
    CLS_TICK_FONTSIZE_GRID = 13
    CLS_LEGEND_FONTSIZE_GRID = 11
    CLS_AXIS_LABEL_FONTSIZE = 14

    CLS_LEGEND_NCOL_E3L_MAX = 2

    # Sizing for Combined Grids
    CLS_GRID_DEFAULT_WIDTH = 8.0
    CLS_GRID_HEIGHT_CALC_FACTOR = 0.4
    CLS_GRID_HEIGHT_CALC_PADDING = 3.0
    CLS_GRID_DEFAULT_OVERALL_HEIGHT = 12.0
    CLS_GRID_X_AXIS_PADDING_FACTOR = 0.05
    CLS_GRID_HSPACE = 0.05
    CLS_GRID_YLABEL_PAD = 20
    CLS_GRID_YLABEL_X_COORD = -0.45
    CLS_GRID_YLABEL_Y_COORD = 0.5
    CLS_GRID_ALPHA = 0.2
    
    # Sizing for Single Model Grids
    CLS_SINGLE_MODEL_GRID_WIDTH = CLS_GRID_DEFAULT_WIDTH / 2 
    
    # Sizing for Vertical Plots
    CLS_VERTICAL_PLOT_HEIGHT = 4.0
    CLS_VERTICAL_PLOT_WIDTH = 6.0
    
    def __init__(self, debug=False):
        """Initialize the plotter."""
        super().__init__()
        self.debug = debug

    def _debug_print(self, message):
        """Print debug information if debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] POI_E3LPlotter: {message}")
    
    def _process_protein_data(self, df, name_column, groups, metric_type='RMSD'):
        """
        Process protein data to calculate mean and std of a given metric across seeds.

        Args:
            df (pd.DataFrame): DataFrame with raw data.
            name_column (str): Column name for protein identifiers.
            groups (dict): Mapping of group names to lists of protein names.
            metric_type (str): The metric to process ('RMSD' or 'DockQ').

        Returns:
            list: A list of dictionaries, each containing processed data for a protein.
        """
        results = []
        
        smiles_col, ccd_col = ('SMILES_RMSD', 'CCD_RMSD') if metric_type == 'RMSD' else ('SMILES_DOCKQ_SCORE', 'CCD_DOCKQ_SCORE')
        
        protein_to_group = {protein: group for group, proteins in groups.items() for protein in proteins}
        
        unique_proteins = df[name_column].dropna().unique()
        
        for protein in unique_proteins:
            protein_df = df[df[name_column] == protein]
            
            # Group by seed and calculate the mean for SMILES and CCD metrics
            seed_metrics = protein_df.groupby('SEED')[[smiles_col, ccd_col]].mean()
            
            # Calculate the combined mean metric for each seed, ignoring NaNs
            seed_mean_metric = seed_metrics.mean(axis=1, skipna=True)
            
            seed_mean_metric.dropna(inplace=True)
            
            if seed_mean_metric.empty:
                continue
            
            overall_mean = seed_mean_metric.mean()
            overall_std = seed_mean_metric.std(ddof=0) if len(seed_mean_metric) > 1 else 0.0

            results.append({
                'name': protein,
                'group': protein_to_group.get(protein, 'Others'),
                'mean_metric': overall_mean,
                'std_metric': overall_std,
                'count': len(protein_df)
            })
        
        return results

    def _calculate_global_axis_lims(self, all_data, padding_factor=None):
        """
        Calculates the global minimum and maximum axis limits from multiple datasets.
        
        Args:
            all_data (list): A list of lists, where each inner list contains data dictionaries.
            padding_factor (float, optional): Factor to pad the range by. 
                                            Defaults to CLS_GRID_X_AXIS_PADDING_FACTOR.
        
        Returns:
            tuple: A tuple containing (global_min, global_max).
        """
        padding = padding_factor if padding_factor is not None else self.CLS_GRID_X_AXIS_PADDING_FACTOR
        
        global_min, global_max = float('inf'), float('-inf')

        for data_series in all_data:
            for item in data_series:
                metric_val = item['mean_metric']
                error_val = item['std_metric']
                global_min = min(global_min, metric_val - error_val)
                global_max = max(global_max, metric_val + error_val)

        if global_min == float('inf') or global_max == float('-inf'):
            return 0, 1

        range_padding = (global_max - global_min) * padding
        return max(0, global_min - range_padding), global_max + range_padding

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
        self.debug = debug or self.debug
        
        poi_data, e3l_data = self._prepare_combined_plot_data(
            df=df,
            model_types=model_types,
            metric_type=metric_type,
            molecule_type=molecule_type
        )
        
        if not poi_data or not e3l_data:
            print(f"Error: No valid POI or E3L data found after filtering")
            return None
        
        # --- Height Calculation ---
        max_poi_count = max(len(poi_data[model_type]) for model_type in model_types)
        max_e3l_count = max(len(e3l_data[model_type]) for model_type in model_types)
        
        if max_poi_count > 0 and max_e3l_count > 0:
            height_ratio = [max_poi_count, max_e3l_count]
            if height is None:
                total_data_points = max_poi_count + max_e3l_count
                height = total_data_points * self.CLS_GRID_HEIGHT_CALC_FACTOR + self.CLS_GRID_HEIGHT_CALC_PADDING
        else:
            height_ratio = [2, 1]
            if height is None:
                height = self.CLS_GRID_DEFAULT_OVERALL_HEIGHT
        
        # --- X-axis Limit Calculation ---
        global_min, global_max = self._calculate_global_axis_lims(
            [poi_data[m] for m in model_types] + [e3l_data[m] for m in model_types]
        )

        fig = plt.figure(figsize=(width, height), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=height_ratio, hspace=self.CLS_GRID_HSPACE)
        
        all_axes = []
        left_axes = []
        
        for i, model_type in enumerate(model_types):
            # --- TOP ROW: POI PLOTS ---
            ax_poi = fig.add_subplot(gs[0, i])
            all_axes.append(ax_poi)
            if i == 0:
                left_axes.append(ax_poi)
            
            poi_results = poi_data[model_type]
            sorted_poi_results = sorted(poi_results, key=lambda x: x['mean_metric'], reverse=False)
            
            self._plot_grid_data(
                ax=ax_poi,
                data=sorted_poi_results,
                color_map=self._get_color_map('POI', molecule_type),
                add_threshold=add_threshold,
                threshold_value=threshold_value,
                bar_width=bar_width,
                title=None,
                orientation='horizontal'
            )
            
            ax_poi.set_xlabel('')
            
            # --- BOTTOM ROW: E3L PLOTS ---
            ax_e3l = fig.add_subplot(gs[1, i])
            all_axes.append(ax_e3l)
            if i == 0:
                left_axes.append(ax_e3l)
            
            e3l_results = e3l_data[model_type]
            sorted_e3l_results = sorted(e3l_results, key=lambda x: x['mean_metric'], reverse=False)
            
            self._plot_grid_data(
                ax=ax_e3l,
                data=sorted_e3l_results,
                color_map=self._get_color_map('E3L', molecule_type),
                add_threshold=add_threshold,
                threshold_value=threshold_value,
                bar_width=bar_width,
                title=None,
                orientation='horizontal'
            )
            
            if metric_type == 'RMSD':
                ax_e3l.set_xlabel('RMSD (Å)', fontsize=self.CLS_AXIS_LABEL_FONTSIZE, fontweight='bold')
            else:
                ax_e3l.set_xlabel('DockQ Score', fontsize=self.CLS_AXIS_LABEL_FONTSIZE, fontweight='bold')
        
        for ax in all_axes:
            ax.set_xlim(global_min, global_max)
        
        fig.canvas.draw()
        max_tick_width = 0
        for ax in left_axes:
            for label in ax.get_yticklabels():
                bbox = label.get_window_extent()
                width_inches = bbox.width / fig.dpi
                max_tick_width = max(max_tick_width, width_inches)
        
        left_axes[0].set_ylabel('Protein of Interest', fontsize=self.CLS_AXIS_LABEL_FONTSIZE, fontweight='bold', labelpad=self.CLS_GRID_YLABEL_PAD)
        left_axes[1].set_ylabel('E3 Ligase', fontsize=self.CLS_AXIS_LABEL_FONTSIZE, fontweight='bold')
        
        left_axes[0].yaxis.set_label_coords(self.CLS_GRID_YLABEL_X_COORD, self.CLS_GRID_YLABEL_Y_COORD)
        left_axes[1].yaxis.set_label_coords(self.CLS_GRID_YLABEL_X_COORD, self.CLS_GRID_YLABEL_Y_COORD)
        
        if save:
            metric_abbr = metric_type.lower()
            models_abbr = "_".join([m.lower().replace('-', '_') for m in model_types])
            filename = f"poi_e3l_grid_{metric_abbr}_{models_abbr}"
            self.save_plot(fig, filename)
            
        return fig
    
    def _prepare_combined_plot_data(self, df, model_types, metric_type, molecule_type):
        """Helper method to prepare data for combined plots."""
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
            
            # Check if we have MOLECULE_TYPE column (newer datasets) or TYPE column (older datasets)
            molecule_type_col = 'MOLECULE_TYPE' if 'MOLECULE_TYPE' in model_df.columns else 'TYPE'
            
            if molecule_type_col in model_df.columns:
                model_df = model_df[model_df[molecule_type_col] == molecule_type].copy()
                
                if model_df.empty:
                    self._debug_print(f"Error: No data available for molecule type '{molecule_type}' with model '{model_type}'")
                    return None, None
            else:
                self._debug_print(f"Warning: No '{molecule_type_col}' column found in data, skipping molecule type filtering")
            
            valid_seeds = [24, 37, 42]
            model_df = model_df[model_df['SEED'].isin(valid_seeds)]
            
            if len(model_df) == 0:
                self._debug_print(f"Error: No data available for selected seeds with model '{model_type}'")
                return None, None
                
            filtered_dfs[model_type] = model_df
        
        if molecule_type == "PROTAC":
            poi_groups = self.PROTAC_POI_GROUPS
            e3_groups = self.PROTAC_E3_GROUPS
        else:
            poi_groups = self.MG_POI_GROUPS
            e3_groups = self.MG_E3_GROUPS
        
        poi_results = {
            model_type: self._process_protein_data(model_df, 'SIMPLE_POI_NAME', poi_groups, metric_type)
            for model_type, model_df in filtered_dfs.items()
        }
        
        e3l_results = {
            model_type: self._process_protein_data(model_df, 'SIMPLE_E3_NAME', e3_groups, metric_type)
            for model_type, model_df in filtered_dfs.items()
        }
        
        for model_type in model_types:
            if not poi_results[model_type] or not e3l_results[model_type]:
                self._debug_print(f"Warning: Missing data for model '{model_type}'")
        
        return poi_results, e3l_results
    
    def _get_color_map(self, protein_type, molecule_type):
        """Helper method to get the appropriate color map."""
        if protein_type == 'POI':
            return self.PROTAC_POI_COLOR_MAP if molecule_type == "PROTAC" else self.MG_POI_COLOR_MAP
        else:
            return self.PROTAC_E3_COLOR_MAP if molecule_type == "PROTAC" else self.MG_E3_COLOR_MAP
    
    def _plot_grid_data(self, ax, data, color_map, add_threshold, threshold_value, bar_width,
                        title=None, orientation='horizontal'):
        """
        Helper method to plot data on a given axis, supporting both horizontal and vertical bars.
        """
        positions = np.arange(len(data))
        names = [item['name'] for item in data]
        means = [item['mean_metric'] for item in data]
        stds = [item['std_metric'] for item in data]
        groups = [item['group'] for item in data]
        
        default_color = PlotConfig.GRAY
        colors = [color_map.get(group, default_color) for group in groups]
        
        error_kw = {
            'ecolor': 'black', 
            'capsize': self.CLS_ERROR_BAR_CAPSIZE, 
            'capthick': self.CLS_ERROR_BAR_THICKNESS, 
            'alpha': self.CLS_ERROR_BAR_ALPHA
        }

        if orientation == 'horizontal':
            ax.barh(positions, means, bar_width, xerr=stds, 
                    color=colors, alpha=self.CLS_BAR_ALPHA, 
                    edgecolor=self.CLS_BAR_EDGE_COLOR, linewidth=self.CLS_BAR_LINEWIDTH,
                    error_kw=error_kw)
            
            if add_threshold and threshold_value is not None:
                ax.axvline(x=threshold_value, color='gray', linestyle='--', 
                           alpha=self.CLS_THRESHOLD_LINE_ALPHA, linewidth=self.CLS_THRESHOLD_LINE_WIDTH, label='Threshold')
            
            ax.set_yticks(positions)
            ax.set_yticklabels(names, fontsize=self.CLS_TICK_FONTSIZE_GRID)
            ax.grid(axis='x', linestyle='--', alpha=self.CLS_GRID_ALPHA)
            
            y_margin = 0.6
            ax.set_ylim(positions[0] - y_margin, positions[-1] + y_margin)

        else: # Vertical
            ax.bar(positions, means, bar_width, yerr=stds, 
                   color=colors, alpha=self.CLS_BAR_ALPHA, 
                   edgecolor=self.CLS_BAR_EDGE_COLOR, linewidth=self.CLS_BAR_LINEWIDTH,
                   error_kw=error_kw)
            
            if add_threshold and threshold_value is not None:
                ax.axhline(y=threshold_value, color='gray', linestyle='--', 
                           alpha=self.CLS_THRESHOLD_LINE_ALPHA, linewidth=self.CLS_THRESHOLD_LINE_WIDTH, label='Threshold')
            
            ax.set_xticks(positions)
            ax.set_xticklabels(names, fontsize=self.CLS_TICK_FONTSIZE_GRID, rotation=90)
            ax.grid(axis='y', linestyle='--', alpha=self.CLS_GRID_ALPHA)
            
            x_margin = 0.6
            ax.set_xlim(positions[0] - x_margin, positions[-1] + x_margin)

        if title:
            ax.set_title(title, fontsize=self.CLS_PLOT_TITLE_FONTSIZE)

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
        self.debug = debug or self.debug
        
        poi_data, e3l_data = self._prepare_combined_plot_data(
            df=df,
            model_types=[model_type],
            metric_type=metric_type,
            molecule_type=molecule_type
        )
        
        if not poi_data or not e3l_data:
            print(f"Error: No valid POI or E3L data found for model {model_type} after filtering")
            return None
        
        # --- Height Calculation ---
        poi_count = len(poi_data[model_type])
        e3l_count = len(e3l_data[model_type])
        
        if poi_count > 0 and e3l_count > 0:
            height_ratio = [poi_count, e3l_count]
            if height is None:
                total_data_points = poi_count + e3l_count
                height = total_data_points * self.CLS_GRID_HEIGHT_CALC_FACTOR + self.CLS_GRID_HEIGHT_CALC_PADDING
        else:
            height_ratio = [1, 1]
            if height is None:
                height = self.CLS_GRID_DEFAULT_OVERALL_HEIGHT
        
        # --- X-axis Limit Calculation ---
        if x_lim is None:
            all_model_data = poi_data[model_type] + e3l_data[model_type]
            x_lim = self._calculate_global_axis_lims([all_model_data])
        
        fig = plt.figure(figsize=(width, height), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=height_ratio, hspace=self.CLS_GRID_HSPACE)
        
        # --- TOP ROW: POI PLOT ---
        ax_poi = fig.add_subplot(gs[0, 0])
        
        poi_results = poi_data[model_type]
        sorted_poi_results = sorted(poi_results, key=lambda x: x['mean_metric'], reverse=False)
        
        self._plot_grid_data(
            ax=ax_poi,
            data=sorted_poi_results,
            color_map=self._get_color_map('POI', molecule_type),
            add_threshold=add_threshold,
            threshold_value=threshold_value,
            bar_width=bar_width,
            title=None,
            orientation='horizontal'
        )
        
        ax_poi.set_xlim(x_lim)
        ax_poi.set_ylabel('Protein of Interest', fontsize=self.CLS_AXIS_LABEL_FONTSIZE, fontweight='bold',
                          labelpad=self.CLS_GRID_YLABEL_PAD)
        ax_poi.yaxis.set_label_coords(self.CLS_GRID_YLABEL_X_COORD, self.CLS_GRID_YLABEL_Y_COORD)
        ax_poi.set_xlabel('')
        
        # --- BOTTOM ROW: E3L PLOT ---
        ax_e3l = fig.add_subplot(gs[1, 0])
        
        e3l_results = e3l_data[model_type]
        sorted_e3l_results = sorted(e3l_results, key=lambda x: x['mean_metric'], reverse=False)
        
        self._plot_grid_data(
            ax=ax_e3l,
            data=sorted_e3l_results,
            color_map=self._get_color_map('E3L', molecule_type),
            add_threshold=add_threshold,
            threshold_value=threshold_value,
            bar_width=bar_width,
            title=None,
            orientation='horizontal'
        )
        
        ax_e3l.set_xlim(x_lim)
        ax_e3l.set_ylabel('E3 Ligase', fontsize=self.CLS_AXIS_LABEL_FONTSIZE, fontweight='bold',
                          labelpad=self.CLS_GRID_YLABEL_PAD)
        ax_e3l.yaxis.set_label_coords(self.CLS_GRID_YLABEL_X_COORD, self.CLS_GRID_YLABEL_Y_COORD)
        
        if metric_type == 'RMSD':
            ax_e3l.set_xlabel('RMSD (Å)', fontsize=self.CLS_AXIS_LABEL_FONTSIZE, fontweight='bold')
        else:
            ax_e3l.set_xlabel('DockQ Score', fontsize=self.CLS_AXIS_LABEL_FONTSIZE, fontweight='bold')
        
        if save:
            metric_abbr = metric_type.lower()
            model_abbr = model_type.lower().replace('-', '_')
            filename = f"poi_e3l_single_{metric_abbr}_{model_abbr}"
            self.save_plot(fig, filename)
            
        return fig
        
    def plot_single_model_grid_vertical(
        self,
        df,
        model_type,
        metric_type='RMSD',
        add_threshold=True,
        threshold_value=4.0,
        width=None,
        height=None,
        bar_width=CLS_DEFAULT_BAR_WIDTH,
        save=False,
        legend_position='lower right',
        molecule_type="PROTAC",
        debug=False,
        y_lim=None
    ):
        """
        Plot RMSD or DockQ for a single model's POIs and E3Ls in a 1x2 grid with VERTICAL bars.
        """
        self.debug = debug or self.debug
        
        poi_data, e3l_data = self._prepare_combined_plot_data(
            df=df,
            model_types=[model_type],
            metric_type=metric_type,
            molecule_type=molecule_type
        )
        
        if not poi_data or not e3l_data:
            print(f"Error: No valid POI or E3L data found for model {model_type} after filtering")
            return None
        
        poi_count = len(poi_data[model_type])
        e3l_count = len(e3l_data[model_type])
        
        if poi_count > 0 and e3l_count > 0:
            width_ratio = [poi_count, e3l_count]
            if width is None:
                max_proteins = max(poi_count, e3l_count)
                width = max_proteins * 0.6 + 4.0
        else:
            width_ratio = [1, 1]
            if width is None:
                width = self.CLS_VERTICAL_PLOT_WIDTH
        
        if height is None:
            height = self.CLS_VERTICAL_PLOT_HEIGHT
        
        if y_lim is None:
            all_model_data = poi_data[model_type] + e3l_data[model_type]
            y_lim = self._calculate_global_axis_lims([all_model_data])
        
        fig = plt.figure(figsize=(width, height), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=width_ratio, wspace=0.15)
        
        # --- LEFT COLUMN: POI PLOT ---
        ax_poi = fig.add_subplot(gs[0, 0])
        
        poi_results = poi_data[model_type]
        sorted_poi_results = sorted(poi_results, key=lambda x: x['mean_metric'], reverse=True)
        
        self._plot_grid_data(
            ax=ax_poi,
            data=sorted_poi_results,
            color_map=self._get_color_map('POI', molecule_type),
            add_threshold=add_threshold,
            threshold_value=threshold_value,
            bar_width=bar_width,
            title=None,
            orientation='vertical'
        )
        
        ax_poi.set_ylim(y_lim)
        ax_poi.set_xlabel('Protein of Interest', fontsize=self.CLS_AXIS_LABEL_FONTSIZE, fontweight='bold', labelpad=10)
        
        if metric_type == 'RMSD':
            ax_poi.set_ylabel('RMSD (Å)', fontsize=self.CLS_AXIS_LABEL_FONTSIZE, fontweight='bold')
        else:
            ax_poi.set_ylabel('DockQ Score', fontsize=self.CLS_AXIS_LABEL_FONTSIZE, fontweight='bold')
        
        # --- RIGHT COLUMN: E3L PLOT ---
        ax_e3l = fig.add_subplot(gs[0, 1])
        
        e3l_results = e3l_data[model_type]
        sorted_e3l_results = sorted(e3l_results, key=lambda x: x['mean_metric'], reverse=True)
        
        self._plot_grid_data(
            ax=ax_e3l,
            data=sorted_e3l_results,
            color_map=self._get_color_map('E3L', molecule_type),
            add_threshold=add_threshold,
            threshold_value=threshold_value,
            bar_width=bar_width,
            title=None,
            orientation='vertical'
        )
        
        ax_e3l.set_ylim(y_lim)
        ax_e3l.set_xlabel('E3 Ligase', fontsize=self.CLS_AXIS_LABEL_FONTSIZE, fontweight='bold', labelpad=10)
        ax_e3l.set_ylabel('')
        
        ax_poi.xaxis.set_label_coords(0.5, -0.40)
        ax_e3l.xaxis.set_label_coords(0.5, -0.40)
        
        if save:
            metric_abbr = metric_type.lower()
            model_abbr = model_type.lower().replace('-', '_')
            filename = f"poi_e3l_single_vertical_{metric_abbr}_{model_abbr}"
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
        """
        self.debug = debug or self.debug
        
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
        
        axes = combined_fig.get_axes()
        global_xlim = axes[0].get_xlim() if axes else None
        
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
    
    def plot_all_model_grids_vertical(
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
        Create individual vertical plots for each model with consistent y-axis limits.
        """
        self.debug = debug or self.debug
        
        poi_data, e3l_data = self._prepare_combined_plot_data(
            df=df,
            model_types=model_types,
            metric_type=metric_type,
            molecule_type=molecule_type
        )
        
        if not poi_data or not e3l_data:
            print("Error: No valid POI or E3L data found for calculating global y-limits")
            return []
        
        all_plot_data = [
            item for model in model_types 
            for item in poi_data.get(model, []) + e3l_data.get(model, [])
        ]
        global_ylim = self._calculate_global_axis_lims([all_plot_data])
        
        self._debug_print(f"Calculated global y-limits for vertical plots: {global_ylim}")
        
        vertical_figs = []
        for model_type in model_types:
            vertical_fig = self.plot_single_model_grid_vertical(
                df=df,
                model_type=model_type,
                metric_type=metric_type,
                add_threshold=add_threshold,
                threshold_value=threshold_value,
                save=save,
                legend_position=legend_position,
                molecule_type=molecule_type,
                debug=debug,
                y_lim=global_ylim
            )
            
            if vertical_fig:
                vertical_figs.append(vertical_fig)
                self._debug_print(f"Created vertical plot for {model_type} with y-limits: {global_ylim}")
        
        return vertical_figs
    
    def create_legend(
        self, 
        protein_type='POI',
        molecule_type="PROTAC", 
        orientation='vertical',
        add_threshold=True,
        width=None, 
        height=None, 
        save=True, 
        filename=None
    ):
        """
        Creates a standalone legend figure, supporting both vertical and horizontal layouts.
        """
        if protein_type == 'POI':
            groups = self.PROTAC_POI_GROUPS if molecule_type == "PROTAC" else self.MG_POI_GROUPS
            color_map = self.PROTAC_POI_COLOR_MAP if molecule_type == "PROTAC" else self.MG_POI_COLOR_MAP
        else:
            groups = self.PROTAC_E3_GROUPS if molecule_type == "PROTAC" else self.MG_E3_GROUPS
            color_map = self.PROTAC_E3_COLOR_MAP if molecule_type == "PROTAC" else self.MG_E3_COLOR_MAP
        legend_order = list(groups.keys())
        
        legend_handles = [
            Patch(facecolor=color_map.get(group, PlotConfig.GRAY), 
                  edgecolor=self.CLS_BAR_EDGE_COLOR, 
                  linewidth=self.CLS_BAR_LINEWIDTH,
                  label=group.replace('_', ' '))
            for group in legend_order
        ]

        if add_threshold:
            legend_handles.append(
                plt.Line2D([0], [0], color='gray', linestyle='--', 
                           linewidth=self.CLS_THRESHOLD_LINE_WIDTH, label='Threshold')
            )
        
        if orientation == 'vertical':
            ncol = 1
            fig_width = width if width is not None else 2
            fig_height = height if height is not None else len(legend_handles) * 0.5
        else:
            ncol = len(legend_handles)
            fig_width = width if width is not None else len(legend_handles) * 1.5
            fig_height = height if height is not None else 1
            
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax.legend(
            handles=legend_handles,
            loc='center',
            ncol=ncol,
            frameon=False,
            fontsize=self.CLS_LEGEND_FONTSIZE_GRID,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=1.0
        )
        ax.axis('off')
        plt.tight_layout()

        if save:
            if filename is None:
                filename = f"poi_e3l_legend_{orientation}_{protein_type.lower()}_{molecule_type.lower().replace(' ', '_')}"
            self.save_plot(fig, filename)
            
        return fig

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
        """
        return self.create_legend(
            protein_type=protein_type,
            molecule_type=molecule_type,
            orientation='vertical',
            add_threshold=add_threshold,
            width=width,
            height=height,
            save=save,
            filename=filename
        )
    
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
        """
        return self.create_legend(
            protein_type=protein_type,
            molecule_type=molecule_type,
            orientation='horizontal',
            add_threshold=add_threshold,
            width=width,
            height=height,
            save=save,
            filename=filename
        )