import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from base_plotter import BasePlotter
from config import PlotConfig
from data_loader import DataLoader
from utils import save_figure, distribute_pdb_ids, create_plot_filename

class ComparisonPlotter(BasePlotter):
    """Class for creating comparison plots between different model types."""
    
    def __init__(self):
        """Initialize the comparison plotter."""
        super().__init__()
    
    def plot_af3_vs_boltz1(
        self, 
        df, 
        metric_type='RMSD', 
        model_types=None,
        seeds=None,
        add_threshold=True,
        threshold_value=None,
        width=None,
        height=None,
        max_structures=None,
        save=True,
        title_suffix="",
        molecule_type="PROTAC"
    ):
        """
        Create simplified horizontal bar plots comparing AlphaFold3 and Boltz1 predictions.
        Shows 4 bars: AF3 CCD, AF3 SMILES, Boltz1 CCD, Boltz1 SMILES.
        
        Args:
            df (pd.DataFrame): Combined results dataframe
            metric_type (str): Metric to plot ('RMSD', 'DOCKQ', or 'PROTAC_RMSD')
            model_types (list): List of model types to include (None for all)
            seeds (list): List of seeds to include (None for all)
            add_threshold (bool): Whether to add a threshold line
            threshold_value (float): Value for the threshold line, if None will use default based on metric_type
            width (int): Figure width (overrides default)
            height (int): Figure height (overrides default)
            max_structures (int): Maximum number of structures to show (overrides default)
            save (bool): Whether to save the plots
            title_suffix (str): Suffix to add to plot titles
            molecule_type (str): Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
        
        Returns:
            tuple: (figures, axes)
        """
        width = width if width is not None else PlotConfig.STANDARD_WIDTH
        height = height if height is not None else PlotConfig.STANDARD_HEIGHT
        max_structures = max_structures if max_structures is not None else PlotConfig.MAX_STRUCTURES_PER_PAGE
        
        if threshold_value is None:
            if metric_type.upper() == 'RMSD':
                threshold_value = PlotConfig.DEFAULT_RMSD_THRESHOLD
            elif metric_type.upper() == 'DOCKQ':
                threshold_value = PlotConfig.DEFAULT_DOCKQ_THRESHOLD
            elif metric_type.upper() == 'PROTAC_RMSD':
                threshold_value = PlotConfig.DEFAULT_PROTAC_RMSD_THRESHOLD
            elif metric_type.upper() == 'PTM':
                threshold_value = PlotConfig.DEFAULT_PTM_THRESHOLD
            else:
                threshold_value = PlotConfig.DEFAULT_RMSD_THRESHOLD
        
        df_filtered = DataLoader.filter_comparison_data(
            df, 
            molecule_type, 
            model_types, 
            seeds, 
            metric_type, 
            self._get_metric_columns
        )
        if df_filtered is None or df_filtered.empty:
            return None, None
        
        if not title_suffix and molecule_type:
            title_suffix = f" ({molecule_type})"
        
        try:
            metric_columns = self._get_metric_columns(metric_type)
            if not metric_columns:
                print(f"ERROR: Metric type '{metric_type}' not supported.")
                return None, None
                
            smiles_col, ccd_col, axis_label = metric_columns
            
            if smiles_col not in df_filtered.columns or ccd_col not in df_filtered.columns:
                print(f"ERROR: Required metric columns not found in dataframe")
                return None, None
            
            pdb_ids = df_filtered['PDB_ID'].unique()
            
            max_pdb_per_page = max_structures
            
            if len(pdb_ids) > max_pdb_per_page:
                paginated_pdb_ids = distribute_pdb_ids(pdb_ids, max_pdb_per_page)
                
                all_figures = []
                all_axes = []
                
                for page_num, page_pdb_ids in enumerate(paginated_pdb_ids, 1):
                    page_df = df_filtered[df_filtered['PDB_ID'].isin(page_pdb_ids)]
                    
                    fig, ax = self._create_comparison_plot(
                        page_df, model_types, metric_type,
                        add_threshold, threshold_value,
                        width, height, save,
                        page_num=page_num, 
                        total_pages=len(paginated_pdb_ids),
                        title_suffix=title_suffix
                    )
                    
                    all_figures.append(fig)
                    all_axes.append(ax)
                    
                return all_figures, all_axes
            else:
                fig, ax = self._create_comparison_plot(
                    df_filtered, model_types, metric_type,
                    add_threshold, threshold_value,
                    width, height, save,
                    title_suffix=title_suffix
                )
                
                return [fig], [ax]
        
        except Exception as e:
            print(f"Error in plot_af3_vs_boltz1: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def plot_mean_comparison(
        self, 
        df, 
        metric_type='RMSD', 
        model_types=None,
        seeds=None,
        add_threshold=True,
        threshold_value=None,
        width=None,
        height=None,
        save=True,
        molecule_type="PROTAC",
        specific_seed=None
    ):
        """
        Create a bar plot showing the mean metric values across all structures
        for each combination of model type (AF3, Boltz1) and ligand type (CCD, SMILES).
        Can optionally filter by a specific seed value.
        
        Args:
            df (pd.DataFrame): Combined results dataframe
            metric_type (str): Metric to plot ('RMSD', 'DOCKQ', or 'PROTAC_RMSD')
            model_types (list): List of model types to include (None for all)
            seeds (list): List of seeds to include (None for all)
            add_threshold (bool): Whether to add a threshold line
            threshold_value (float): Value for the threshold line, if None will use default based on metric_type
            width (int): Figure width (overrides default)
            height (int): Figure height (overrides default)
            save (bool): Whether to save the plot
            molecule_type (str): Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
            specific_seed (int): If provided, filter data to only include this seed
        
        Returns:
            fig, ax: The created figure and axis
        """
        try:
            width = width if width is not None else PlotConfig.STANDARD_WIDTH
            height = height if height is not None else PlotConfig.STANDARD_HEIGHT
            
            if threshold_value is None:
                if metric_type.upper() == 'RMSD':
                    threshold_value = PlotConfig.DEFAULT_RMSD_THRESHOLD
                elif metric_type.upper() == 'DOCKQ':
                    threshold_value = PlotConfig.DEFAULT_DOCKQ_THRESHOLD
                elif metric_type.upper() == 'PROTAC_RMSD':
                    threshold_value = PlotConfig.DEFAULT_PROTAC_RMSD_THRESHOLD
                elif metric_type.upper() == 'PTM':
                    threshold_value = PlotConfig.DEFAULT_PTM_THRESHOLD
                else:
                    threshold_value = PlotConfig.DEFAULT_RMSD_THRESHOLD
            
            is_seed_specific = specific_seed is not None
            
            if is_seed_specific:
                seeds = [specific_seed]
            
            df_filtered = DataLoader.filter_comparison_data(
                df, 
                molecule_type, 
                model_types, 
                seeds, 
                metric_type,
                self._get_metric_columns
            )
            if df_filtered is None or df_filtered.empty:
                print(f"Error: No data available for molecule type '{molecule_type}' after filtering")
                if is_seed_specific:
                    print(f"No data found with seed {specific_seed}")
                return None, None
            
            metric_columns = self._get_metric_columns(metric_type)
            if not metric_columns:
                print(f"Error: Metric type '{metric_type}' not supported.")
                return None, None
                
            smiles_col, ccd_col, y_label = metric_columns
            
            if smiles_col not in df_filtered.columns or ccd_col not in df_filtered.columns:
                print(f"Error: Required metric columns ({smiles_col}, {ccd_col}) not found in dataframe. Available columns: {', '.join(df_filtered.columns)}")
                return None, None
            
            fig, ax = plt.subplots(figsize=(width, height))
            
            try:
                # Use DataLoader utility to calculate metrics
                metrics = DataLoader.calculate_comparison_metrics(
                    df_filtered, 
                    model_types, 
                    metric_columns
                )
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                import traceback
                traceback.print_exc()
                return None, None
            
            means = metrics['means']
            errors = metrics['errors']
            counts = metrics['counts']
            
            bar_width = PlotConfig.BAR_WIDTH
            spacing_factor = PlotConfig.BAR_SPACING_FACTOR
            bar_positions = [0, bar_width*spacing_factor, bar_width*2*spacing_factor, bar_width*3*spacing_factor]
            
            colors = [
                PlotConfig.AF3_CCD_COLOR,      # AF3 CCD
                PlotConfig.AF3_SMILES_COLOR,   # AF3 SMILES
                PlotConfig.BOLTZ1_CCD_COLOR,   # Boltz1 CCD
                PlotConfig.BOLTZ1_SMILES_COLOR # Boltz1 SMILES
            ]
            
            bar_labels = [
                "AF3 CCD",
                "AF3 SMILES",
                "B1 CCD",
                "B1 SMILES"
            ]
            
            values = [
                means.get("AlphaFold3_CCD", 0),
                means.get("AlphaFold3_SMILES", 0),
                means.get("Boltz1_CCD", 0),
                means.get("Boltz1_SMILES", 0)
            ]
            
            error_values = [
                errors.get("AlphaFold3_CCD", 0),
                errors.get("AlphaFold3_SMILES", 0),
                errors.get("Boltz1_CCD", 0),
                errors.get("Boltz1_SMILES", 0)
            ]
            
            bars = ax.bar(
                bar_positions,
                values,
                bar_width,
                color=colors,
                edgecolor=PlotConfig.BAR_EDGE_COLOR,
                linewidth=PlotConfig.BAR_EDGE_LINE_WIDTH,
                yerr=error_values,
                error_kw={'ecolor': PlotConfig.ERROR_BAR_COLOR, 'capsize': PlotConfig.ERROR_BAR_CAPSIZE, 
                           'capthick': PlotConfig.ERROR_BAR_THICKNESS, 'alpha': PlotConfig.ERROR_BAR_ALPHA}
            )
            
            if add_threshold and metric_type.upper() != 'PTM':
                threshold_line = ax.axhline(
                    y=threshold_value,
                    color=PlotConfig.THRESHOLD_LINE_COLOR,
                    linestyle=PlotConfig.THRESHOLD_LINE_STYLE,
                    alpha=PlotConfig.THRESHOLD_LINE_ALPHA,
                    linewidth=PlotConfig.THRESHOLD_LINE_WIDTH
                )
            
            legend_handles = []
            
            for i, (label, color) in enumerate(zip(bar_labels, colors)):
                patch = plt.Rectangle(
                    (0, 0), 1, 1, 
                    facecolor=color, 
                    edgecolor='black', 
                    linewidth=0.5, 
                    label=label
                )
                legend_handles.append(patch)
            
            if add_threshold and metric_type.upper() != 'PTM':
                threshold_line = plt.Line2D(
                    [0, 1], [0, 0],
                    color=PlotConfig.THRESHOLD_LINE_COLOR,
                    linestyle=PlotConfig.THRESHOLD_LINE_STYLE,
                    linewidth=PlotConfig.THRESHOLD_LINE_WIDTH, 
                    label='Threshold'
                )
                legend_handles.append(threshold_line)
            
            # Add legend - COMMENTED OUT FOR NOW
            # if add_threshold and threshold_value is not None and metric_type.upper() != 'PTM' and max(values) < threshold_value * 1.5:
            #     # Use background for low-value plots where threshold line might cross the legend
            #     ax.legend(handles=legend_handles, loc='best', frameon=False, # Ensure no border
            #               facecolor='white', framealpha=0.8, edgecolor='lightgray', fontsize=self.LEGEND_FONT_SIZE, labelspacing=0.2)
            # else:
            #     # No background needed
            #     ax.legend(handles=legend_handles, loc='best', frameon=False, fontsize=self.LEGEND_FONT_SIZE, labelspacing=0.2)
            
            # Remove x-ticks and labels as legend is used instead
            ax.set_xticks([])
            ax.set_xticklabels([])
            
            ax.grid(axis='y', linestyle=PlotConfig.GRID_LINESTYLE, alpha=PlotConfig.GRID_ALPHA)
            
            ax.set_ylabel(y_label, fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
            
            if is_seed_specific:
                if metric_type.upper() in ['DOCKQ', 'PTM']:
                    title = f"Seed {specific_seed}"
                else:
                    title = f"Seed {specific_seed}"
            else:
                title = ""
                
            if title:
                fig.suptitle(title, fontsize=PlotConfig.TITLE_SIZE, fontweight='bold')
            
            ymax = max([v + e * 1.5 for v, e in zip(values, error_values)])
            if metric_type.upper() == 'RMSD' and threshold_value >= 4.0:
                ax.set_ylim(0, max(4.5, ymax * 1.1))
            elif metric_type.upper() == 'DOCKQ':
                # DockQ scores range from 0 to 1
                ax.set_ylim(0, min(1.05, max(0.5, ymax * 1.1)))
            elif metric_type.upper() == 'PROTAC_RMSD':
                # Use a larger y-axis limit for PROTAC RMSD to prevent label overlapping
                ax.set_ylim(0, 10)
            elif metric_type.upper() == 'PTM':
                # PTM scores range from 0 to 1
                ax.set_ylim(0, 1.0)
            else:
                ax.set_ylim(0, ymax * 1.1)
            
            plt.tight_layout()
            fig.subplots_adjust(top=0.95)
            
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('black')
            
            if save:
                if is_seed_specific:
                    filename = create_plot_filename('comparison', molecule_type=molecule_type, 
                                                  metric_type=metric_type, comparison_type='seed', 
                                                  seed=specific_seed)
                else:
                    filename = create_plot_filename('comparison', molecule_type=molecule_type, 
                                                  metric_type=metric_type, comparison_type='mean')
                save_figure(fig, filename)
            
            return fig, ax
        except Exception as e:
            print(f"Error in plot_mean_comparison: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def plot_seed_comparison(
        self, 
        df, 
        metric_type='RMSD', 
        model_types=None,
        specific_seed=42,
        add_threshold=True,
        threshold_value=None,
        width=None,
        height=None,
        save=True,
        molecule_type="PROTAC"
    ):
        """
        Create a bar plot showing metric values for a specific seed value.
        This allows direct comparison between AlphaFold3 and Boltz1 using the same seed.
        
        This is a wrapper around plot_mean_comparison with specific_seed parameter.
        
        Args:
            df (pd.DataFrame): Combined results dataframe
            metric_type (str): Metric to plot ('RMSD', 'DOCKQ', or 'PROTAC_RMSD')
            model_types (list): List of model types to include (None for all)
            specific_seed (int): The specific seed value to filter by
            add_threshold (bool): Whether to add a threshold line
            threshold_value (float): Value for the threshold line, if None will use default based on metric_type
            width (int): Figure width (overrides default)
            height (int): Figure height (overrides default)
            save (bool): Whether to save the plot
            molecule_type (str): Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
        
        Returns:
            fig, ax: The created figure and axis
        """
        width = width if width is not None else PlotConfig.STANDARD_WIDTH
        height = height if height is not None else PlotConfig.STANDARD_HEIGHT
        
        # No threshold for PTM
        if metric_type.upper() == 'PTM':
            add_threshold = False

        elif threshold_value is None:
            if metric_type.upper() == 'RMSD':
                threshold_value = PlotConfig.DEFAULT_RMSD_THRESHOLD
            elif metric_type.upper() == 'DOCKQ':
                threshold_value = PlotConfig.DEFAULT_DOCKQ_THRESHOLD
            elif metric_type.upper() == 'PROTAC_RMSD':
                threshold_value = PlotConfig.DEFAULT_PROTAC_RMSD_THRESHOLD
            else:
                threshold_value = PlotConfig.DEFAULT_RMSD_THRESHOLD
                
        return self.plot_mean_comparison(
            df=df,
            metric_type=metric_type,
            model_types=model_types,
            seeds=None,
            add_threshold=add_threshold,
            threshold_value=threshold_value,
            width=width,
            height=height,
            save=save,
            molecule_type=molecule_type,
            specific_seed=specific_seed
        )

    def create_horizontal_legend(self, width=6, height=1, save=True, filename=None):
        """
        Create a standalone horizontal legend figure for comparison plots.
        
        Args:
            width (float): Width of the legend figure
            height (float): Height of the legend figure 
            save (bool): Whether to save the figure
            filename (str): Filename for saving
            
        Returns:
            fig: The created legend figure
        """
        fig, ax = plt.subplots(figsize=(width, height))
        
        colors = [
            PlotConfig.AF3_CCD_COLOR,      # AF3 CCD
            PlotConfig.AF3_SMILES_COLOR,   # AF3 SMILES
            PlotConfig.BOLTZ1_CCD_COLOR,   # B1 CCD
            PlotConfig.BOLTZ1_SMILES_COLOR # B1 SMILES
        ]
        
        labels = [
            "AF3 CCD",
            "AF3 SMILES", 
            "B1 CCD",
            "B1 SMILES"
        ]
        
        legend_handles = []
        for label, color in zip(labels, colors):
            patch = plt.Rectangle(
                (0, 0), 1, 1,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                label=label
            )
            legend_handles.append(patch)
        
        threshold_line = plt.Line2D(
            [0, 1], [0, 0],
            color='grey',
            linestyle='--',
            linewidth=1.0,
            label='Threshold'
        )
        legend_handles.append(threshold_line)
        
        legend = ax.legend(
            handles=legend_handles,
            loc='center',
            ncol=5,  # 5 columns for horizontal layout (4 bars + threshold)
            frameon=False,
            fontsize=PlotConfig.LEGEND_TEXT_SIZE,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=1.0
        )
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = create_plot_filename('comparison', legend='legend')
            save_figure(fig, filename)
            
        return fig

    def _get_metric_columns(self, metric_type):
        """Get the column names for a specific metric type."""
        if metric_type.upper() == 'RMSD':
            return ('SMILES_RMSD', 'CCD_RMSD', 'RMSD (Å)')
        elif metric_type.upper() == 'DOCKQ':
            return ('SMILES_DOCKQ_SCORE', 'CCD_DOCKQ_SCORE', 'DockQ Score')
        elif metric_type.upper() == 'PROTAC_RMSD':
            return ('SMILES_PROTAC_RMSD', 'CCD_PROTAC_RMSD', 'PROTAC RMSD (Å)')
        elif metric_type.upper() == 'PTM':
            return ('SMILES_PTM', 'CCD_PTM', 'pTM Score')
        else:
            return None

    def _create_comparison_plot(
        self, df, model_types, metric_type,
        add_threshold, threshold_value,
        width, height, save,
        page_num=1, total_pages=1, title_suffix=""
    ):
        """
        Create a comparison plot between AlphaFold3 and Boltz1 with 
        side-by-side vertical bars.
        
        Args:
            df: DataFrame with filtered data
            model_types: List of model types to include
            metric_type: Type of metric to plot (RMSD or DOCKQ)
            add_threshold: Whether to add a threshold line
            threshold_value: Value for threshold line
            width, height: Figure dimensions
            save: Whether to save the figure
            page_num: Current page number when paginating
            total_pages: Total number of pages
            title_suffix: Additional text for the title
            
        Returns:
            fig, ax: The created figure and axis
        """
        metric_columns = self._get_metric_columns(metric_type)
        smiles_col, ccd_col, y_label = metric_columns
        
        fig, ax = plt.subplots(figsize=(width, height))
        
        pdb_ids = df['PDB_ID'].unique()
        pdb_release_dates = {}
        
        for pdb_id in pdb_ids:
            pdb_data = df[df['PDB_ID'] == pdb_id]
            if 'RELEASE_DATE' in pdb_data.columns:
                release_date = pd.to_datetime(pdb_data['RELEASE_DATE'].iloc[0])
                pdb_release_dates[pdb_id] = release_date
        
        sorted_pdb_ids = sorted(pdb_ids, key=lambda x: pdb_release_dates.get(x, pd.Timestamp.min))
        
        x_positions = np.arange(len(sorted_pdb_ids))
        
        bar_width = PlotConfig.BAR_WIDTH
        spacing_factor = PlotConfig.BAR_SPACING_FACTOR
        bar_positions = [0, bar_width*spacing_factor, bar_width*2*spacing_factor, bar_width*3*spacing_factor]
        
        legend_handles = []
        
        max_value = 0
        
        for i, pdb_id in enumerate(sorted_pdb_ids):
            pdb_df = df[df['PDB_ID'] == pdb_id]
            
            af3_df = pdb_df[pdb_df['MODEL_TYPE'] == 'AlphaFold3']
            
            if len(af3_df) > 0 and ccd_col in af3_df.columns and not af3_df[ccd_col].isna().all():
                ccd_value = af3_df[ccd_col].values[0]
                bar = ax.bar(
                    x_positions[i] + bar_positions[0],
                    ccd_value,
                    bar_width,
                    color=PlotConfig.AF3_CCD_COLOR,
                    edgecolor=PlotConfig.BAR_EDGE_COLOR,
                    linewidth=PlotConfig.BAR_EDGE_LINE_WIDTH,
                    label='AF3 CCD' if i == 0 else None
                )
                if i == 0:
                    legend_handles.append(bar)
                max_value = max(max_value, ccd_value)
                
            if len(af3_df) > 0 and smiles_col in af3_df.columns and not af3_df[smiles_col].isna().all():
                smiles_value = af3_df[smiles_col].values[0]
                bar = ax.bar(
                    x_positions[i] + bar_positions[1],
                    smiles_value,
                    bar_width,
                    color=PlotConfig.AF3_SMILES_COLOR,
                    edgecolor=PlotConfig.BAR_EDGE_COLOR,
                    linewidth=PlotConfig.BAR_EDGE_LINE_WIDTH,
                    label='AF3 SMILES' if i == 0 else None
                )
                if i == 0:
                    legend_handles.append(bar)
                max_value = max(max_value, smiles_value)
            
            boltz1_df = pdb_df[pdb_df['MODEL_TYPE'] == 'Boltz1']
            
            if len(boltz1_df) > 0 and ccd_col in boltz1_df.columns and not boltz1_df[ccd_col].isna().all():
                ccd_value = boltz1_df[ccd_col].values[0]
                bar = ax.bar(
                    x_positions[i] + bar_positions[2],
                    ccd_value,
                    bar_width,
                    color=PlotConfig.BOLTZ1_CCD_COLOR,
                    edgecolor=PlotConfig.BAR_EDGE_COLOR,
                    linewidth=PlotConfig.BAR_EDGE_LINE_WIDTH,
                    label='B1 CCD' if i == 0 else None
                )
                if i == 0:
                    legend_handles.append(bar)
                max_value = max(max_value, ccd_value)
                
            if len(boltz1_df) > 0 and smiles_col in boltz1_df.columns and not boltz1_df[smiles_col].isna().all():
                smiles_value = boltz1_df[smiles_col].values[0]
                bar = ax.bar(
                    x_positions[i] + bar_positions[3],
                    smiles_value,
                    bar_width,
                    color=PlotConfig.BOLTZ1_SMILES_COLOR,
                    edgecolor=PlotConfig.BAR_EDGE_COLOR,
                    linewidth=PlotConfig.BAR_EDGE_LINE_WIDTH,
                    label='B1 SMILES' if i == 0 else None
                )
                if i == 0:
                    legend_handles.append(bar)
                max_value = max(max_value, smiles_value)
        
        if add_threshold and metric_type.upper() != 'PTM':
            threshold_line = ax.axhline(
                y=threshold_value,
                color=PlotConfig.THRESHOLD_LINE_COLOR,
                linestyle=PlotConfig.THRESHOLD_LINE_STYLE,
                alpha=PlotConfig.THRESHOLD_LINE_ALPHA,
                linewidth=PlotConfig.THRESHOLD_LINE_WIDTH
            )
        
        # Create PDB labels with an asterisk for structures released after the AF3 training cutoff
        pdb_labels = []
        for pdb_id in sorted_pdb_ids:
            release_date = pdb_release_dates.get(pdb_id)
            af3_cutoff = pd.to_datetime(PlotConfig.AF3_CUTOFF) if hasattr(PlotConfig, 'AF3_CUTOFF') else pd.to_datetime('2021-09-30')
            pdb_label = f"{pdb_id}*" if release_date and release_date > af3_cutoff else pdb_id
            pdb_labels.append(pdb_label)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(pdb_labels, rotation=90, ha='center', fontsize=PlotConfig.TICK_LABEL_SIZE)
        
        ax.set_ylabel(y_label, fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
        
        title = f"AlphaFold3 vs. Boltz1 - {metric_type}{title_suffix}"
        
        if total_pages > 1:
            title += f" (Page {page_num} of {total_pages})"
        
        fig.suptitle(title, fontsize=PlotConfig.TITLE_SIZE)
        
        ax.set_ylim(0, max_value * 1.1)
        
        ax.grid(axis='y', linestyle=PlotConfig.GRID_LINESTYLE, alpha=PlotConfig.GRID_ALPHA)
        
        # Add legend - COMMENTED OUT FOR NOW
        # if add_threshold and threshold_value is not None and metric_type.upper() != 'PTM' and max_value < threshold_value * 1.5:
        #     # Use background for low-value plots where threshold line might cross the legend
        #     ax.legend(handles=legend_handles, loc='best', frameon=False, # Ensure no border
        #               facecolor='white', framealpha=0.8, edgecolor='lightgray', fontsize=self.LEGEND_FONT_SIZE, labelspacing=0.2)
        # else:
        #     ax.legend(handles=legend_handles, loc='best', frameon=False, fontsize=self.LEGEND_FONT_SIZE, labelspacing=0.2)
        
        plt.tight_layout()
        
        if save:
            if metric_type.upper() == 'RMSD':
                filename = f"af3_vs_boltz1_rmsd"
            elif metric_type.upper() == 'DOCKQ':
                filename = f"af3_vs_boltz1_dockq"
            elif metric_type.upper() == 'PROTAC_RMSD':
                filename = f"af3_vs_boltz1_protac_rmsd"
            elif metric_type.upper() == 'PTM':
                filename = f"af3_vs_boltz1_ptm"
            else:
                filename = f"af3_vs_boltz1_{metric_type.lower()}"
            
            if total_pages > 1:
                filename += f"_page_{page_num}of{total_pages}"
            
            save_figure(fig, filename)
        
        return fig, ax