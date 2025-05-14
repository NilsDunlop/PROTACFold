import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from base_plotter import BasePlotter
from config import PlotConfig
from data_loader import DataLoader
from utils import save_figure, distribute_structures_evenly, distribute_pdb_ids

class ComparisonPlotter(BasePlotter):
    """Class for creating comparison plots between different model types."""
    
    # Plot dimensions
    PLOT_WIDTH = 4
    PLOT_HEIGHT = 4
    
    # Font sizes
    TITLE_FONT_SIZE = 14
    AXIS_LABEL_FONT_SIZE = 12
    TICK_LABEL_FONT_SIZE = 11
    VALUE_LABEL_FONT_SIZE = 12
    LEGEND_FONT_SIZE = 9.5
    
    # Bar appearance
    BAR_WIDTH = 0.05
    BAR_EDGE_LINE_WIDTH = 0.5
    BAR_SPACING_FACTOR = 2  # Controls spacing between bars
    
    # Error bar appearance
    ERROR_BAR_CAPSIZE = 4
    ERROR_BAR_THICKNESS = 1
    ERROR_BAR_ALPHA = 0.7
    
    # Grid properties
    GRID_ALPHA = 0.2
    GRID_LINESTYLE = '--'
    
    # Threshold line properties
    THRESHOLD_LINE_ALPHA = 1
    THRESHOLD_LINE_WIDTH = 1.0
    
    # Default threshold values
    DEFAULT_RMSD_THRESHOLD = 4.0
    DEFAULT_DOCKQ_THRESHOLD = 0.5
    DEFAULT_LRMSD_THRESHOLD = 4.0
    DEFAULT_PTM_THRESHOLD = 0.7
    
    # Maximum structures per page
    MAX_STRUCTURES_PER_PAGE = 20
    
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
            metric_type (str): Metric to plot ('RMSD', 'DOCKQ', or 'LRMSD')
            model_types (list): List of model types to include (None for all)
            seeds (list): List of seeds to include (None for all)
            add_threshold (bool): Whether to add a threshold line
            threshold_value (float): Value for the threshold line, if None will use default based on metric_type
            width (int): Figure width (overrides default)
            height (int): Figure height (overrides default)
            max_structures (int): Maximum number of structures to show (overrides default)
            width (int): Figure width (overrides default)
            height (int): Figure height (overrides default)
            max_structures (int): Maximum number of structures to show (overrides default)
            save (bool): Whether to save the plots
            title_suffix (str): Suffix to add to plot titles
            molecule_type (str): Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
        
        Returns:
            tuple: (figures, axes)
        """
        # Use class constants if parameters are not provided
        width = width if width is not None else self.PLOT_WIDTH
        height = height if height is not None else self.PLOT_HEIGHT
        max_structures = max_structures if max_structures is not None else self.MAX_STRUCTURES_PER_PAGE
        
        # Use class constants if parameters are not provided
        width = width if width is not None else self.PLOT_WIDTH
        height = height if height is not None else self.PLOT_HEIGHT
        max_structures = max_structures if max_structures is not None else self.MAX_STRUCTURES_PER_PAGE
        
        # Set default threshold values based on metric type if not provided
        if threshold_value is None:
            if metric_type.upper() == 'RMSD':
                threshold_value = self.DEFAULT_RMSD_THRESHOLD
                threshold_value = self.DEFAULT_RMSD_THRESHOLD
            elif metric_type.upper() == 'DOCKQ':
                threshold_value = self.DEFAULT_DOCKQ_THRESHOLD
                threshold_value = self.DEFAULT_DOCKQ_THRESHOLD
            elif metric_type.upper() == 'LRMSD':
                threshold_value = self.DEFAULT_LRMSD_THRESHOLD
                threshold_value = self.DEFAULT_LRMSD_THRESHOLD
            elif metric_type.upper() == 'PTM':
                threshold_value = self.DEFAULT_PTM_THRESHOLD
                threshold_value = self.DEFAULT_PTM_THRESHOLD
            else:
                threshold_value = self.DEFAULT_RMSD_THRESHOLD
                threshold_value = self.DEFAULT_RMSD_THRESHOLD
        
        # Get filtered dataframe
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
        
        # Create title suffix based on molecule type if not provided
        if not title_suffix and molecule_type:
            title_suffix = f" ({molecule_type})"
        
        try:
            # Get the appropriate metric columns based on metric_type
            metric_columns = self._get_metric_columns(metric_type)
            if not metric_columns:
                print(f"ERROR: Metric type '{metric_type}' not supported.")
                return None, None
                
            smiles_col, ccd_col, axis_label = metric_columns
            
            # Verify metric columns exist
            if smiles_col not in df_filtered.columns or ccd_col not in df_filtered.columns:
                print(f"ERROR: Required metric columns not found in dataframe")
                return None, None
            
            # Get all PDB IDs
            pdb_ids = df_filtered['PDB_ID'].unique()
            
            # If too many structures, paginate them
            max_pdb_per_page = max_structures
            
            if len(pdb_ids) > max_pdb_per_page:
                # Use the utility function from utils.py
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
                # Create a single plot with all PDB IDs
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
            metric_type (str): Metric to plot ('RMSD', 'DOCKQ', or 'LRMSD')
            model_types (list): List of model types to include (None for all)
            seeds (list): List of seeds to include (None for all)
            add_threshold (bool): Whether to add a threshold line
            threshold_value (float): Value for the threshold line, if None will use default based on metric_type
            width (int): Figure width (overrides default)
            height (int): Figure height (overrides default)
            width (int): Figure width (overrides default)
            height (int): Figure height (overrides default)
            save (bool): Whether to save the plot
            molecule_type (str): Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
            specific_seed (int): If provided, filter data to only include this seed
        
        Returns:
            fig, ax: The created figure and axis
        """
        try:
            # Use class constants if parameters are not provided
            width = width if width is not None else self.PLOT_WIDTH
            height = height if height is not None else self.PLOT_HEIGHT
            
            # Use class constants if parameters are not provided
            width = width if width is not None else self.PLOT_WIDTH
            height = height if height is not None else self.PLOT_HEIGHT
            
            # Set default threshold values based on metric type if not provided
            if threshold_value is None:
                if metric_type.upper() == 'RMSD':
                    threshold_value = self.DEFAULT_RMSD_THRESHOLD
                    threshold_value = self.DEFAULT_RMSD_THRESHOLD
                elif metric_type.upper() == 'DOCKQ':
                    threshold_value = self.DEFAULT_DOCKQ_THRESHOLD
                    threshold_value = self.DEFAULT_DOCKQ_THRESHOLD
                elif metric_type.upper() == 'LRMSD':
                    threshold_value = self.DEFAULT_LRMSD_THRESHOLD
                    threshold_value = self.DEFAULT_LRMSD_THRESHOLD
                elif metric_type.upper() == 'PTM':
                    threshold_value = self.DEFAULT_PTM_THRESHOLD
                    threshold_value = self.DEFAULT_PTM_THRESHOLD
                else:
                    threshold_value = self.DEFAULT_RMSD_THRESHOLD
                    threshold_value = self.DEFAULT_RMSD_THRESHOLD
            
            # Check if we're filtering by a specific seed
            is_seed_specific = specific_seed is not None
            
            # If we have a specific seed, override the seeds parameter
            if is_seed_specific:
                seeds = [specific_seed]
            
            # Get filtered dataframe
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
            
            # Get the appropriate metric columns based on metric_type
            metric_columns = self._get_metric_columns(metric_type)
            if not metric_columns:
                print(f"Error: Metric type '{metric_type}' not supported.")
                return None, None
                
            smiles_col, ccd_col, y_label = metric_columns
            
            # Verify metric columns exist
            if smiles_col not in df_filtered.columns or ccd_col not in df_filtered.columns:
                print(f"Error: Required metric columns ({smiles_col}, {ccd_col}) not found in dataframe. Available columns: {', '.join(df_filtered.columns)}")
                return None, None
            
            # Create figure
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
            
            # Unpack the data
            means = metrics['means']
            errors = metrics['errors']
            counts = metrics['counts']
            
            # Set up bar positions and width
            bar_width = self.BAR_WIDTH
            spacing_factor = self.BAR_SPACING_FACTOR
            bar_positions = [0, bar_width*spacing_factor, bar_width*2*spacing_factor, bar_width*3*spacing_factor]
            bar_width = self.BAR_WIDTH
            spacing_factor = self.BAR_SPACING_FACTOR
            bar_positions = [0, bar_width*spacing_factor, bar_width*2*spacing_factor, bar_width*3*spacing_factor]
            
            # Define colors for each bar
            colors = [
                PlotConfig.AF3_CCD_COLOR,      # AF3 CCD
                PlotConfig.AF3_SMILES_COLOR,   # AF3 SMILES
                PlotConfig.BOLTZ1_CCD_COLOR,   # Boltz1 CCD
                PlotConfig.BOLTZ1_SMILES_COLOR # Boltz1 SMILES
            ]
            
            # Define bar labels
            bar_labels = [
                "AF3 CCD",
                "AF3 SMILES",
                "Boltz-1 CCD",
                "Boltz-1 SMILES"
            ]
            
            # Get the mean values in the correct order
            values = [
                means.get("AlphaFold3_CCD", 0),
                means.get("AlphaFold3_SMILES", 0),
                means.get("Boltz1_CCD", 0),
                means.get("Boltz1_SMILES", 0)
            ]
            
            # Get the error values in the correct order
            error_values = [
                errors.get("AlphaFold3_CCD", 0),
                errors.get("AlphaFold3_SMILES", 0),
                errors.get("Boltz1_CCD", 0),
                errors.get("Boltz1_SMILES", 0)
            ]
            
            # Plot the bars
            bars = ax.bar(
                bar_positions,
                values,
                bar_width,
                color=colors,
                edgecolor='black',
                linewidth=self.BAR_EDGE_LINE_WIDTH,
                linewidth=self.BAR_EDGE_LINE_WIDTH,
                yerr=error_values,
                error_kw={'ecolor': 'black', 'capsize': self.ERROR_BAR_CAPSIZE, 
                           'capthick': self.ERROR_BAR_THICKNESS, 'alpha': self.ERROR_BAR_ALPHA}
            )
            
            # Add threshold line if requested and not PTM
            if add_threshold and metric_type.upper() != 'PTM':
                threshold_line = ax.axhline(
                    y=threshold_value,
                    color='gray',
                    linestyle='--',
                    alpha=self.THRESHOLD_LINE_ALPHA,
                    linewidth=self.THRESHOLD_LINE_WIDTH
                )
            
            # Create custom legend patches
            legend_handles = []
            
            # Add color patches for each bar type
            for i, (label, color) in enumerate(zip(bar_labels, colors)):
                patch = plt.Rectangle(
                    (0, 0), 1, 1, 
                    facecolor=color, 
                    edgecolor='black', 
                    linewidth=0.5,
                    label=label
                )
                legend_handles.append(patch)
            
            # Add threshold line to legend if requested and not PTM
            if add_threshold and metric_type.upper() != 'PTM':
                threshold_line = plt.Line2D(
                    [0, 1], [0, 0],
                    color='gray',
                    linestyle='--',
                    linewidth=1.0, 
                    label='Threshold'
                )
                legend_handles.append(threshold_line_handle)
            
            # Add legend
            if add_threshold and threshold_value is not None and metric_type.upper() != 'PTM' and max(values) < threshold_value * 1.5:
                # Use background for low-value plots where threshold line might cross the legend
                ax.legend(handles=legend_handles, loc='best', frameon=False, # Ensure no border
                          facecolor='white', framealpha=0.8, edgecolor='lightgray', fontsize=self.LEGEND_FONT_SIZE, labelspacing=0.2)
            else:
                # No background needed
                ax.legend(handles=legend_handles, loc='best', frameon=False, fontsize=self.LEGEND_FONT_SIZE, labelspacing=0.2)
            
            # Remove x-ticks and labels
            ax.set_xticks([])
            ax.set_xticklabels([])
            
            # Add subtle grid lines
            ax.grid(axis='y', linestyle=self.GRID_LINESTYLE, alpha=self.GRID_ALPHA)
            ax.grid(axis='y', linestyle=self.GRID_LINESTYLE, alpha=self.GRID_ALPHA)
            
            # Set axis labels
            ax.set_ylabel(y_label, fontsize=self.AXIS_LABEL_FONT_SIZE, fontweight='bold')
            ax.set_ylabel(y_label, fontsize=self.AXIS_LABEL_FONT_SIZE, fontweight='bold')
            
            # Create appropriate title based on whether we're filtering by seed
            if is_seed_specific:
                # Use proper capitalization for specific metrics
                if metric_type.upper() == 'DOCKQ':
                    title = f"Seed {specific_seed}"
                elif metric_type.upper() == 'PTM':
                    title = f"Seed {specific_seed}"
                else:
                    title = f"Seed {specific_seed}"
            else:
                # No title as requested
                title = ""
                
            # Only set a title if there's actually title text
            if title:
                fig.suptitle(title, fontsize=self.TITLE_FONT_SIZE, fontweight='bold')
                fig.suptitle(title, fontsize=self.TITLE_FONT_SIZE, fontweight='bold')
            # If title is empty, don't set a title at all
            
            # Adjust y-axis to accommodate value labels
            ymax = max([v + e * 1.5 for v, e in zip(values, error_values)])
            # Set y-axis limit appropriately based on metric type
            if metric_type.upper() == 'RMSD' and threshold_value >= 4.0:
                ax.set_ylim(0, max(4.5, ymax * 1.1))
            elif metric_type.upper() == 'DOCKQ':
                # DockQ scores range from 0 to 1
                ax.set_ylim(0, min(1.05, max(0.5, ymax * 1.1)))
            elif metric_type.upper() == 'LRMSD':
                # Use a larger y-axis limit for LRMSD to prevent label overlapping
                ax.set_ylim(0, 40)
            elif metric_type.upper() == 'PTM':
                # PTM scores range from 0 to 1
                ax.set_ylim(0, 1.0)
            else:
                ax.set_ylim(0, ymax * 1.1)
            
            # Use tight layout for better spacing and reduce space between plot and title
            plt.tight_layout()
            fig.subplots_adjust(top=0.95)
            
            # Add a subtle border to the plot
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('black')
            
            # Save the plot if requested
            if save:
                if is_seed_specific:
                    filename = f"af3_vs_boltz1_seed{specific_seed}_{molecule_type.lower()}_{metric_type.lower()}"
                else:
                    filename = f"af3_vs_boltz1_mean_{molecule_type.lower()}_{metric_type.lower()}"
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
            metric_type (str): Metric to plot ('RMSD', 'DOCKQ', or 'LRMSD')
            model_types (list): List of model types to include (None for all)
            specific_seed (int): The specific seed value to filter by
            add_threshold (bool): Whether to add a threshold line
            threshold_value (float): Value for the threshold line, if None will use default based on metric_type
            width (int): Figure width (overrides default)
            height (int): Figure height (overrides default)
            width (int): Figure width (overrides default)
            height (int): Figure height (overrides default)
            save (bool): Whether to save the plot
            molecule_type (str): Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
        
        Returns:
            fig, ax: The created figure and axis
        """
        # Use class constants if parameters are not provided
        width = width if width is not None else self.PLOT_WIDTH
        height = height if height is not None else self.PLOT_HEIGHT
        
        # Use class constants if parameters are not provided
        width = width if width is not None else self.PLOT_WIDTH
        height = height if height is not None else self.PLOT_HEIGHT
        
        # No threshold for PTM
        if metric_type.upper() == 'PTM':
            add_threshold = False

        # Set default threshold values based on metric type if not provided
        elif threshold_value is None:
            if metric_type.upper() == 'RMSD':
                threshold_value = self.DEFAULT_RMSD_THRESHOLD
                threshold_value = self.DEFAULT_RMSD_THRESHOLD
            elif metric_type.upper() == 'DOCKQ':
                threshold_value = self.DEFAULT_DOCKQ_THRESHOLD
                threshold_value = self.DEFAULT_DOCKQ_THRESHOLD
            elif metric_type.upper() == 'LRMSD':
                threshold_value = self.DEFAULT_LRMSD_THRESHOLD
                threshold_value = self.DEFAULT_LRMSD_THRESHOLD
            else:
                threshold_value = self.DEFAULT_RMSD_THRESHOLD
                threshold_value = self.DEFAULT_RMSD_THRESHOLD
                
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

    def _get_metric_columns(self, metric_type):
        """Get the column names for a specific metric type."""
        if metric_type.upper() == 'RMSD':
            return ('SMILES_RMSD', 'CCD_RMSD', 'RMSD (Å)')
        elif metric_type.upper() == 'DOCKQ':
            return ('SMILES_DOCKQ_SCORE', 'CCD_DOCKQ_SCORE', 'DockQ Score')
        elif metric_type.upper() == 'LRMSD':
            return ('SMILES_DOCKQ_LRMSD', 'CCD_DOCKQ_LRMSD', 'Ligand RMSD (Å)')
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
        # Get metric column names
        metric_columns = self._get_metric_columns(metric_type)
        smiles_col, ccd_col, y_label = metric_columns
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Get release dates for PDB IDs
        pdb_ids = df['PDB_ID'].unique()
        pdb_release_dates = {}
        
        for pdb_id in pdb_ids:
            pdb_data = df[df['PDB_ID'] == pdb_id]
            if 'RELEASE_DATE' in pdb_data.columns:
                release_date = pd.to_datetime(pdb_data['RELEASE_DATE'].iloc[0])
                pdb_release_dates[pdb_id] = release_date
        
        # Sort PDB IDs by release date
        sorted_pdb_ids = sorted(pdb_ids, key=lambda x: pdb_release_dates.get(x, pd.Timestamp.min))
        
        # Create x positions for bars
        x_positions = np.arange(len(sorted_pdb_ids))
        
        # Set up bar positions and width
        bar_width = self.BAR_WIDTH
        spacing_factor = self.BAR_SPACING_FACTOR
        bar_positions = [0, bar_width*spacing_factor, bar_width*2*spacing_factor, bar_width*3*spacing_factor]
        
        # Create legend handles
        legend_handles = []
        
        # Track max value for y-axis scaling
        max_value = 0
        
        # Define bar labels
        bar_labels = [
            "AF3 CCD",
            "AF3 SMILES",
            "Boltz-1 CCD",
            "Boltz-1 SMILES"
        ]
        
        # Define colors for each bar
        colors = [
            PlotConfig.AF3_CCD_COLOR,      # AF3 CCD
            PlotConfig.AF3_SMILES_COLOR,   # AF3 SMILES
            PlotConfig.BOLTZ1_CCD_COLOR,   # Boltz1 CCD
            PlotConfig.BOLTZ1_SMILES_COLOR # Boltz1 SMILES
        ]
        
        # For each PDB ID, plot the 4 bars
        for i, pdb_id in enumerate(sorted_pdb_ids):
            pdb_df = df[df['PDB_ID'] == pdb_id]
            
            # Get AF3 data
            af3_df = pdb_df[pdb_df['MODEL_TYPE'] == 'AlphaFold3']
            
            # Plot AF3 CCD
            if len(af3_df) > 0 and ccd_col in af3_df.columns and not af3_df[ccd_col].isna().all():
                ccd_value = af3_df[ccd_col].values[0]
                bar = ax.bar(
                    x_positions[i] + bar_positions[0],
                    x_positions[i] + bar_positions[0],
                    ccd_value,
                    bar_width,
                    color=PlotConfig.AF3_CCD_COLOR,
                    edgecolor='black',
                    linewidth=self.BAR_EDGE_LINE_WIDTH,
                    label='AF3 CCD' if i == 0 else None
                )
                max_value = max(max_value, ccd_value)
            
            # Plot AF3 SMILES
            if len(af3_df) > 0 and smiles_col in af3_df.columns and not af3_df[smiles_col].isna().all():
                smiles_value = af3_df[smiles_col].values[0]
                bar = ax.bar(
                    x_positions[i] + bar_positions[1],
                    x_positions[i] + bar_positions[1],
                    smiles_value,
                    bar_width,
                    color=PlotConfig.AF3_SMILES_COLOR,
                    edgecolor='black',
                    linewidth=self.BAR_EDGE_LINE_WIDTH,
                    label='AF3 SMILES' if i == 0 else None
                )
                max_value = max(max_value, smiles_value)
            
            # Get Boltz1 data
            boltz1_df = pdb_df[pdb_df['MODEL_TYPE'] == 'Boltz1']
            
            # Plot Boltz1 CCD
            if len(boltz1_df) > 0 and ccd_col in boltz1_df.columns and not boltz1_df[ccd_col].isna().all():
                ccd_value = boltz1_df[ccd_col].values[0]
                bar = ax.bar(
                    x_positions[i] + bar_positions[2],
                    x_positions[i] + bar_positions[2],
                    ccd_value,
                    bar_width,
                    color=PlotConfig.BOLTZ1_CCD_COLOR,
                    edgecolor='black',
                    linewidth=self.BAR_EDGE_LINE_WIDTH,
                    label='Boltz1 CCD' if i == 0 else None
                )
                max_value = max(max_value, ccd_value)
                
            # Plot Boltz1 SMILES
            if len(boltz1_df) > 0 and smiles_col in boltz1_df.columns and not boltz1_df[smiles_col].isna().all():
                smiles_value = boltz1_df[smiles_col].values[0]
                bar = ax.bar(
                    x_positions[i] + bar_positions[3],
                    x_positions[i] + bar_positions[3],
                    smiles_value,
                    bar_width,
                    color=PlotConfig.BOLTZ1_SMILES_COLOR,
                    edgecolor='black',
                    linewidth=self.BAR_EDGE_LINE_WIDTH,
                    label='Boltz1 SMILES' if i == 0 else None
                )
                max_value = max(max_value, smiles_value)
        
        # Add threshold line if requested
        if add_threshold and metric_type.upper() != 'PTM':
            threshold_line = ax.axhline(
                y=threshold_value,
                color='black',
                linestyle='--',
                alpha=self.THRESHOLD_LINE_ALPHA,
                linewidth=self.THRESHOLD_LINE_WIDTH
                alpha=self.THRESHOLD_LINE_ALPHA,
                linewidth=self.THRESHOLD_LINE_WIDTH
            )
        
        # Create PDB labels with asterisk for newer structures
        pdb_labels = []
        for pdb_id in sorted_pdb_ids:
            release_date = pdb_release_dates.get(pdb_id)
            af3_cutoff = pd.to_datetime(PlotConfig.AF3_CUTOFF) if hasattr(PlotConfig, 'AF3_CUTOFF') else pd.to_datetime('2021-09-30')
            pdb_label = f"{pdb_id}*" if release_date and release_date > af3_cutoff else pdb_id
            pdb_labels.append(pdb_label)
        
        # Set x-ticks and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(pdb_labels, rotation=90, ha='center', fontsize=self.TICK_LABEL_FONT_SIZE)
        ax.set_xticklabels(pdb_labels, rotation=90, ha='center', fontsize=self.TICK_LABEL_FONT_SIZE)
        
        # Set axis labels and title
        ax.set_ylabel(y_label, fontsize=self.AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=self.AXIS_LABEL_FONT_SIZE, fontweight='bold')
        
        # Set the title
        title = f"AlphaFold3 vs. Boltz1 - {metric_type}{title_suffix}"
        
        # Add page information if multiple pages
        if total_pages > 1:
            title += f" (Page {page_num} of {total_pages})"
        
        fig.suptitle(title, fontsize=self.TITLE_FONT_SIZE)
        fig.suptitle(title, fontsize=self.TITLE_FONT_SIZE)
        
        # Add some padding to the y-axis maximum for better visualization
        ax.set_ylim(0, max_value * 1.1)
        
        # Add grid lines
        ax.grid(axis='y', linestyle=self.GRID_LINESTYLE, alpha=self.GRID_ALPHA)
        ax.grid(axis='y', linestyle=self.GRID_LINESTYLE, alpha=self.GRID_ALPHA)
        
        # Add legend
        if add_threshold and threshold_value is not None and metric_type.upper() != 'PTM' and max_value < threshold_value * 1.5:
            # Use background for low-value plots where threshold line might cross the legend
            ax.legend(handles=legend_handles, loc='best', frameon=False, # Ensure no border
                      facecolor='white', framealpha=0.8, edgecolor='lightgray', fontsize=self.LEGEND_FONT_SIZE, labelspacing=0.2)
        else:
            ax.legend(handles=legend_handles, loc='best', frameon=False, fontsize=self.LEGEND_FONT_SIZE, labelspacing=0.2)
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            if metric_type.upper() == 'RMSD':
                filename = f"af3_vs_boltz1_rmsd"
            elif metric_type.upper() == 'DOCKQ':
                filename = f"af3_vs_boltz1_dockq"
            elif metric_type.upper() == 'LRMSD':
                filename = f"af3_vs_boltz1_lrmsd"
            elif metric_type.upper() == 'PTM':
                filename = f"af3_vs_boltz1_ptm"
            else:
                filename = f"af3_vs_boltz1_{metric_type.lower()}"
            
            # Add page number if multiple pages
            if total_pages > 1:
                filename += f"_page_{page_num}of{total_pages}"
            
            save_figure(fig, filename)
        
        return fig, ax