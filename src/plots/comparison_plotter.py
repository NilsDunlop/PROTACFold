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
        threshold_value=4.0,
        width=12,
        height=9,
        max_structures=20,
        save=True,
        title_suffix="",
        molecule_type="PROTAC"  # Default to PROTAC
    ):
        """
        Create simplified horizontal bar plots comparing AlphaFold3 and Boltz1 predictions.
        Shows 4 bars: AF3 CCD, AF3 SMILES, Boltz1 CCD, Boltz1 SMILES.
        
        Args:
            df (pd.DataFrame): Combined results dataframe
            metric_type (str): Metric to plot ('RMSD' or 'DOCKQ')
            model_types (list): List of model types to include (None for all)
            seeds (list): List of seeds to include (None for all)
            add_threshold (bool): Whether to add a threshold line
            threshold_value (float): Value for the threshold line
            width (int): Figure width
            height (int): Figure height
            max_structures (int): Maximum number of structures to show
            save (bool): Whether to save the plots
            title_suffix (str): Suffix to add to plot titles
            molecule_type (str): Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
        
        Returns:
            tuple: (figures, axes)
        """
        # Get filtered dataframe
        df_filtered = self._filter_data(df, molecule_type, model_types, seeds)
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
        threshold_value=4.0,
        width=10,
        height=8,
        save=True,
        molecule_type="PROTAC",  # Default to PROTAC
        specific_seed=None  # New parameter to make seed filtering optional
    ):
        """
        Create a bar plot showing the mean metric values across all structures
        for each combination of model type (AF3, Boltz1) and ligand type (CCD, SMILES).
        Can optionally filter by a specific seed value.
        
        Args:
            df (pd.DataFrame): Combined results dataframe
            metric_type (str): Metric to plot ('RMSD' or 'DOCKQ')
            model_types (list): List of model types to include (None for all)
            seeds (list): List of seeds to include (None for all)
            add_threshold (bool): Whether to add a threshold line
            threshold_value (float): Value for the threshold line
            width (int): Figure width
            height (int): Figure height
            save (bool): Whether to save the plot
            molecule_type (str): Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
            specific_seed (int): If provided, filter data to only include this seed
        
        Returns:
            fig, ax: The created figure and axis
        """
        try:
            # Check if we're filtering by a specific seed
            is_seed_specific = specific_seed is not None
            
            # If we have a specific seed, override the seeds parameter
            if is_seed_specific:
                seeds = [specific_seed]
            
            # Get filtered dataframe
            df_filtered = self._filter_data(df, molecule_type, model_types, seeds)
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
            bar_positions = [0, 0.9, 2.4, 3.3]
            bar_width = 0.6
            
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
                linewidth=0.5,
                yerr=error_values,
                error_kw={'ecolor': 'black', 'capsize': 4, 'capthick': 1, 'alpha': 0.7}
            )
            
            # Apply hatches to the SMILES bars for visual differentiation
            bars[1].set_hatch('/')  # AF3 SMILES 
            bars[3].set_hatch('/')  # Boltz-1 SMILES
            
            # Add threshold line if requested
            if add_threshold:
                threshold_line = ax.axhline(
                    y=threshold_value,
                    color='black',
                    linestyle='--',
                    alpha=0.7,
                    linewidth=1.0
                )
            
            # Add value labels directly on the bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + error_values[i] + 0.05,
                    f"{value:.2f}",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color='black'
                )
            
            # Create custom legend patches
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
            
            # Add threshold line to legend if requested
            if add_threshold:
                threshold_line = plt.Line2D(
                    [0, 1], [0, 0], 
                    color='gray', 
                    linestyle='--',
                    linewidth=1.0, 
                    label='Threshold'
                )
                legend_handles.append(threshold_line)
            
            # Add the legend to the top-right corner
            ax.legend(
                handles=legend_handles, 
                loc='upper right',
                facecolor='white',
                edgecolor='lightgray',
                borderpad=0.8
            )
            
            # Remove x-ticks and labels since we have the legend
            ax.set_xticks([])
            ax.set_xticklabels([])
            
            # Add subtle grid lines
            ax.grid(axis='y', linestyle='--', alpha=0.2)
            
            # Set axis labels
            ax.set_ylabel(y_label, fontweight='bold')
            
            # Create appropriate title based on whether we're filtering by seed
            if is_seed_specific:
                title = f"Mean {metric_type} (Seed {specific_seed})"
            else:
                title = f"Mean {metric_type}"
                
            fig.suptitle(title, fontsize=PlotConfig.TITLE_SIZE, fontweight='bold')
            
            # Adjust y-axis to accommodate value labels
            ymax = max([v + e * 1.5 for v, e in zip(values, error_values)])
            if metric_type == 'RMSD' and threshold_value >= 4.0:
                ax.set_ylim(0, 4.5)
            else:
                ax.set_ylim(0, ymax * 1.1)
            
            # Use tight layout for better spacing and reduce space between plot and title
            plt.tight_layout()
            fig.subplots_adjust(top=0.95)
            
            # Add a subtle border to the plot
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('gray')
            
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
        threshold_value=4.0,
        width=10,
        height=8,
        save=True,
        molecule_type="PROTAC"  # Default to PROTAC
    ):
        """
        Create a bar plot showing metric values for a specific seed value.
        This allows direct comparison between AlphaFold3 and Boltz1 using the same seed.
        
        This is a wrapper around plot_mean_comparison with specific_seed parameter.
        
        Args:
            df (pd.DataFrame): Combined results dataframe
            metric_type (str): Metric to plot ('RMSD' or 'DOCKQ')
            model_types (list): List of model types to include (None for all)
            specific_seed (int): The specific seed value to filter by
            add_threshold (bool): Whether to add a threshold line
            threshold_value (float): Value for the threshold line
            width (int): Figure width
            height (int): Figure height
            save (bool): Whether to save the plot
            molecule_type (str): Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
        
        Returns:
            fig, ax: The created figure and axis
        """
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

    def _filter_data(self, df, molecule_type, model_types=None, seeds=None):
        """Helper method to filter data based on common criteria."""
        # Check if MOLECULE_TYPE column exists, otherwise use TYPE column
        molecule_type_col = 'MOLECULE_TYPE' if 'MOLECULE_TYPE' in df.columns else 'TYPE'
        
        if model_types is None:
            model_types = ["AlphaFold3", "Boltz1"]
        
        if seeds is None:
            if 'SEED' in df.columns:
                seeds = sorted(df['SEED'].unique())
            else:
                seeds = [1]  # Default if no SEED column
                
        # Filter the dataframe for the specified molecule type
        if molecule_type_col in df.columns:
            df_filtered = df[df[molecule_type_col] == molecule_type].copy()
        else:
            df_filtered = df.copy()
            print("Warning: TYPE column not found, no molecule type filtering applied")
        
        # Filter for specified model types
        df_filtered = df_filtered[df_filtered['MODEL_TYPE'].isin(model_types)]
        
        # Filter for specified seeds
        if 'SEED' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['SEED'].isin(seeds)]
            
        # Check if we have any data after filtering
        if df_filtered.empty:
            print(f"ERROR: No data available after filtering")
            return None
            
        return df_filtered

    def _get_metric_columns(self, metric_type):
        """Get the column names for a specific metric type."""
        if metric_type.upper() == 'RMSD':
            return ('SMILES_RMSD', 'CCD_RMSD', 'RMSD (Å)')
        elif metric_type.upper() == 'DOCKQ':
            return ('SMILES_DOCKQ_SCORE', 'CCD_DOCKQ_SCORE', 'DockQ Score')
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
        bar_width = 0.18
        
        # Define positions for the 4 bar types as in the sketch
        af3_ccd_pos = -bar_width * 1.5     # Leftmost position
        af3_smiles_pos = -bar_width * 0.5  # Second from left
        boltz1_ccd_pos = bar_width * 0.5   # Third from left
        boltz1_smiles_pos = bar_width * 1.5 # Rightmost position
        
        # Create legend handles
        legend_handles = []
        
        # Track max value for y-axis scaling
        max_value = 0
        
        # For each PDB ID, plot the 4 bars
        for i, pdb_id in enumerate(sorted_pdb_ids):
            pdb_df = df[df['PDB_ID'] == pdb_id]
            
            # Get AF3 data
            af3_df = pdb_df[pdb_df['MODEL_TYPE'] == 'AlphaFold3']
            
            # Plot AF3 CCD
            if len(af3_df) > 0 and ccd_col in af3_df.columns and not af3_df[ccd_col].isna().all():
                ccd_value = af3_df[ccd_col].values[0]
                bar = ax.bar(
                    x_positions[i] + af3_ccd_pos,
                    ccd_value,
                    width=bar_width,
                    color=PlotConfig.AF3_CCD_COLOR,
                    edgecolor='black',
                    linewidth=0.5,
                    label='AF3 CCD' if i == 0 else None
                )
                if i == 0:
                    legend_handles.append(bar)
                max_value = max(max_value, ccd_value)
                
            # Plot AF3 SMILES
            if len(af3_df) > 0 and smiles_col in af3_df.columns and not af3_df[smiles_col].isna().all():
                smiles_value = af3_df[smiles_col].values[0]
                bar = ax.bar(
                    x_positions[i] + af3_smiles_pos,
                    smiles_value,
                    width=bar_width,
                    color=PlotConfig.AF3_SMILES_COLOR,
                    edgecolor='black',
                    linewidth=0.5,
                    label='AF3 SMILES' if i == 0 else None
                )
                if i == 0:
                    legend_handles.append(bar)
                max_value = max(max_value, smiles_value)
            
            # Get Boltz1 data
            boltz1_df = pdb_df[pdb_df['MODEL_TYPE'] == 'Boltz1']
            
            # Plot Boltz1 CCD
            if len(boltz1_df) > 0 and ccd_col in boltz1_df.columns and not boltz1_df[ccd_col].isna().all():
                ccd_value = boltz1_df[ccd_col].values[0]
                bar = ax.bar(
                    x_positions[i] + boltz1_ccd_pos,
                    ccd_value,
                    width=bar_width,
                    color=PlotConfig.BOLTZ1_CCD_COLOR,
                    edgecolor='black',
                    linewidth=0.5,
                    label='Boltz1 CCD' if i == 0 else None
                )
                if i == 0:
                    legend_handles.append(bar)
                max_value = max(max_value, ccd_value)
                
            # Plot Boltz1 SMILES
            if len(boltz1_df) > 0 and smiles_col in boltz1_df.columns and not boltz1_df[smiles_col].isna().all():
                smiles_value = boltz1_df[smiles_col].values[0]
                bar = ax.bar(
                    x_positions[i] + boltz1_smiles_pos,
                    smiles_value,
                    width=bar_width,
                    color=PlotConfig.BOLTZ1_SMILES_COLOR,
                    edgecolor='black',
                    linewidth=0.5,
                    label='Boltz1 SMILES' if i == 0 else None
                )
                if i == 0:
                    legend_handles.append(bar)
                max_value = max(max_value, smiles_value)
        
        # Add threshold line if requested
        if add_threshold:
            threshold_line = ax.axhline(
                y=threshold_value,
                color='black',
                linestyle='--',
                alpha=0.7,
                linewidth=1.0
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
        ax.set_xticklabels(pdb_labels, rotation=90, ha='center')
        
        # Set axis labels and title
        ax.set_ylabel(y_label)
        
        # Set the title
        title = f"AlphaFold3 vs. Boltz1 - {metric_type}{title_suffix}"
        
        # Add page information if multiple pages
        if total_pages > 1:
            title += f" (Page {page_num} of {total_pages})"
        
        fig.suptitle(title, fontsize=PlotConfig.TITLE_SIZE)
        
        # Add some padding to the y-axis maximum for better visualization
        ax.set_ylim(0, max_value * 1.1)
        
        # Add grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add legend
        ax.legend(handles=legend_handles, loc='upper right', framealpha=0.8)
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            if metric_type == 'RMSD':
                filename = f"af3_vs_boltz1_rmsd"
            else:  # DOCKQ
                filename = f"af3_vs_boltz1_dockq"
            
            # Add page number if multiple pages
            if total_pages > 1:
                filename += f"_page_{page_num}of{total_pages}"
            
            save_figure(fig, filename)
        
        return fig, ax