import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from base_plotter import BasePlotter
from config import PlotConfig
from data_loader import DataLoader
from utils import save_figure, distribute_structures_evenly

class ComparisonPlotter(BasePlotter):
    """Class for creating comparison plots between different model types."""
    
    def __init__(self):
        """Initialize the comparison plotter."""
        super().__init__()
    
    def _distribute_pdb_ids(self, pdb_ids, max_per_page):
        """
        Distribute PDB IDs evenly across multiple pages for plotting.
        
        Args:
            pdb_ids: Array of PDB IDs
            max_per_page: Maximum number of PDB IDs per page
            
        Returns:
            List of lists containing PDB IDs for each page
        """
        total_ids = len(pdb_ids)
        
        # If all IDs fit on one page, return them all
        if total_ids <= max_per_page:
            return [pdb_ids]
        
        # Calculate number of pages needed
        num_pages = math.ceil(total_ids / max_per_page)
        
        # Distribute IDs evenly
        ids_per_page = math.ceil(total_ids / num_pages)
        ids_per_page = min(ids_per_page, max_per_page)
        
        # Create batches
        batches = []
        for i in range(0, total_ids, ids_per_page):
            batch = pdb_ids[i:min(i + ids_per_page, total_ids)]
            batches.append(batch)
            
        return batches
    
    def plot_af3_vs_boltz1(
        self, 
        df, 
        metric_type='RMSD', 
        model_types=None,
        seeds=None,
        add_threshold=True,
        threshold_value=4.0,
        show_y_labels_on_all=False,
        width=9,
        height=9,
        max_structures=20,
        save=True,
        title_suffix="",
        molecule_type="PROTAC",  # Default to PROTAC
        fixed_ylim=True  # Use consistent y-axis limits across pages
    ):
        """
        Create horizontal bar plots comparing AlphaFold3 and Boltz1 predictions.
        
        Args:
            df (pd.DataFrame): Combined results dataframe
            metric_type (str): Metric to plot ('RMSD' or 'DOCKQ')
            model_types (list): List of model types to include (None for all)
            seeds (list): List of seeds to include (None for all)
            add_threshold (bool): Whether to add a threshold line
            threshold_value (float): Value for the threshold line
            show_y_labels_on_all (bool): Show y-labels on all subplots
            width (int): Figure width
            height (int): Figure height
            max_structures (int): Maximum number of structures to show
            save (bool): Whether to save the plots
            title_suffix (str): Suffix to add to plot titles
            molecule_type (str): Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
            fixed_ylim (bool): Use consistent y-axis limits across all plot pages
        
        Returns:
            tuple: (figures, axes)
        """
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
        
        # For DockQ plots, filter out PDB IDs where both Boltz-1 CCD and SMILES DOCKQ scores are 'N/A'
        if metric_type.upper() == 'DOCKQ':
            # Convert string 'N/A' values to actual NaN for proper filtering
            for col in df_filtered.columns:
                if df_filtered[col].dtype == 'object':
                    df_filtered[col] = df_filtered[col].replace('N/A', np.nan)
            
            # Identify PDB IDs where both Boltz-1 CCD and SMILES DOCKQ scores are NaN
            boltz1_df = df_filtered[df_filtered['MODEL_TYPE'] == 'Boltz1']
            problematic_pdb_ids = boltz1_df[
                boltz1_df['CCD_DOCKQ_SCORE'].isna() & 
                boltz1_df['SMILES_DOCKQ_SCORE'].isna()
            ]['PDB_ID'].unique()
            
            # Filter out these PDB IDs from the dataset
            if len(problematic_pdb_ids) > 0:
                df_filtered = df_filtered[~df_filtered['PDB_ID'].isin(problematic_pdb_ids)]
        
        # Filter for specified model types
        df_filtered = df_filtered[df_filtered['MODEL_TYPE'].isin(model_types)]
        
        # Filter for specified seeds
        if 'SEED' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['SEED'].isin(seeds)]
        
        # Create title suffix based on molecule type if not provided
        if not title_suffix and molecule_type:
            title_suffix = f" ({molecule_type})"
            
        # Group by SYSTEM and MODEL_TYPE and calculate mean for the metric
        group_cols = ['SYSTEM', 'MODEL_TYPE']
        if 'SEED' in df_filtered.columns:
            group_cols.append('SEED')
        
        try:
            # Check if we have any data after filtering
            if df_filtered.empty:
                print(f"ERROR: No data available after filtering")
                return None, None
            
            # Get the appropriate metric columns based on metric_type
            metric_columns = self._get_metric_columns(metric_type)
            if not metric_columns:
                print(f"ERROR: Metric type '{metric_type}' not supported.")
                return None, None
                
            smiles_col, ccd_col, _ = metric_columns
            
            # Verify metric columns exist
            if smiles_col not in df_filtered.columns or ccd_col not in df_filtered.columns:
                print(f"ERROR: Required metric columns not found in dataframe")
                return None, None
            
            # Now create comparison plots with the filtered data
            return self._create_comparison_plots(
                df_filtered, model_types, metric_type,
                add_threshold, threshold_value, show_y_labels_on_all,
                width, height, save, title_suffix,
                fixed_ylim
            )
        except Exception as e:
            print(f"Error in plot_af3_vs_boltz1: {e}")
            return None, None
    
    def _get_metric_columns(self, metric_type):
        """Get the column names for a specific metric type."""
        if metric_type.upper() == 'RMSD':
            return ('SMILES_RMSD', 'CCD_RMSD', 'RMSD (Å)')
        elif metric_type.upper() == 'DOCKQ':
            return ('SMILES_DOCKQ_SCORE', 'CCD_DOCKQ_SCORE', 'DockQ Score')
        else:
            return None
    
    def _create_comparison_plots(self, df, model_types, metric_type,
                              add_threshold, threshold_value,
                              show_y_labels_on_all, width, height, save,
                              title_suffix="",
                              fixed_ylim=True):
        """
        Create a set of comparison plots between different model types.
        This method is a wrapper around _create_comparison_plot that handles
        distribution of data across multiple pages if needed.
        
        Args:
            df: DataFrame containing both model types
            model_types: List of model types to include
            metric_type: Type of metric to plot
            add_threshold: Whether to add a threshold line
            threshold_value: Value for the threshold line
            show_y_labels_on_all: Whether to show y-labels on all subplots
            width, height: Figure dimensions
            save: Whether to save the figure
            title_suffix: Suffix to add to the plot title
            fixed_ylim: Use consistent y-axis limits across all plot pages
            
        Returns:
            Lists of figures and axes created
        """
        # Check for presence of required columns
        metric_columns = self._get_metric_columns(metric_type)
        if not metric_columns:
            print(f"Error: Metric type '{metric_type}' not supported.")
            return [], []
        
        smiles_col, ccd_col, _ = metric_columns
        
        # Check if these columns exist in the dataframe
        if smiles_col not in df.columns or ccd_col not in df.columns:
            print(f"ERROR: Required columns not found in dataframe.")
            return [], []
        
        # Get PDB IDs
        pdb_ids = df['PDB_ID'].unique()
        
        if len(pdb_ids) == 0:
            print("ERROR: No PDB IDs found in filtered DataFrame")
            return [], []
        
        # Check if we need multiple pages
        max_pdbs_per_page = 30  # Avoid crowding the plot
        if len(pdb_ids) > max_pdbs_per_page:
            # Distribute PDB IDs across multiple pages using our custom method
            batched_pdb_ids = self._distribute_pdb_ids(pdb_ids, max_pdbs_per_page)
            
            all_figures = []
            all_axes = []
            
            for page_num, batch_pdb_ids in enumerate(batched_pdb_ids, 1):
                # Filter DataFrame for this batch of PDB IDs
                batch_df = df[df['PDB_ID'].isin(batch_pdb_ids)]
                
                # Create plot for this batch
                figs, axes = self._create_comparison_plot(
                    batch_df, model_types, metric_type,
                    add_threshold, threshold_value, show_y_labels_on_all,
                    width, height, save, page_num, len(batched_pdb_ids), title_suffix,
                    fixed_ylim
                )
                
                all_figures.extend(figs)
                all_axes.extend(axes)
            
            return all_figures, all_axes
        else:
            # Create a single plot
            return self._create_comparison_plot(
                df, model_types, metric_type,
                add_threshold, threshold_value, show_y_labels_on_all,
                width, height, save, 1, 1, title_suffix,
                fixed_ylim
            )
    
    def _create_comparison_plot(self, df, model_types, metric_type,
                              add_threshold, threshold_value,
                              show_y_labels_on_all, width, height, save,
                              page_num=1, total_pages=1, title_suffix="",
                              fixed_ylim=True):
        """
        Create a comparison plot between different model types.
        
        Args:
            df: DataFrame containing both model types
            model_types: List of model types to include
            metric_type: Type of metric to plot
            add_threshold: Whether to add a threshold line
            threshold_value: Value for the threshold line
            show_y_labels_on_all: Whether to show y-labels on all subplots
            width, height: Figure dimensions
            save: Whether to save the figure
            page_num: Current page number
            total_pages: Total number of pages
            title_suffix: Suffix to add to the plot title
            fixed_ylim: Use consistent y-axis limits across all plot pages
            
        Returns:
            Lists of created figures and axes
        """
        # Unpack metric columns
        metric_columns = self._get_metric_columns(metric_type)
        if not metric_columns:
            print(f"Error: Metric type '{metric_type}' not supported.")
            return [], []
            
        smiles_col, ccd_col, axis_label = metric_columns
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Get unique PDB IDs in this batch
        pdb_ids = df['PDB_ID'].unique()
        
        # Get release dates for sorting
        pdb_release_dates = {}
        for pdb_id in pdb_ids:
            pdb_data = df[df['PDB_ID'] == pdb_id]
            if 'RELEASE_DATE' in pdb_data.columns:
                release_date = pd.to_datetime(pdb_data['RELEASE_DATE'].iloc[0])
                pdb_release_dates[pdb_id] = release_date
        
        # Sort by release date
        sorted_pdb_ids = sorted(pdb_ids, key=lambda x: pdb_release_dates.get(x, pd.Timestamp.min))
        
        # Set up positions for the bars
        n_pdbs = len(sorted_pdb_ids)
        y_positions = np.arange(n_pdbs)
        
        # Create PDB labels (with asterisk for newer structures)
        pdb_labels = []
        for pdb_id in sorted_pdb_ids:
            release_date = pdb_release_dates.get(pdb_id)
            af3_cutoff = pd.to_datetime(PlotConfig.AF3_CUTOFF) if hasattr(PlotConfig, 'AF3_CUTOFF') else pd.to_datetime('2021-09-30')
            pdb_label = f"{pdb_id}*" if release_date and release_date > af3_cutoff else pdb_id
            pdb_labels.append(pdb_label)
        
        # Set up bar positions and width for vertical bars
        # AF3 to the left of PDB ID, Boltz1 to the right
        bar_width = 0.18
        bar_positions = {
            # AF3 on the left side
            'AF3_CCD': -bar_width * 3/4,      # Inner left
            'AF3_SMILES': -bar_width * 9/4,   # Outer left
            # Boltz1 on the right side
            'Boltz1_CCD': bar_width * 3/4,    # Inner right
            'Boltz1_SMILES': bar_width * 9/4, # Outer right
        }
        
        # Track structures that exceed thresholds (for bolding labels)
        exceed_threshold = np.zeros(n_pdbs, dtype=bool)
        
        # Plot bars for each model type and metric
        for i, pdb_id in enumerate(sorted_pdb_ids):
            pdb_data = df[df['PDB_ID'] == pdb_id]
            
            # Plot AF3 data (if available)
            af3_data = pdb_data[pdb_data['MODEL_TYPE'] == 'AlphaFold3']
            
            if len(af3_data) > 0:
                # Plot AF3 CCD
                if ccd_col in af3_data.columns:
                    if not af3_data[ccd_col].isna().all():
                        ccd_value = af3_data[ccd_col].values[0]
                        ax.bar(
                            y_positions[i] + bar_positions['AF3_CCD'],
                            ccd_value,
                            width=bar_width,
                            color=PlotConfig.AF3_CCD_COLOR,
                            edgecolor='black',
                            linewidth=0.5,
                            orientation='vertical'
                        )
                        
                        # Check threshold based on metric type
                        if metric_type == 'RMSD' and ccd_value > threshold_value:
                            exceed_threshold[i] = True
                        elif metric_type == 'DOCKQ' and ccd_value > threshold_value:
                            exceed_threshold[i] = True
                
                # Plot AF3 SMILES
                if smiles_col in af3_data.columns:
                    if not af3_data[smiles_col].isna().all():
                        smiles_value = af3_data[smiles_col].values[0]
                        ax.bar(
                            y_positions[i] + bar_positions['AF3_SMILES'],
                            smiles_value,
                            width=bar_width,
                            color=PlotConfig.AF3_SMILES_COLOR,
                            edgecolor='black',
                            linewidth=0.5,
                            orientation='vertical'
                        )
                        
                        # Check threshold based on metric type
                        if metric_type == 'RMSD' and smiles_value > threshold_value:
                            exceed_threshold[i] = True
                        elif metric_type == 'DOCKQ' and smiles_value > threshold_value:
                            exceed_threshold[i] = True
            
            # Plot Boltz1 data (if available)
            boltz1_data = pdb_data[pdb_data['MODEL_TYPE'] == 'Boltz1']
            
            if len(boltz1_data) > 0:
                # Plot Boltz1 CCD
                if ccd_col in boltz1_data.columns:
                    if not boltz1_data[ccd_col].isna().all():
                        ccd_value = boltz1_data[ccd_col].values[0]
                        ax.bar(
                            y_positions[i] + bar_positions['Boltz1_CCD'],
                            ccd_value,
                            width=bar_width,
                            color=PlotConfig.BOLTZ1_CCD_COLOR,
                            edgecolor='black',
                            linewidth=0.5,
                            orientation='vertical'
                        )
                        
                        # Check threshold based on metric type
                        if metric_type == 'RMSD' and ccd_value > threshold_value:
                            exceed_threshold[i] = True
                        elif metric_type == 'DOCKQ' and ccd_value > threshold_value:
                            exceed_threshold[i] = True
                
                # Plot Boltz1 SMILES
                if smiles_col in boltz1_data.columns:
                    if not boltz1_data[smiles_col].isna().all():
                        smiles_value = boltz1_data[smiles_col].values[0]
                        ax.bar(
                            y_positions[i] + bar_positions['Boltz1_SMILES'],
                            smiles_value,
                            width=bar_width,
                            color=PlotConfig.BOLTZ1_SMILES_COLOR,
                            edgecolor='black', 
                            linewidth=0.5,
                            orientation='vertical'
                        )
                        
                        # Check threshold based on metric type
                        if metric_type == 'RMSD' and smiles_value > threshold_value:
                            exceed_threshold[i] = True
                        elif metric_type == 'DOCKQ' and smiles_value > threshold_value:
                            exceed_threshold[i] = True
        
        # Add threshold if requested
        if add_threshold and threshold_value is not None:
            threshold_line = ax.axhline(
                y=threshold_value,
                color='black',
                linestyle='--',
                alpha=0.7,
                linewidth=1.0,
                label='Threshold'
            )
        
        # Set axis labels and titles
        ax.set_ylabel(axis_label)
        ax.set_xlabel('PDB Identifier')
        
        # Set x-ticks
        ax.set_xticks(y_positions)
        ax.set_xticklabels(pdb_labels, rotation=90, ha='center')
        
        # Apply bolding to labels after they've been created
        for i, label in enumerate(ax.get_xticklabels()):
            if exceed_threshold[i]:
                label.set_fontweight('bold')
        
        # Set constant y-tick intervals
        if metric_type == 'RMSD':
            ax.yaxis.set_major_locator(MultipleLocator(5))  # RMSD values
            # Set fixed y-axis limits for RMSD
            if fixed_ylim:
                ax.set_ylim(0, 20)  # 0-20 Å range for RMSD
            else:
                # Use dynamic limits
                ax.set_ylim(0)
        else:  # DOCKQ
            ax.yaxis.set_major_locator(MultipleLocator(0.05))  # DockQ scores typically 0-1
            # Set y-axis limits for DOCKQ
            ax.set_ylim(0, 1)  # 0-1 range for DOCKQ
        
        # Set x-axis limits
        ax.set_xlim(-0.5 - 2*bar_width, len(sorted_pdb_ids) - 0.5 + 2*bar_width)
        
        # Add legend
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=PlotConfig.AF3_CCD_COLOR, edgecolor='black', linewidth=0.5, label='AF3 CCD'),
            plt.Rectangle((0, 0), 1, 1, facecolor=PlotConfig.AF3_SMILES_COLOR, edgecolor='black', linewidth=0.5, label='AF3 SMILES'),
            plt.Rectangle((0, 0), 1, 1, facecolor=PlotConfig.BOLTZ1_CCD_COLOR, edgecolor='black', linewidth=0.5, label='Boltz1 CCD'),
            plt.Rectangle((0, 0), 1, 1, facecolor=PlotConfig.BOLTZ1_SMILES_COLOR, edgecolor='black', linewidth=0.5, label='Boltz1 SMILES')
        ]
        
        if add_threshold:
            from matplotlib.lines import Line2D
            threshold_handle = Line2D([0], [0], color='black', linestyle='--', alpha=0.7, linewidth=1.0, label='Threshold')
            legend_handles.append(threshold_handle)
        
        ax.legend(
            handles=legend_handles,
            loc='upper right',
            framealpha=0,
            edgecolor='none'
        )
        
        # Add plot title without page numbers
        if metric_type == 'RMSD':
            title = f"AlphaFold3 vs Boltz1 RMSD"
        else:  # DOCKQ
            title = f"AlphaFold3 vs Boltz1 DockQ Score"
        
        fig.suptitle(title, fontsize=PlotConfig.TITLE_SIZE)
        
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
        
        return [fig], [ax] 