import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.lines import Line2D
from base_plotter import BasePlotter
from config import PlotConfig
from data_loader import DataLoader
from utils import save_figure

class TrainingCutoffPlotter(BasePlotter):
    """
    Class for creating plots comparing model performance before and after
    the AlphaFold3/Boltz-1 training cutoff date (2021-09-30).
    """
    
    def __init__(self):
        """Initialize the training cutoff plotter."""
        super().__init__()
        # Define the training cutoff date
        self.training_cutoff = pd.to_datetime('2021-09-30')
    
    def plot_training_cutoff_comparison(
        self, 
        df, 
        metric_type='RMSD', 
        model_type='AlphaFold3',
        add_threshold=True,
        threshold_value=None,
        width=10,
        height=8,
        save=True,
        molecule_type="PROTAC"
    ):
        """
        Create a bar plot showing the mean metric values across structures
        from before and after the training cutoff date.
        
        Args:
            df (pd.DataFrame): Data frame containing results
            metric_type (str): Metric to plot ('RMSD', 'DOCKQ', or 'LRMSD')
            model_type (str): Model type to analyze ('AlphaFold3' or 'Boltz1')
            add_threshold (bool): Whether to add a threshold line
            threshold_value (float): Value for the threshold line, if None will use default based on metric_type
            width (int): Figure width
            height (int): Figure height
            save (bool): Whether to save the plot
            molecule_type (str): Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
        
        Returns:
            fig, ax: The created figure and axis
        """
        try:
            # Set default threshold values based on metric type if not provided
            if threshold_value is None:
                if metric_type.upper() == 'RMSD':
                    threshold_value = 4.0
                elif metric_type.upper() == 'DOCKQ':
                    threshold_value = 0.5
                elif metric_type.upper() == 'LRMSD':
                    threshold_value = 4.0
                elif metric_type.upper() == 'PTM':
                    # No threshold needed for PTM
                    add_threshold = False
                else:
                    threshold_value = 4.0
            
            # Filter for the specific model type
            df_filtered = df[df['MODEL_TYPE'] == model_type].copy()
            
            if df_filtered.empty:
                print(f"Error: No data available for model type '{model_type}'")
                return None, None
            
            # Add a column to indicate if the structure is pre-training or post-training
            df_filtered['RELEASE_DATE'] = pd.to_datetime(df_filtered['RELEASE_DATE'])
            df_filtered['IS_POST_TRAINING'] = df_filtered['RELEASE_DATE'] > self.training_cutoff
            
            # Get the appropriate metric columns based on metric_type
            metric_columns = self._get_metric_columns(metric_type)
            if not metric_columns:
                print(f"Error: Metric type '{metric_type}' not supported.")
                return None, None
                
            smiles_col, ccd_col, y_label = metric_columns
            
            # Verify metric columns exist
            if smiles_col not in df_filtered.columns or ccd_col not in df_filtered.columns:
                print(f"Error: Required metric columns ({smiles_col}, {ccd_col}) not found in dataframe.")
                return None, None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(width, height))
            
            # Calculate metrics for pre and post training data
            metrics = self._calculate_period_metrics(df_filtered, metric_columns)
            
            # Unpack the data
            means = metrics['means']
            errors = metrics['errors']
            counts = metrics['counts']
            
            # Set up bar positions and width
            bar_positions = [0, 0.9, 2.4, 3.3]
            bar_width = 0.6
            
            # Define colors for each bar
            colors = [
                PlotConfig.CCD_PRIMARY,       # Pre-2021 CCD (orange)
                PlotConfig.SMILES_PRIMARY,    # Pre-2021 SMILES (blue)
                PlotConfig.CCD_PRIMARY,       # Post-2021 CCD (orange with pattern)
                PlotConfig.SMILES_PRIMARY     # Post-2021 SMILES (blue with pattern)
            ]
            
            # Define hatches for post-training bars
            hatches = ['', '', '//', '//']
            
            # Define denser hatches for the legend
            legend_hatches = ['', '', '///', '///']
            
            # Define bar labels
            bar_labels = [
                "Pre-2021 CCD",
                "Pre-2021 SMILES",
                "Post-2021 CCD",
                "Post-2021 SMILES"
            ]
            
            # Get the mean values in the correct order
            values = [
                means.get("Pre_CCD", 0),
                means.get("Pre_SMILES", 0),
                means.get("Post_CCD", 0),
                means.get("Post_SMILES", 0)
            ]
            
            # Get the error values in the correct order
            error_values = [
                errors.get("Pre_CCD", 0),
                errors.get("Pre_SMILES", 0),
                errors.get("Post_CCD", 0),
                errors.get("Post_SMILES", 0)
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
                error_kw={'ecolor': 'black', 'capsize': 4, 'capthick': 1, 'alpha': 0.7},
                hatch=hatches
            )
            
            # Add threshold line if requested
            if add_threshold and metric_type.upper() != 'PTM':
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
                # Use a dynamic offset based on metric type
                if metric_type == 'DOCKQ':
                    offset = 0.005
                elif metric_type.upper() == 'PTM':
                    offset = 0.01  # Smaller offset for PTM plots
                else:
                    offset = 0.03
                
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + error_values[i] + offset,  # Position just above error bar
                    f"{value:.2f}",
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    fontweight='bold',
                    color='black'
                )
            
            # Create custom legend patches
            legend_handles = []
            
            for i, (label, color, hatch) in enumerate(zip(bar_labels, colors, legend_hatches)):
                patch = plt.Rectangle(
                    (0, 0), 1, 1, 
                    facecolor=color, 
                    edgecolor='black', 
                    hatch=hatch,
                    linewidth=0.5, 
                    label=label
                )
                legend_handles.append(patch)
            
            # Add threshold line to legend if requested
            if add_threshold and metric_type.upper() != 'PTM':
                threshold_line = Line2D(
                    [0, 1], [0, 0], 
                    color='gray', 
                    linestyle='--',
                    linewidth=1.0, 
                    label='Threshold'
                )
                legend_handles.append(threshold_line)
            
            # Add legend
            if add_threshold and threshold_value is not None and metric_type.upper() != 'PTM' and max(values) < threshold_value * 1.5:
                # Use background when data values are near or below threshold
                ax.legend(
                    handles=legend_handles, 
                    loc='upper center',
                    frameon=True,
                    facecolor='white',
                    framealpha=0.8,
                    edgecolor='lightgray',
                    borderpad=0.8,
                    fontsize=12
                )
            else:
                ax.legend(
                    handles=legend_handles, 
                    loc='upper center',
                    frameon=False,
                    borderpad=0.8,
                    fontsize=12
                )
            
            # Remove x-ticks and labels since we have the legend
            ax.set_xticks([])
            ax.set_xticklabels([])
            
            # Add grid lines for x-axis only
            ax.grid(axis='y', linestyle='--', alpha=0.2)
            
            # Set axis labels
            ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
            
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
                ax.set_ylim(0, 45)
            elif metric_type.upper() == 'PTM':
                # PTM scores range from 0 to 1
                ax.set_ylim(0, 1.0)
            else:
                ax.set_ylim(0, ymax * 1.1)
            
            # Set tick label font sizes
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            # Use tight layout for better spacing
            plt.tight_layout()
            
            # Add a subtle border to the plot
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('gray')
            
            # Save the plot if requested
            if save:
                filename = f"{model_type.lower()}_training_cutoff_{molecule_type.lower()}_{metric_type.lower()}"
                save_figure(fig, filename)
            
            return fig, ax
        except Exception as e:
            print(f"Error in plot_training_cutoff_comparison: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
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
    
    def _calculate_period_metrics(self, df, metric_columns):
        """
        Calculate mean, standard error, and count statistics for pre and post training periods.
        
        Args:
            df: DataFrame containing filtered data with IS_POST_TRAINING column
            metric_columns: Tuple of (smiles_col, ccd_col, label)
            
        Returns:
            Dictionary containing means, errors, and counts for each period/metric combination
        """
        smiles_col, ccd_col, _ = metric_columns
        
        # Initialize dictionaries to store values
        means = {}
        errors = {}
        counts = {}
        
        # Get pre-training data
        pre_df = df[~df['IS_POST_TRAINING']]
        
        # Get post-training data
        post_df = df[df['IS_POST_TRAINING']]
        
        # Calculate pre-training metrics for CCD
        ccd_pre_values = pre_df[ccd_col].dropna()
        if len(ccd_pre_values) > 0:
            means["Pre_CCD"] = ccd_pre_values.mean()
            errors["Pre_CCD"] = ccd_pre_values.std() / np.sqrt(len(ccd_pre_values))  # Standard error
            counts["Pre_CCD"] = len(ccd_pre_values)
        else:
            means["Pre_CCD"] = 0
            errors["Pre_CCD"] = 0
            counts["Pre_CCD"] = 0
            
        # Calculate pre-training metrics for SMILES
        smiles_pre_values = pre_df[smiles_col].dropna()
        if len(smiles_pre_values) > 0:
            means["Pre_SMILES"] = smiles_pre_values.mean()
            errors["Pre_SMILES"] = smiles_pre_values.std() / np.sqrt(len(smiles_pre_values))
            counts["Pre_SMILES"] = len(smiles_pre_values)
        else:
            means["Pre_SMILES"] = 0
            errors["Pre_SMILES"] = 0
            counts["Pre_SMILES"] = 0
        
        # Calculate post-training metrics for CCD
        ccd_post_values = post_df[ccd_col].dropna()
        if len(ccd_post_values) > 0:
            means["Post_CCD"] = ccd_post_values.mean()
            errors["Post_CCD"] = ccd_post_values.std() / np.sqrt(len(ccd_post_values))
            counts["Post_CCD"] = len(ccd_post_values)
        else:
            means["Post_CCD"] = 0
            errors["Post_CCD"] = 0
            counts["Post_CCD"] = 0
            
        # Calculate post-training metrics for SMILES
        smiles_post_values = post_df[smiles_col].dropna()
        if len(smiles_post_values) > 0:
            means["Post_SMILES"] = smiles_post_values.mean()
            errors["Post_SMILES"] = smiles_post_values.std() / np.sqrt(len(smiles_post_values))
            counts["Post_SMILES"] = len(smiles_post_values)
        else:
            means["Post_SMILES"] = 0
            errors["Post_SMILES"] = 0
            counts["Post_SMILES"] = 0
        
        # Calculate improvement percentages
        metrics = {
            'means': means,
            'errors': errors,
            'counts': counts,
            'improvements': {}
        }
        
        # Calculate percentage differences between pre and post training
        if "Pre_CCD" in means and "Post_CCD" in means and means["Pre_CCD"] > 0:
            metrics['improvements']['CCD'] = (means["Post_CCD"] - means["Pre_CCD"]) / means["Pre_CCD"] * 100
            
        if "Pre_SMILES" in means and "Post_SMILES" in means and means["Pre_SMILES"] > 0:
            metrics['improvements']['SMILES'] = (means["Post_SMILES"] - means["Pre_SMILES"]) / means["Pre_SMILES"] * 100
        
        return metrics 