import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.lines import Line2D
from base_plotter import BasePlotter
from config import PlotConfig
from data_loader import DataLoader
from utils import save_figure
import logging

class TrainingCutoffPlotter(BasePlotter):
    """
    Class for creating plots comparing model performance before and after
    the AlphaFold3/Boltz-1 training cutoff date (2021-09-30).
    """
    
    # --- Plot Configuration Constants ---
    # Plot dimensions
    PLOT_WIDTH = 4
    PLOT_HEIGHT = 4

    # Font sizes (can be fine-tuned)
    TITLE_FONT_SIZE = 16 # Example, adjust as needed
    AXIS_LABEL_FONT_SIZE = 12
    VALUE_LABEL_FONT_SIZE = 12
    LEGEND_FONT_SIZE = 9.5
    TICK_LABEL_FONT_SIZE = 11

    # Bar appearance
    BAR_WIDTH = 0.05
    BAR_EDGE_COLOR = 'black'
    BAR_EDGE_LINE_WIDTH = 0.5
    BAR_SPACING_FACTOR = 2 

    # Hatches for post-training bars (empty for pre-training)
    PRE_TRAINING_HATCH = ''
    POST_TRAINING_HATCH = '//' 
    POST_TRANING_HATCH_LINE_WIDTH = 0.5
    # Denser hatches for legend representation
    LEGEND_POST_TRAINING_HATCH = '///'

    # Bar Colors (can use PlotConfig or define specific ones here)
    # For simplicity, using PlotConfig if available, otherwise define fallbacks
    PRE_CCD_COLOR_AF3 = getattr(PlotConfig, 'CCD_PRIMARY', '#FF7F50')      # Coral-like for AF3
    PRE_SMILES_COLOR_AF3 = getattr(PlotConfig, 'SMILES_PRIMARY', '#1F77B4') # Blue-like for AF3
    POST_CCD_COLOR_AF3 = PRE_CCD_COLOR_AF3 # Same color, differentiated by hatch
    POST_SMILES_COLOR_AF3 = PRE_SMILES_COLOR_AF3 # Same color, differentiated by hatch

    # Boltz1 specific colors
    PRE_CCD_COLOR_BOLTZ1 = '#A157DB'    # Purple-like for Boltz1 CCD
    PRE_SMILES_COLOR_BOLTZ1 = '#57DB80' # Green-like for Boltz1 SMILES
    POST_CCD_COLOR_BOLTZ1 = PRE_CCD_COLOR_BOLTZ1 # Same color, differentiated by hatch
    POST_SMILES_COLOR_BOLTZ1 = PRE_SMILES_COLOR_BOLTZ1 # Same color, differentiated by hatch

    # Error bar appearance
    ERROR_BAR_COLOR = 'black'
    ERROR_BAR_CAPSIZE = 4
    ERROR_BAR_THICKNESS = 1
    ERROR_BAR_ALPHA = 0.7

    # Grid properties
    GRID_LINESTYLE = '--'
    GRID_ALPHA = 0.2

    # Threshold line properties
    THRESHOLD_LINE_COLOR = 'gray' # For the line itself
    THRESHOLD_LEGEND_COLOR = 'gray' # For the legend entry
    THRESHOLD_LINE_STYLE = '--'
    THRESHOLD_LINE_ALPHA = 1
    THRESHOLD_LINE_WIDTH = 1.0

    # Default threshold values for metrics
    DEFAULT_RMSD_THRESHOLD = 4.0
    DEFAULT_DOCKQ_THRESHOLD = 0.5
    DEFAULT_LRMSD_THRESHOLD = 4.0
    # PTM usually doesn't have a threshold line in these plots

    # Legend properties
    LEGEND_LOCATION = 'best' # Default to best, PTM will override to upper right
    LEGEND_BORDER_PADDING = 0.8
    # --- End Plot Configuration Constants ---

    TRAINING_CUTOFF_DATE = pd.to_datetime('2021-09-30')
    
    def __init__(self, debug=False):
        """Initialize the training cutoff plotter."""
        super().__init__()
        # Define the training cutoff date using the class constant
        self.training_cutoff = self.TRAINING_CUTOFF_DATE
        self.debug = debug
    
    def _debug_print(self, message):
        """Print debug information if debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] TrainingCutoffPlotter: {message}")
    
    def plot_training_cutoff_comparison(
        self, 
        df, 
        metric_type='RMSD', 
        model_type='AlphaFold3',
        add_threshold=True,
        threshold_value=None,
        width=None,
        height=None,
        save=True,
        molecule_type="PROTAC",
        debug=False,
        fixed_ylim=None
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
            width (int): Figure width (overrides default PLOT_WIDTH if provided)
            height (int): Figure height (overrides default PLOT_HEIGHT if provided)
            save (bool): Whether to save the plot
            molecule_type (str): Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
            debug (bool): Enable additional debugging output
            fixed_ylim (tuple, optional): A tuple (min, max) to set a fixed y-axis limit.
        
        Returns:
            tuple: (fig, ax, used_ylim) The figure, axis, and the y-limits used.
        """
        self.debug = debug or self.debug
        
        # Use class constants for width and height if not provided in args
        plot_width = width if width is not None else self.PLOT_WIDTH
        plot_height = height if height is not None else self.PLOT_HEIGHT

        # Store original hatch linewidth and set desired one
        original_hatch_linewidth = plt.rcParams.get('hatch.linewidth')
        plt.rcParams['hatch.linewidth'] = self.POST_TRANING_HATCH_LINE_WIDTH
        
        try:
            self._debug_print(f"Starting plot_training_cutoff_comparison with metric_type={metric_type}, model_type={model_type}")
            self._debug_print(f"Plot dimensions: width={plot_width}, height={plot_height}")
            self._debug_print(f"Input dataframe shape: {df.shape}")
            
            # Verify that the dataframe is not empty
            if df.empty:
                print(f"Error: Input dataframe is empty")
                return None, None, None
                
            # Verify required columns
            required_columns = ['MODEL_TYPE', 'RELEASE_DATE']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Error: Required columns missing from dataframe: {missing_columns}")
                self._debug_print(f"Available columns: {df.columns.tolist()}")
                return None, None, None
                
            # Verify values in MODEL_TYPE
            available_models = df['MODEL_TYPE'].unique()
            if model_type not in available_models:
                print(f"Error: Model type '{model_type}' not found in data. Available models: {available_models}")
                return None, None, None
            
            # Set default threshold values based on metric type if not provided
            if threshold_value is None:
                if metric_type.upper() == 'RMSD':
                    threshold_value = self.DEFAULT_RMSD_THRESHOLD
                elif metric_type.upper() == 'DOCKQ':
                    threshold_value = self.DEFAULT_DOCKQ_THRESHOLD
                elif metric_type.upper() == 'LRMSD':
                    threshold_value = self.DEFAULT_LRMSD_THRESHOLD
                elif metric_type.upper() == 'PTM':
                    # No threshold needed for PTM
                    add_threshold = False
                else:
                    threshold_value = self.DEFAULT_RMSD_THRESHOLD
                self._debug_print(f"Using default threshold value: {threshold_value} for metric {metric_type}")
            
            # Filter for the specific model type
            df_filtered = df[df['MODEL_TYPE'] == model_type].copy()
            self._debug_print(f"After model_type filter, dataframe shape: {df_filtered.shape}")
            
            if df_filtered.empty:
                print(f"Error: No data available for model type '{model_type}'")
                return None, None, None
                
            # Filter by molecule type
            # Check if we have MOLECULE_TYPE column (newer datasets) or TYPE column (older datasets)
            molecule_type_col = 'MOLECULE_TYPE' if 'MOLECULE_TYPE' in df_filtered.columns else 'TYPE'
            
            if molecule_type_col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[molecule_type_col] == molecule_type].copy()
                self._debug_print(f"After molecule_type filter ({molecule_type}), dataframe shape: {df_filtered.shape}")
                
                if df_filtered.empty:
                    print(f"Error: No data available for molecule type '{molecule_type}'")
                    available_types = df[df['MODEL_TYPE'] == model_type][molecule_type_col].unique()
                    print(f"Available molecule types: {available_types}")
                    return None, None, None
            else:
                print(f"Warning: No '{molecule_type_col}' column found in data, skipping molecule type filtering")
                self._debug_print(f"Available columns: {df_filtered.columns.tolist()}")
            
            # Make sure RELEASE_DATE column is in proper datetime format
            try:
                df_filtered['RELEASE_DATE'] = pd.to_datetime(df_filtered['RELEASE_DATE'])
            except Exception as e:
                print(f"Error: Could not convert RELEASE_DATE column to datetime format: {e}")
                self._debug_print(f"RELEASE_DATE values sample: {df_filtered['RELEASE_DATE'].head()}")
                return None, None, None
                
            # Add a column to indicate if the structure is pre-training or post-training
            df_filtered['IS_POST_TRAINING'] = df_filtered['RELEASE_DATE'] > self.training_cutoff
            
            # Debug counts of pre and post training data
            pre_count = sum(~df_filtered['IS_POST_TRAINING'])
            post_count = sum(df_filtered['IS_POST_TRAINING'])
            self._debug_print(f"Pre-training samples: {pre_count}, Post-training samples: {post_count}")
            
            # Verify we have both pre and post training data
            if pre_count == 0:
                print(f"Warning: No pre-training data available (before {self.training_cutoff})")
            if post_count == 0:
                print(f"Warning: No post-training data available (after {self.training_cutoff})")
            
            # Get the appropriate metric columns based on metric_type
            metric_columns = self._get_metric_columns(metric_type)
            if not metric_columns:
                print(f"Error: Metric type '{metric_type}' not supported.")
                return None, None, None
                
            smiles_col, ccd_col, y_label = metric_columns
            self._debug_print(f"Using metric columns: SMILES={smiles_col}, CCD={ccd_col}")
            
            # Verify metric columns exist
            if smiles_col not in df_filtered.columns or ccd_col not in df_filtered.columns:
                print(f"Error: Required metric columns ({smiles_col}, {ccd_col}) not found in dataframe.")
                self._debug_print(f"Available columns: {df_filtered.columns.tolist()}")
                return None, None, None
            
            # Debug: Check for missing data in metric columns
            smiles_missing = df_filtered[smiles_col].isna().sum()
            ccd_missing = df_filtered[ccd_col].isna().sum()
            self._debug_print(f"Missing values: {smiles_col}={smiles_missing}, {ccd_col}={ccd_missing}")
            
            # Check if we have enough data for a meaningful plot
            valid_data_count = df_filtered[smiles_col].notna().sum() + df_filtered[ccd_col].notna().sum()
            if valid_data_count == 0:
                print(f"Error: No valid data available for metric columns {smiles_col} and {ccd_col}")
                return None, None, None
            elif valid_data_count < 4:
                print(f"Warning: Very limited data available for plotting ({valid_data_count} non-null values)")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            
            # Calculate metrics for pre and post training data
            metrics = self._calculate_period_metrics(df_filtered, metric_columns)
            
            # Debug: Print calculated metrics
            self._debug_print("Calculated metrics:")
            for category, value in metrics['means'].items():
                self._debug_print(f"  Mean {category}: {value:.4f}")
            for category, value in metrics['errors'].items():
                self._debug_print(f"  Error {category}: {value:.4f}")
            for category, value in metrics['counts'].items():
                self._debug_print(f"  Count {category}: {value}")
            
            # Unpack the data
            means = metrics['means']
            errors = metrics['errors']
            counts = metrics['counts']
            
            # Set up bar positions and width
            bar_width = self.BAR_WIDTH
            spacing_factor = self.BAR_SPACING_FACTOR
            bar_positions = [0, bar_width*spacing_factor, bar_width*2*spacing_factor, bar_width*3*spacing_factor]
            
            # Define colors for each bar based on model_type
            if model_type == 'AlphaFold3':
                colors = [
                    self.PRE_CCD_COLOR_AF3,
                    self.PRE_SMILES_COLOR_AF3,
                    self.POST_CCD_COLOR_AF3,
                    self.POST_SMILES_COLOR_AF3
                ]
            elif model_type == 'Boltz1' or model_type == 'Boltz-1': # Handling potential variations in name
                colors = [
                    self.PRE_CCD_COLOR_BOLTZ1,
                    self.PRE_SMILES_COLOR_BOLTZ1,
                    self.POST_CCD_COLOR_BOLTZ1,
                    self.POST_SMILES_COLOR_BOLTZ1
                ]
            else:
                # Fallback to AF3 colors if model_type is unexpected, with a warning
                self._debug_print(f"Warning: Model type '{model_type}' not recognized for specific colors. Defaulting to AlphaFold3 colors.")
                colors = [
                    self.PRE_CCD_COLOR_AF3,
                    self.PRE_SMILES_COLOR_AF3,
                    self.POST_CCD_COLOR_AF3,
                    self.POST_SMILES_COLOR_AF3
                ]
            
            # Define hatches for post-training bars
            hatches = [self.PRE_TRAINING_HATCH, self.PRE_TRAINING_HATCH, self.POST_TRAINING_HATCH, self.POST_TRAINING_HATCH]
            
            # Define denser hatches for the legend
            legend_hatches = [self.PRE_TRAINING_HATCH, self.PRE_TRAINING_HATCH, self.LEGEND_POST_TRAINING_HATCH, self.LEGEND_POST_TRAINING_HATCH]
            
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
            
            self._debug_print(f"Plotting values: {values}")
            self._debug_print(f"Error values: {error_values}")
            
            # Plot the bars
            bars = ax.bar(
                bar_positions,
                values,
                bar_width,
                color=colors,
                edgecolor=self.BAR_EDGE_COLOR,
                linewidth=self.BAR_EDGE_LINE_WIDTH,
                yerr=error_values,
                error_kw={'ecolor': self.ERROR_BAR_COLOR, 'capsize': self.ERROR_BAR_CAPSIZE, 'capthick': self.ERROR_BAR_THICKNESS, 'alpha': self.ERROR_BAR_ALPHA},
                hatch=hatches
            )
            
            # Add threshold line if requested
            if add_threshold and metric_type.upper() != 'PTM':
                threshold_line = ax.axhline(
                    y=threshold_value,
                    color=self.THRESHOLD_LINE_COLOR,
                    linestyle=self.THRESHOLD_LINE_STYLE,
                    alpha=self.THRESHOLD_LINE_ALPHA,
                    linewidth=self.THRESHOLD_LINE_WIDTH
                )
                self._debug_print(f"Added threshold line at y={threshold_value}")
            
            # Create custom legend patches
            legend_handles = []
            
            for i, (label, color, hatch) in enumerate(zip(bar_labels, colors, legend_hatches)):
                patch = plt.Rectangle(
                    (0, 0), 1, 1, 
                    facecolor=color, 
                    edgecolor=self.BAR_EDGE_COLOR, 
                    hatch=hatch,
                    linewidth=self.BAR_EDGE_LINE_WIDTH, 
                    label=label
                )
                legend_handles.append(patch)
            
            # Add threshold line to legend if requested
            if add_threshold and metric_type.upper() != 'PTM':
                threshold_line = Line2D(
                    [0, 1], [0, 0], 
                    color=self.THRESHOLD_LEGEND_COLOR, 
                    linestyle=self.THRESHOLD_LINE_STYLE,
                    linewidth=self.THRESHOLD_LINE_WIDTH, 
                    label='Threshold'
                )
                legend_handles.append(threshold_line)
            
            # Add legend
            current_legend_loc = self.LEGEND_LOCATION
            if metric_type.upper() == 'PTM':
                current_legend_loc = 'upper right'
            
            if add_threshold and threshold_value is not None and metric_type.upper() != 'PTM' and max(values) < threshold_value * 1.5:
                # Use background when data values are near or below threshold
                ax.legend(
                    handles=legend_handles, 
                    loc=current_legend_loc, 
                    frameon=False, # Ensure no border
                    facecolor='white', # Kept for consistency, but won't show with frameon=False
                    framealpha=0.8, # Kept for consistency, but won't show with frameon=False
                    edgecolor='lightgray', # Kept for consistency, but won't show with frameon=False
                    borderpad=self.LEGEND_BORDER_PADDING,
                    fontsize=self.LEGEND_FONT_SIZE,
                    labelspacing=0.2
                )
            else:
                ax.legend(
                    handles=legend_handles, 
                    loc=current_legend_loc, 
                    frameon=False, # Ensure no border
                    borderpad=self.LEGEND_BORDER_PADDING,
                    fontsize=self.LEGEND_FONT_SIZE,
                    labelspacing=0.2
                )
            
            # Remove x-ticks and labels since we have the legend
            ax.set_xticks([])
            ax.set_xticklabels([])
            
            # Add grid lines for x-axis only
            ax.grid(axis='y', linestyle=self.GRID_LINESTYLE, alpha=self.GRID_ALPHA)
            
            # Set axis labels
            ax.set_ylabel(y_label, fontsize=self.AXIS_LABEL_FONT_SIZE, fontweight='bold')
            
            # Adjust y-axis to accommodate value labels
            ymax = max([v + e * 1.5 for v, e in zip(values, error_values)])
            self._debug_print(f"Calculated ymax for data: {ymax}")
            
            current_ylim = None
            if fixed_ylim:
                ax.set_ylim(fixed_ylim)
                current_ylim = fixed_ylim
                self._debug_print(f"Using fixed Y-axis limits: {fixed_ylim}")
            else:
                # Set y-axis limit appropriately based on metric type
                if metric_type.upper() == 'RMSD' and threshold_value >= self.DEFAULT_RMSD_THRESHOLD:
                    current_ylim = (0, max(self.DEFAULT_RMSD_THRESHOLD + 0.5, ymax * 1.1))
                elif metric_type.upper() == 'DOCKQ':
                    # DockQ scores range from 0 to 1
                    current_ylim = (0, min(1.05, max(self.DEFAULT_DOCKQ_THRESHOLD + 0.1, ymax * 1.1)))
                elif metric_type.upper() == 'LRMSD':
                    # Use a larger y-axis limit for LRMSD to prevent label overlapping
                    current_ylim = (0, 45) # Max typical value, or adjust based on ymax
                elif metric_type.upper() == 'PTM':
                    # PTM scores range from 0 to 1
                    current_ylim = (0, 1.0)
                else:
                    current_ylim = (0, ymax * 1.1)
                ax.set_ylim(current_ylim)
            
            self._debug_print(f"Y-axis limits set to: {ax.get_ylim()}")
            used_ylim = ax.get_ylim() # Get the actual limits set on the axis
            
            # Set tick label font sizes
            ax.tick_params(axis='both', which='major', labelsize=self.TICK_LABEL_FONT_SIZE)
            
            # Use tight layout for better spacing
            plt.tight_layout()
            
            # Add a subtle border to the plot
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('black')
            
            # Add a title with debug info if in debug mode
            if self.debug:
                # Instead of adding titles, just log the sample counts
                self._debug_print(f"Sample counts - Pre: {pre_count}, Post: {post_count}")
                # No titles will be added, allowing the user to add their own
            
            # Save the plot if requested
            if save:
                filename = f"{model_type.lower()}_training_cutoff_{molecule_type.lower()}_{metric_type.lower()}"
                if self.debug:
                    filename += "_debug"
                save_figure(fig, filename)
                self._debug_print(f"Saved figure as {filename}")
            
            self._debug_print("Successfully completed plot_training_cutoff_comparison")
            return fig, ax, used_ylim # Return the used y-limits
            
        except Exception as e:
            print(f"Error in plot_training_cutoff_comparison: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None # Return None for ylim on error
        finally:
            # Restore original hatch linewidth
            if original_hatch_linewidth is not None:
                plt.rcParams['hatch.linewidth'] = original_hatch_linewidth
            else:
                # If it wasn't set before, perhaps remove it or set to a Matplotlib default
                # For simplicity, if it wasn't there, we might not need to do anything, 
                # or we could plt.rcParams.pop('hatch.linewidth', None) if we want to be very clean.
                # Alternatively, find matplotlib's default and set to that if it was None.
                # For now, just setting it back if it existed.
                pass # Or print a warning if original_hatch_linewidth was None and it matters.
    
    def _get_metric_columns(self, metric_type):
        """Get the column names for a specific metric type."""
        if metric_type.upper() == 'RMSD':
            return ('SMILES_RMSD', 'CCD_RMSD', 'RMSD (Å)')
        elif metric_type.upper() == 'DOCKQ':
            return ('SMILES_DOCKQ_SCORE', 'CCD_DOCKQ_SCORE', 'DockQ Score')
        elif metric_type.upper() == 'LRMSD':
            return ('SMILES_DOCKQ_LRMSD', 'CCD_DOCKQ_LRMSD', 'LRMSD (Å)')
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
        
        if self.debug:
            self._debug_print(f"Pre-training data shape: {pre_df.shape}")
            self._debug_print(f"Post-training data shape: {post_df.shape}")
            self._debug_print(f"Pre CCD non-NA values: {pre_df[ccd_col].notna().sum()}")
            self._debug_print(f"Pre SMILES non-NA values: {pre_df[smiles_col].notna().sum()}")
            self._debug_print(f"Post CCD non-NA values: {post_df[ccd_col].notna().sum()}")
            self._debug_print(f"Post SMILES non-NA values: {post_df[smiles_col].notna().sum()}")
        
        # Calculate pre-training metrics for CCD
        ccd_pre_values = pre_df[ccd_col].dropna()
        if len(ccd_pre_values) > 0:
            means["Pre_CCD"] = ccd_pre_values.mean()
            errors["Pre_CCD"] = ccd_pre_values.std() / np.sqrt(len(ccd_pre_values))  # Standard error
            counts["Pre_CCD"] = len(ccd_pre_values)
            if self.debug:
                self._debug_print(f"Pre CCD values: {ccd_pre_values.tolist()}")
                self._debug_print(f"Pre CCD mean: {means['Pre_CCD']}, std: {ccd_pre_values.std()}")
        else:
            means["Pre_CCD"] = 0
            errors["Pre_CCD"] = 0
            counts["Pre_CCD"] = 0
            if self.debug:
                self._debug_print("No Pre CCD values found")
            
        # Calculate pre-training metrics for SMILES
        smiles_pre_values = pre_df[smiles_col].dropna()
        if len(smiles_pre_values) > 0:
            means["Pre_SMILES"] = smiles_pre_values.mean()
            errors["Pre_SMILES"] = smiles_pre_values.std() / np.sqrt(len(smiles_pre_values))
            counts["Pre_SMILES"] = len(smiles_pre_values)
            if self.debug:
                self._debug_print(f"Pre SMILES values: {smiles_pre_values.tolist()}")
                self._debug_print(f"Pre SMILES mean: {means['Pre_SMILES']}, std: {smiles_pre_values.std()}")
        else:
            means["Pre_SMILES"] = 0
            errors["Pre_SMILES"] = 0
            counts["Pre_SMILES"] = 0
            if self.debug:
                self._debug_print("No Pre SMILES values found")
        
        # Calculate post-training metrics for CCD
        ccd_post_values = post_df[ccd_col].dropna()
        if len(ccd_post_values) > 0:
            means["Post_CCD"] = ccd_post_values.mean()
            errors["Post_CCD"] = ccd_post_values.std() / np.sqrt(len(ccd_post_values))
            counts["Post_CCD"] = len(ccd_post_values)
            if self.debug:
                self._debug_print(f"Post CCD values: {ccd_post_values.tolist()}")
                self._debug_print(f"Post CCD mean: {means['Post_CCD']}, std: {ccd_post_values.std()}")
        else:
            means["Post_CCD"] = 0
            errors["Post_CCD"] = 0
            counts["Post_CCD"] = 0
            if self.debug:
                self._debug_print("No Post CCD values found")
            
        # Calculate post-training metrics for SMILES
        smiles_post_values = post_df[smiles_col].dropna()
        if len(smiles_post_values) > 0:
            means["Post_SMILES"] = smiles_post_values.mean()
            errors["Post_SMILES"] = smiles_post_values.std() / np.sqrt(len(smiles_post_values))
            counts["Post_SMILES"] = len(smiles_post_values)
            if self.debug:
                self._debug_print(f"Post SMILES values: {smiles_post_values.tolist()}")
                self._debug_print(f"Post SMILES mean: {means['Post_SMILES']}, std: {smiles_post_values.std()}")
        else:
            means["Post_SMILES"] = 0
            errors["Post_SMILES"] = 0
            counts["Post_SMILES"] = 0
            if self.debug:
                self._debug_print("No Post SMILES values found")
        
        # Calculate improvement percentages
        metrics_summary = { # Renamed from metrics to avoid conflict
            'means': means,
            'errors': errors,
            'counts': counts,
            'improvements': {}
        }
        
        # Calculate percentage differences between pre and post training
        if "Pre_CCD" in means and "Post_CCD" in means and means["Pre_CCD"] > 0:
            metrics_summary['improvements']['CCD'] = (means["Post_CCD"] - means["Pre_CCD"]) / means["Pre_CCD"] * 100
            if self.debug:
                self._debug_print(f"CCD improvement: {metrics_summary['improvements']['CCD']:.2f}%")
            
        if "Pre_SMILES" in means and "Post_SMILES" in means and means["Pre_SMILES"] > 0:
            metrics_summary['improvements']['SMILES'] = (means["Post_SMILES"] - means["Pre_SMILES"]) / means["Pre_SMILES"] * 100
            if self.debug:
                self._debug_print(f"SMILES improvement: {metrics_summary['improvements']['SMILES']:.2f}%")
        
        return metrics_summary # Return the renamed summary dictionary 