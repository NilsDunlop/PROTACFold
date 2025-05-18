import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from base_plotter import BasePlotter
from config import PlotConfig
from data_loader import DataLoader
from utils import categorize_by_cutoffs, save_figure

# Constant for bar height across all plots
BAR_HEIGHT = 0.3
BAR_SPACING = 0.05

# Font size constants
TICK_LABEL_FONTSIZE = 10
AXIS_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 10
ANNOTATION_FONTSIZE = 9

class HorizontalBarPlotter(BasePlotter):
    """Class for creating horizontal bar plots comparing means with error bars."""
    
    def plot_bars(self, df_agg, molecule_type="PROTAC", classification_cutoff=None, 
                metrics=None, add_threshold=False, threshold_values=None,
                show_y_labels_on_all=False, width=10, height=None, 
                save=False, max_structures_per_plot=20,
                binary_plot_width: float | None = 4, # New parameter for binary plot width
                binary_title_fontsize: float | None = None, # New parameter for binary plot title font size
                smiles_color: str | None = None, # New parameter for SMILES bar color
                ccd_color: str | None = None # New parameter for CCD bar color
                ):
        """
        Create horizontal bar plots for different metrics, grouped by category.
        
        Args:
            df_agg: Aggregated DataFrame with mean and std values
            molecule_type: Type of molecule to filter by (e.g., "PROTAC")
            classification_cutoff: List of cutoff values for categories
            metrics: List of tuples (smiles_col, ccd_col, label) to plot. 
                     Default order for ternary: RMSD, DockQ, LRMSD.
            add_threshold: Whether to add threshold lines
            threshold_values: Values for threshold lines, corresponding to metrics order.
            show_y_labels_on_all: Whether to show y-labels on all subplots
            width: Figure width for ternary plots
            height: Figure height (if None, calculated based on number of structures)
            save: Whether to save the figure
            max_structures_per_plot: Maximum number of structures to show in a single plot
            binary_plot_width: Optional width for the binary plot. Defaults to general `width` if None.
            binary_title_fontsize: Optional font size for the binary plot title. Defaults to PlotConfig.TITLE_SIZE if None.
            smiles_color: Optional color for SMILES bars. Defaults to PlotConfig.SMILES_PRIMARY.
            ccd_color: Optional color for CCD bars. Defaults to PlotConfig.CCD_PRIMARY.
            
        Returns:
            Lists of created figures and axes
        """
        # Default metrics if none provided - RMSD, DockQ, LRMSD
        if metrics is None:
            metrics = [
                ('SMILES_RMSD', 'CCD_RMSD', 'RMSD (Å)'),                 # RMSD
                ('SMILES_DOCKQ_SCORE', 'CCD_DOCKQ_SCORE', 'DockQ Score'), # DockQ
                ('SMILES_DOCKQ_LRMSD', 'CCD_DOCKQ_LRMSD', 'LRMSD (Å)')  # LRMSD
            ]
            
        binary_rmsd_metric = ('SMILES_RMSD', 'CCD_RMSD', 'RMSD (Å)')
        
        # Default threshold values if needed but not provided, matching new metrics order
        if add_threshold and threshold_values is None:
            threshold_values = [4, 0.23, 4] # RMSD, DockQ, LRMSD
            
        # For binary plot, RMSD threshold is the first element of the reordered threshold_values
        binary_threshold_value = threshold_values[0] if add_threshold and threshold_values and len(threshold_values) > 0 else (4 if add_threshold else None)
        
        df_filtered = df_agg[df_agg['TYPE'] == molecule_type].copy()
        
        for col in df_filtered.columns:
            if col.endswith('_mean') or col.endswith('_std'):
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        
        if 'RELEASE_DATE' in df_filtered.columns:
            df_filtered['RELEASE_DATE'] = pd.to_datetime(df_filtered['RELEASE_DATE'])
            df_filtered = df_filtered.sort_values('RELEASE_DATE', ascending=True)
        
        df_filtered = DataLoader.identify_binary_structures(df_filtered)
        
        if classification_cutoff is None:
            ccd_rmsd_mean = df_filtered['CCD_RMSD_mean'].dropna()
            if len(ccd_rmsd_mean) > 0:
                classification_cutoff = [
                    np.percentile(ccd_rmsd_mean, 20),
                    np.percentile(ccd_rmsd_mean, 40),
                    np.percentile(ccd_rmsd_mean, 60),
                    np.percentile(ccd_rmsd_mean, 80)
                ]
            else:
                classification_cutoff = [2, 4, 6, 8]
        
        df_filtered = categorize_by_cutoffs(
            df_filtered, 'CCD_RMSD_mean', classification_cutoff, 'Category'
        )
        
        category_labels = [
            f"< {classification_cutoff[0]:.2f}",
            f"{classification_cutoff[0]:.2f} - {classification_cutoff[1]:.2f}",
            f"{classification_cutoff[1]:.2f} - {classification_cutoff[2]:.2f}",
            f"{classification_cutoff[2]:.2f} - {classification_cutoff[3]:.2f}",
            f"> {classification_cutoff[3]:.2f}"
        ]

        # Merge the first category into the second for ternary plots
        # This assumes category_labels always has at least 2 elements, which it should
        # given classification_cutoff always has 4 elements.
        label_to_merge_from = category_labels[0]
        label_to_merge_into = category_labels[1]
        df_filtered.loc[df_filtered['Category'] == label_to_merge_from, 'Category'] = label_to_merge_into
        
        # Adjust the list of labels to iterate over for plotting ternary categories
        plotting_category_labels_for_ternary = category_labels[1:]
        last_category_label_for_ternary = category_labels[-1] # Identify the label of the very last category
        
        binary_category_title = "Binary Structures"
        
        all_figures = []
        all_axes = []
        
        binary_df = df_filtered[df_filtered['is_binary']]
        if len(binary_df) > 0:
            current_binary_plot_width = binary_plot_width if binary_plot_width is not None else width
            current_binary_title_fontsize = binary_title_fontsize if binary_title_fontsize is not None else PlotConfig.TITLE_SIZE

            self._create_compact_binary_plots(
                binary_df, binary_category_title, 
                binary_rmsd_metric,
                add_threshold, binary_threshold_value, 
                current_binary_plot_width, 
                height, 
                save, 
                current_binary_title_fontsize, 
                all_figures, all_axes,
                smiles_color=smiles_color,
                ccd_color=ccd_color
            )
        
        for category_plot_label in plotting_category_labels_for_ternary: 
            category_df = df_filtered[(df_filtered['Category'] == category_plot_label) & (~df_filtered['is_binary'])]
            
            if len(category_df) == 0:
                continue

            if 'CCD_RMSD_mean' in category_df.columns:
                category_df = category_df.sort_values(by='CCD_RMSD_mean', ascending=True, na_position='first')
            else:
                print("Warning: CCD_RMSD_mean column not found for sorting ternary plots. Plots may not be sorted as requested.")
                
            self._create_paginated_plots(
                category_df, f"Ternary: {category_plot_label}", metrics, 
                add_threshold, threshold_values, 
                width, height, show_y_labels_on_all, save,
                max_structures_per_plot, all_figures, all_axes,
                last_category_label_for_ternary,
                smiles_color=smiles_color,
                ccd_color=ccd_color
            )
        
        return all_figures, all_axes
    
    def _create_compact_binary_plots(self, binary_df, category_title, 
                                    rmsd_metric, add_threshold, threshold_value, 
                                    width, height, save, title_fontsize,
                                    all_figures, all_axes,
                                    smiles_color: str | None = None,
                                    ccd_color: str | None = None
                                    ):
        """
        Create a single compact plot for all binary structures, sorted by RMSD.
        Higher RMSD values are at the top.
        
        Args:
            binary_df: DataFrame containing binary structures
            category_title: Title for the plot category (e.g., "Binary Structures")
            rmsd_metric: Tuple with (smiles_col, ccd_col, label) for RMSD
            add_threshold: Whether to add threshold lines
            threshold_value: Value for threshold line
            width: Figure width for this specific binary plot
            height: Figure height (if None, calculated based on number of structures)
            save: Whether to save the figure
            title_fontsize: Font size for the plot title.
            all_figures, all_axes: Lists to append the created figures and axes to
            smiles_color: Optional color for SMILES bars.
            ccd_color: Optional color for CCD bars.
        """
        if len(binary_df) == 0:
            return

        current_smiles_color = smiles_color if smiles_color else PlotConfig.SMILES_PRIMARY
        current_ccd_color = ccd_color if ccd_color else PlotConfig.CCD_PRIMARY

        smiles_col, ccd_col, axis_label = rmsd_metric
        sort_col = f"{ccd_col}_mean" # Sort by CCD RMSD mean

        # Sort by RMSD ascending (so highest RMSD is at the top of barh plot)
        # Ensure the sort column exists and handle potential NaNs by putting them at the bottom (or top if ascending=False)
        if sort_col in binary_df.columns:
            binary_df_sorted = binary_df.sort_values(by=sort_col, ascending=True, na_position='first')
        else:
            # Fallback if sort_col is somehow not there, though it should be.
            print(f"Warning: Sort column {sort_col} not found for binary plot. Using original order.")
            binary_df_sorted = binary_df.copy()

        n_structures = len(binary_df_sorted)
        
        plot_height_val = height
        if plot_height_val is None:
            structure_height_per_entry = 2 * BAR_HEIGHT + BAR_SPACING # Two bars (SMILES, CCD) per PDB
            plot_height_val = max(4, 1.5 + n_structures * structure_height_per_entry) # Min height 4, 1.5 for margins/title
            # Ensure height is not excessively large if few structures
            if n_structures < 5: # Adjust this threshold as needed
                 plot_height_val = max(plot_height_val, 4) # Ensure a reasonable minimum for few entries


        fig, ax = plt.subplots(1, 1, figsize=(width, plot_height_val))
        
        # Create PDB ID labels with asterisk for newer structures
        pdb_labels = [
            f"{pdb}*" if date > PlotConfig.AF3_CUTOFF else pdb 
            for pdb, date in zip(binary_df_sorted['PDB_ID'], binary_df_sorted['RELEASE_DATE'])
        ]
        
        y_positions = np.arange(n_structures)
        
        smiles_offset = -BAR_HEIGHT/2 - BAR_SPACING/2
        ccd_offset = BAR_HEIGHT/2 + BAR_SPACING/2
        
        smiles_mean_col = f"{smiles_col}_mean"
        smiles_std_col = f"{smiles_col}_std"
        ccd_mean_col = f"{ccd_col}_mean"
        ccd_std_col = f"{ccd_col}_std"

        smiles_xerr = binary_df_sorted[smiles_std_col].fillna(0).values if smiles_std_col in binary_df_sorted else np.zeros(n_structures)
        ccd_xerr = binary_df_sorted[ccd_std_col].fillna(0).values if ccd_std_col in binary_df_sorted else np.zeros(n_structures)
        
        smiles_values = binary_df_sorted[smiles_mean_col].fillna(0).values if smiles_mean_col in binary_df_sorted else np.zeros(n_structures)
        # Capture the bar container for SMILES bars
        smiles_container = ax.barh(
            y_positions + smiles_offset, smiles_values, height=BAR_HEIGHT,
            color=current_smiles_color, edgecolor='black', 
            linewidth=0.5, xerr=smiles_xerr,
            error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}
        )
        
        ccd_values = binary_df_sorted[ccd_mean_col].fillna(0).values if ccd_mean_col in binary_df_sorted else np.zeros(n_structures)
        # Capture the bar container for CCD bars
        ccd_container = ax.barh(
            y_positions + ccd_offset, ccd_values, height=BAR_HEIGHT,
            color=current_ccd_color, edgecolor='black', 
            linewidth=0.5, xerr=ccd_xerr,
            error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}
        )
        
        # Create legend handles using actual plotted artists
        legend_handles = []
        if n_structures > 0: # Ensure containers are not empty
            # CCD Handle (typically blue)
            ccd_handle = ccd_container[0] 
            ccd_handle.set_label('Ligand CCD')
            legend_handles.append(ccd_handle)

            # SMILES Handle (typically red/orange)
            smiles_handle = smiles_container[0]
            smiles_handle.set_label('Ligand SMILES')
            legend_handles.append(smiles_handle)

        if add_threshold and threshold_value is not None:
            ax.axvline(
                x=threshold_value, color='black', linestyle='--', 
                alpha=0.7, linewidth=1.0
            )
            from matplotlib.lines import Line2D
            threshold_line = Line2D([0], [0], color='black', linestyle='--', 
                                    alpha=0.7, linewidth=1.0, label='Threshold')
            legend_handles.append(threshold_line)
        
        ax.set_xlabel(axis_label, fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax.set_ylabel('PDB Identifier', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(pdb_labels)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        
        ax.set_xlim(0.0)
        # Adjust x-max if all values are small, e.g. all RMSD < 1
        all_rmsds = np.concatenate([smiles_values, ccd_values])
        max_rmsd_val = np.nanmax(all_rmsds) if len(all_rmsds) > 0 else 0
        if max_rmsd_val > 0:
             ax.set_xlim(0.0, max(max_rmsd_val * 1.1, ax.get_xlim()[1] if threshold_value is None else max(threshold_value * 1.1, max_rmsd_val * 1.1) ))


        ax.set_ylim(-0.5, n_structures - 0.5)
        
        ax.legend(handles=legend_handles, loc='best', framealpha=0, edgecolor='none', fontsize=LEGEND_FONTSIZE)
        
        plt.tight_layout()
        
        if save:
            sanitized_title = category_title.replace(' ', '_')
            sanitized_label = axis_label.replace(' (Å)', '_A').replace(' ', '_')
            filename = f"plot_{sanitized_title}_{sanitized_label}_all.png"
            save_figure(fig, filename)
        
        all_figures.append(fig)
        all_axes.append([ax]) # all_axes expects a list of axes arrays/lists
    
    def _create_paginated_plots(self, df, category_title, metrics, 
                               add_threshold, threshold_values, 
                               width, height, show_y_labels_on_all, save,
                               max_structures_per_plot, all_figures, all_axes,
                               last_category_label_for_ternary,
                               smiles_color: str | None = None,
                               ccd_color: str | None = None
                               ):
        """
        Create paginated plots for a category with many structures.
        
        Args:
            df: DataFrame containing structures in this category
            category_title: Title for the plot
            metrics: List of tuples (smiles_col, ccd_col, label) to plot
            add_threshold: Whether to add threshold lines
            threshold_values: Values for threshold lines
            width: Figure width
            height: Figure height (if None, calculated based on number of structures)
            show_y_labels_on_all: Whether to show y-labels on all subplots
            save: Whether to save the figure
            max_structures_per_plot: Maximum number of structures per plot
            all_figures, all_axes: Lists to append the created figures and axes to
            last_category_label_for_ternary: The label string of the last ternary category.
            smiles_color: Optional color for SMILES bars.
            ccd_color: Optional color for CCD bars.
        """
        # If the number of structures is small, create a single plot
        if len(df) <= max_structures_per_plot:
            self._create_category_plot(
                df, category_title, metrics, 
                add_threshold, threshold_values, 
                width, height, show_y_labels_on_all, save,
                all_figures, all_axes,
                last_category_label_for_ternary=last_category_label_for_ternary,
                smiles_color=smiles_color,
                ccd_color=ccd_color
            )
            return
        
        # Split the dataframe into chunks
        num_plots = math.ceil(len(df) / max_structures_per_plot)
        
        # Create a plot for each chunk
        for i in range(num_plots):
            start_idx = i * max_structures_per_plot
            end_idx = min((i + 1) * max_structures_per_plot, len(df))
            
            page_df = df.iloc[start_idx:end_idx].copy()
            
            page_title_with_info = f"{category_title} (Page {i+1} of {num_plots})"
            
            self._create_category_plot(
                page_df, category_title, metrics, 
                add_threshold, threshold_values, 
                width, height, show_y_labels_on_all, save,
                all_figures, all_axes,
                filename_with_page=page_title_with_info,
                last_category_label_for_ternary=last_category_label_for_ternary,
                smiles_color=smiles_color,
                ccd_color=ccd_color
            )
    
    def _create_category_plot(self, category_df, category_title, metrics, 
                             add_threshold, threshold_values, 
                             width, height, show_y_labels_on_all, save,
                             all_figures, all_axes,
                             filename_with_page=None,
                             last_category_label_for_ternary=None,
                             smiles_color: str | None = None,
                             ccd_color: str | None = None
                             ):
        """
        Create a plot for a specific category of structures.
        
        Args:
            category_df: DataFrame containing only the structures in this category
            category_title: Title for the plot
            metrics: List of tuples (smiles_col, ccd_col, label) to plot
            add_threshold: Whether to add threshold lines
            threshold_values: Values for threshold lines
            width: Figure width
            height: Figure height (if None, calculated based on number of structures)
            show_y_labels_on_all: Whether to show y-labels on all subplots
            save: Whether to save the figure
            all_figures, all_axes: Lists to append the created figure and axes to
            filename_with_page: If provided, use this for generating filenames with page info
            last_category_label_for_ternary: The label string of the last ternary category.
            smiles_color: Optional color for SMILES bars.
            ccd_color: Optional color for CCD bars.
        """
        if len(category_df) == 0:
            return
            
        # Create PDB ID labels with asterisk for newer structures
        pdb_labels = [
            f"{pdb}*" if date > PlotConfig.AF3_CUTOFF else pdb 
            for pdb, date in zip(category_df['PDB_ID'], category_df['RELEASE_DATE'])
        ]
        
        # Set up figure and subplots
        n_metrics = len(metrics)
        n_pdbs = len(category_df)
        
        # Calculate plot height if not provided
        plot_height = height
        if plot_height is None:
            # Calculate height based on number of structures and bar height
            # Each structure needs space for two bars (SMILES and CCD) with spacing
            structure_height = 2 * BAR_HEIGHT + BAR_SPACING
            plot_height = max(4, 1.5 + n_pdbs * structure_height)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(width, plot_height), 
                                 sharey=not show_y_labels_on_all)
        
        # Ensure axes is a list even with one metric
        if n_metrics == 1:
            axes = [axes]
        
        # Y-axis positions
        y_positions = np.arange(n_pdbs)
        
        # Bar offsets
        smiles_offset = -BAR_HEIGHT/2 - BAR_SPACING/2
        ccd_offset = BAR_HEIGHT/2 + BAR_SPACING/2
        
        # Create legend handles
        current_smiles_color = smiles_color if smiles_color else PlotConfig.SMILES_PRIMARY
        current_ccd_color = ccd_color if ccd_color else PlotConfig.CCD_PRIMARY
        smiles_handle = plt.Rectangle((0, 0), 1, 1, 
                                     facecolor=current_smiles_color, 
                                     edgecolor='black', linewidth=0.5, 
                                     label='Ligand SMILES')
        ccd_handle = plt.Rectangle((0, 0), 1, 1, 
                                  facecolor=current_ccd_color, 
                                  edgecolor='black', linewidth=0.5, 
                                  label='Ligand CCD')
        legend_handles = [ccd_handle, smiles_handle]
        
        # Check if this is a binary structure plot with all RMSD metrics
        is_binary_plot = "Binary" in category_title and all(metric[2] == "RMSD (Å)" for metric in metrics)
        
        # Plot each metric
        for i, (smiles_col, ccd_col, axis_label) in enumerate(metrics):
            ax = axes[i]
            
            # Define mean and std columns
            smiles_mean_col = f"{smiles_col}_mean"
            smiles_std_col = f"{smiles_col}_std"
            ccd_mean_col = f"{ccd_col}_mean"
            ccd_std_col = f"{ccd_col}_std"
            
            # Replace NaN with 0 for error bars
            smiles_xerr = category_df[smiles_std_col].fillna(0).values
            ccd_xerr = category_df[ccd_std_col].fillna(0).values
            
            # For binary structures or if metric values are NaN, skip plotting bars but keep positions
            is_binary_df = 'is_binary' in category_df.columns and category_df['is_binary'].any()
            
            # Plot SMILES bars (only for non-binary or if values exist)
            smiles_y = y_positions + smiles_offset
            smiles_values = category_df[smiles_mean_col].fillna(0).values
            # For binary structures, set to 0 width to make it invisible
            if is_binary_df and 'DOCKQ' in smiles_col:
                smiles_mask = ~category_df[smiles_mean_col].isna().values
                # Only plot for non-NaN values
                if any(smiles_mask):
                    ax.barh(
                        smiles_y[smiles_mask], smiles_values[smiles_mask], height=BAR_HEIGHT,
                        color=current_smiles_color, edgecolor='black', 
                        linewidth=0.5, xerr=smiles_xerr[smiles_mask],
                        error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}
                    )
            else:
                ax.barh(
                    smiles_y, smiles_values, height=BAR_HEIGHT,
                    color=current_smiles_color, edgecolor='black', 
                    linewidth=0.5, xerr=smiles_xerr,
                    error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}
                )
            
            # Plot CCD bars (only for non-binary or if values exist)
            ccd_y = y_positions + ccd_offset
            ccd_values = category_df[ccd_mean_col].fillna(0).values
            # For binary structures, set to 0 width to make it invisible
            if is_binary_df and 'DOCKQ' in ccd_col:
                ccd_mask = ~category_df[ccd_mean_col].isna().values
                # Only plot for non-NaN values
                if any(ccd_mask):
                    ax.barh(
                        ccd_y[ccd_mask], ccd_values[ccd_mask], height=BAR_HEIGHT,
                        color=current_ccd_color, edgecolor='black', 
                        linewidth=0.5, xerr=ccd_xerr[ccd_mask],
                        error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}
                    )
            else:
                ax.barh(
                    ccd_y, ccd_values, height=BAR_HEIGHT,
                    color=current_ccd_color, edgecolor='black', 
                    linewidth=0.5, xerr=ccd_xerr,
                    error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}
                )
            
            # Add threshold if requested
            if add_threshold and i < len(threshold_values) and threshold_values[i] is not None:
                ax.axvline(
                    x=threshold_values[i], color='black', linestyle='--', 
                    alpha=0.7, linewidth=1.0
                )
                
                # Add threshold to legend if first subplot
                if i == 0:
                    from matplotlib.lines import Line2D
                    threshold_line = Line2D([0], [0], color='black', linestyle='--', 
                                          alpha=0.7, linewidth=1.0, label='Threshold')
                    legend_handles.append(threshold_line)
            
            # Add annotation for binary structures if DockQ metrics
            if is_binary_df and ('DOCKQ' in smiles_col or 'DOCKQ' in ccd_col) and not is_binary_plot:
                for idx, is_binary in enumerate(category_df['is_binary']):
                    if is_binary:
                        y_pos = y_positions[idx]
                        ax.text(
                            0.05, y_pos, 'Binary', 
                            verticalalignment='center', 
                            fontsize=ANNOTATION_FONTSIZE,
                            color=PlotConfig.GRAY
                        )
            
            # Set axis labels
            ax.set_xlabel(axis_label, fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
            if i == 0:
                ax.set_ylabel('PDB Identifier', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
            
            # Set y-ticks and labels
            ax.set_yticks(y_positions)
            if i == 0 or show_y_labels_on_all:
                ax.set_yticklabels(pdb_labels)
            
            ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
            
            # Set axis limits
            if "DockQ" in axis_label:
                ax.set_xlim(0.0, 1.10)
            else:
                ax.set_xlim(0.0) # Keep lower bound at 0 for other plots
            ax.set_ylim(-0.5, len(category_df) - 0.5)
        
        # Add legend to the first subplot (RMSD plot) only if it's NOT the last category plot
        current_plot_label_simple = category_title.replace("Ternary:", "").strip()
        if not (category_title.startswith("Ternary:") and 
                last_category_label_for_ternary is not None and 
                current_plot_label_simple == last_category_label_for_ternary):
            axes[0].legend(
                handles=legend_handles,
                loc='best',
                framealpha=0,
                edgecolor='none',
                fontsize=LEGEND_FONTSIZE
            )
        
        # Add category as title
        # For binary structure plots with RMSD metrics, add a note about missing DockQ metrics
        if is_binary_plot:
            # Use a modified title for binary structures
            pass # Binary plot title is handled/removed in _create_compact_binary_plots
        else:
            # Check if this is a ternary structure plot
            if category_title.startswith("Ternary:"):
                # Extract the RMSD range from the title
                rmsd_range = category_title.replace("Ternary:", "").strip()
                fig.suptitle(f"RMSD: {rmsd_range}", fontsize=PlotConfig.TITLE_SIZE)
            else:
                # For other categories, use the original format
                fig.suptitle(f"Category: {category_title}", fontsize=PlotConfig.TITLE_SIZE)
            
        plt.tight_layout()
        
        # Save if requested
        if save:
            sanitized_category = category_title.replace('<', 'lt').replace('>', 'gt').replace(' ', '_').replace('-', 'to')
            if filename_with_page and "Page" in filename_with_page:
                page_info = filename_with_page.split("(Page")[1].split(")")[0].strip().replace(" ", "_")
                sanitized_category = f"{sanitized_category}_page_{page_info}"
            filename = f"plot_category_{sanitized_category}"
            save_figure(fig, filename)
        
        all_figures.append(fig)
        all_axes.append(axes)