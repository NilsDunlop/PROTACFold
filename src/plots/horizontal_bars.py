import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from base_plotter import BasePlotter
from config import PlotConfig
from data_loader import DataLoader
from utils import categorize_by_cutoffs, save_figure

class HorizontalBarPlotter(BasePlotter):
    """Class for creating horizontal bar plots comparing means with error bars."""
    
    def plot_bars(self, df_agg, molecule_type="PROTAC", classification_cutoff=None, 
                metrics=None, add_threshold=False, threshold_values=None,
                show_y_labels_on_all=False, width=10, height=8, 
                bar_height=0.3, bar_spacing=0.05, save=False, 
                max_structures_per_plot=20, binary_target_pages=2):
        """
        Create horizontal bar plots for different metrics, grouped by category.
        
        Args:
            df_agg: Aggregated DataFrame with mean and std values
            molecule_type: Type of molecule to filter by (e.g., "PROTAC")
            classification_cutoff: List of cutoff values for categories
            metrics: List of tuples (smiles_col, ccd_col, label) to plot
            add_threshold: Whether to add threshold lines
            threshold_values: Values for threshold lines
            show_y_labels_on_all: Whether to show y-labels on all subplots
            width, height: Figure dimensions
            bar_height, bar_spacing: Bar dimensions and spacing
            save: Whether to save the figure
            max_structures_per_plot: Maximum number of structures to show in a single plot
            binary_target_pages: Target number of pages for binary structures
            
        Returns:
            Lists of created figures and axes
        """
        # Default metrics if none provided
        if metrics is None:
            metrics = [
                ('SMILES_DOCKQ_SCORE', 'CCD_DOCKQ_SCORE', 'DockQ Score'),
                ('SMILES_RMSD', 'CCD_RMSD', 'RMSD (Å)'),
                ('SMILES_DOCKQ_LRMSD', 'CCD_DOCKQ_LRMSD', 'LRMSD (Å)')
            ]
            
        # Special metrics for binary structures - use RMSD for all three panels
        binary_metrics = [
            ('SMILES_RMSD', 'CCD_RMSD', 'RMSD (Å)'),
            ('SMILES_RMSD', 'CCD_RMSD', 'RMSD (Å)'),
            ('SMILES_RMSD', 'CCD_RMSD', 'RMSD (Å)')
        ]
        
        # Default threshold values if needed but not provided
        if add_threshold and threshold_values is None:
            threshold_values = [0.23, 4, 4]
            
        # Threshold values for binary plots (use the RMSD threshold for all three panels)
        binary_threshold_values = [4, 4, 4] if add_threshold else None
        
        # Filter by molecule type
        df_filtered = df_agg[df_agg['TYPE'] == molecule_type].copy()
        
        # Convert non-numeric values to NaN for all numeric columns
        for col in df_filtered.columns:
            if col.endswith('_mean') or col.endswith('_std'):
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        
        # Sort by release date
        if 'RELEASE_DATE' in df_filtered.columns:
            df_filtered['RELEASE_DATE'] = pd.to_datetime(df_filtered['RELEASE_DATE'])
            df_filtered = df_filtered.sort_values('RELEASE_DATE', ascending=True)
        
        # Identify binary structures using DataLoader utility
        df_filtered = DataLoader.identify_binary_structures(df_filtered)
        
        # Default cutoffs based on molecule type if not provided
        if classification_cutoff is None:
            # Use the predefined cutoffs or calculate them
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
        
        # Categorize data
        df_filtered = categorize_by_cutoffs(
            df_filtered, 'CCD_RMSD_mean', classification_cutoff, 'Category'
        )
        
        # Get category labels
        category_labels = [
            f"< {classification_cutoff[0]:.2f}",
            f"{classification_cutoff[0]:.2f} - {classification_cutoff[1]:.2f}",
            f"{classification_cutoff[1]:.2f} - {classification_cutoff[2]:.2f}",
            f"{classification_cutoff[2]:.2f} - {classification_cutoff[3]:.2f}",
            f"> {classification_cutoff[3]:.2f}"
        ]
        
        # Create a special category for binary structures
        binary_category = "Binary Structures"
        
        # Store all created figures and axes
        all_figures = []
        all_axes = []
        
        # Handle binary structures separately - plot different PDBs in each panel
        binary_df = df_filtered[df_filtered['is_binary']]
        if len(binary_df) > 0:
            # Calculate optimal structures per panel to fill the target number of pages
            panels_per_page = 3
            total_panels = binary_target_pages * panels_per_page
            optimal_structures_per_panel = math.ceil(len(binary_df) / total_panels)
            
            # Limit by max_structures_per_plot
            structures_per_panel = min(optimal_structures_per_panel, max_structures_per_plot)
            
            # Create special binary structure plots with different PDBs in each panel
            self._create_compact_binary_plots(
                binary_df, binary_category, 
                ('SMILES_RMSD', 'CCD_RMSD', 'RMSD (Å)'),
                add_threshold, binary_threshold_values[0] if binary_threshold_values else None, 
                width, height, bar_height, bar_spacing,
                show_y_labels_on_all, save,
                structures_per_panel, all_figures, all_axes
            )
        
        # Then handle each category of ternary structures
        for category in category_labels:
            # Filter for ternary structures in this category
            category_df = df_filtered[(df_filtered['Category'] == category) & (~df_filtered['is_binary'])]
            
            if len(category_df) == 0:
                continue
                
            # Create paginated plots for this category using standard metrics
            self._create_paginated_plots(
                category_df, f"Ternary: {category}", metrics, 
                add_threshold, threshold_values, 
                width, height, bar_height, bar_spacing,
                show_y_labels_on_all, save,
                max_structures_per_plot, all_figures, all_axes
            )
        
        return all_figures, all_axes
    
    def _create_compact_binary_plots(self, binary_df, category_title, 
                                    rmsd_metric, add_threshold, threshold_value, 
                                    width, height, bar_height, bar_spacing,
                                    show_y_labels_on_all, save,
                                    max_structures_per_panel, all_figures, all_axes):
        """
        Create compact binary structure plots with different PDBs in each panel.
        
        Args:
            binary_df: DataFrame containing binary structures
            category_title: Title for the plot
            rmsd_metric: Tuple with (smiles_col, ccd_col, label) for RMSD
            add_threshold: Whether to add threshold lines
            threshold_value: Value for threshold line
            width, height: Figure dimensions
            bar_height, bar_spacing: Bar dimensions and spacing
            show_y_labels_on_all: Whether to show y-labels on all subplots
            save: Whether to save the figure
            max_structures_per_panel: Maximum number of structures per panel
            all_figures, all_axes: Lists to append the created figures and axes to
        """
        if len(binary_df) == 0:
            return
            
        # Calculate optimal distribution for balanced panels
        panels_per_page = 3  # Fixed number of panels per page
        
        # Target exactly 2 pages if possible
        target_pages = 2
        
        # Calculate structures per panel to fill exactly target_pages
        optimal_structures_per_panel = math.ceil(len(binary_df) / (target_pages * panels_per_page))
        
        # Use max_structures_per_panel as an upper limit
        structures_per_panel = min(optimal_structures_per_panel, max_structures_per_panel)
        
        # Recalculate number of pages needed
        total_panels_needed = math.ceil(len(binary_df) / structures_per_panel)
        num_plots = math.ceil(total_panels_needed / panels_per_page)
        
        # Process each plot
        for plot_idx in range(num_plots):
            # Set up figure and subplots
            fig, axes = plt.subplots(1, 3, figsize=(width, height), sharey=False)
            
            # Track if any panel has content for this page
            page_has_content = False
            
            # Process each panel in the figure
            for panel_idx in range(panels_per_page):
                # Calculate the starting index for this panel
                start_idx = plot_idx * panels_per_page * structures_per_panel + panel_idx * structures_per_panel
                
                # Skip if we've run out of structures
                if start_idx >= len(binary_df):
                    # Empty panel - hide axis
                    axes[panel_idx].set_visible(False)
                    continue
                
                # Get the structures for this panel
                end_idx = min(start_idx + structures_per_panel, len(binary_df))
                panel_df = binary_df.iloc[start_idx:end_idx].copy()
                
                # Create the panel plot
                self._create_rmsd_panel(
                    panel_df, axes[panel_idx], rmsd_metric,
                    add_threshold, threshold_value,
                    bar_height, bar_spacing,
                    panel_idx == 0 or show_y_labels_on_all
                )
                
                page_has_content = True
            
            # Skip adding this page if no content was added
            if not page_has_content:
                plt.close(fig)
                continue
                
            # Add overall plot title without page numbers
            fig.suptitle("Binary - RMSD Values", fontsize=PlotConfig.TITLE_SIZE)
            
            # Add legend to the last panel that's visible
            legend_panel = 2
            while legend_panel >= 0 and not axes[legend_panel].get_visible():
                legend_panel -= 1
                
            if legend_panel >= 0:
                # Create legend handles
                smiles_handle = plt.Rectangle((0, 0), 1, 1, 
                                            facecolor=PlotConfig.SMILES_PRIMARY, 
                                            edgecolor='black', linewidth=0.5, 
                                            label='Ligand SMILES')
                ccd_handle = plt.Rectangle((0, 0), 1, 1, 
                                            facecolor=PlotConfig.CCD_PRIMARY, 
                                            edgecolor='black', linewidth=0.5, 
                                            label='Ligand CCD')
                
                legend_handles = [ccd_handle, smiles_handle]
                
                # Add threshold to legend if requested
                if add_threshold and threshold_value is not None:
                    from matplotlib.lines import Line2D
                    threshold_line = Line2D([0], [0], color='black', linestyle='--', 
                                            alpha=0.7, linewidth=1.0, label='Threshold')
                    legend_handles.append(threshold_line)
                
                axes[legend_panel].legend(
                    handles=legend_handles,
                    loc='upper right',
                    framealpha=0,
                    edgecolor='none'
                )
            
            plt.tight_layout()
            
            # Save if requested
            if save:
                sanitized_category = "Binary_Structures"
                if num_plots > 1:
                    page_number = f"page_{plot_idx+1}of{num_plots}"
                    sanitized_category = f"{sanitized_category}_{page_number}"
                filename = f"plot_category_{sanitized_category}"
                save_figure(fig, filename)
            
            all_figures.append(fig)
            all_axes.append(axes)
    
    def _create_rmsd_panel(self, df, ax, rmsd_metric, 
                          add_threshold, threshold_value,
                          bar_height, bar_spacing,
                          show_y_labels):
        """
        Create a single panel for RMSD values.
        
        Args:
            df: DataFrame containing structures for this panel
            ax: Axis to plot on
            rmsd_metric: Tuple with (smiles_col, ccd_col, label) for RMSD
            add_threshold: Whether to add threshold line
            threshold_value: Value for threshold line
            bar_height, bar_spacing: Bar dimensions and spacing
            show_y_labels: Whether to show y-labels on this panel
        """
        if len(df) == 0:
            ax.set_visible(False)
            return
            
        # Unpack the metric
        smiles_col, ccd_col, axis_label = rmsd_metric
        
        # Define mean and std columns
        smiles_mean_col = f"{smiles_col}_mean"
        smiles_std_col = f"{smiles_col}_std"
        ccd_mean_col = f"{ccd_col}_mean"
        ccd_std_col = f"{ccd_col}_std"
        
        # Create PDB ID labels with asterisk for newer structures
        pdb_labels = [
            f"{pdb}*" if date > PlotConfig.AF3_CUTOFF else pdb 
            for pdb, date in zip(df['PDB_ID'], df['RELEASE_DATE'])
        ]
        
        # Y-axis positions
        y_positions = np.arange(len(df))
        
        # Bar offsets
        smiles_offset = -bar_height/2 - bar_spacing/2
        ccd_offset = bar_height/2 + bar_spacing/2
        
        # Replace NaN with 0 for error bars
        smiles_xerr = df[smiles_std_col].fillna(0).values
        ccd_xerr = df[ccd_std_col].fillna(0).values
        
        # Plot SMILES bars
        smiles_y = y_positions + smiles_offset
        smiles_values = df[smiles_mean_col].fillna(0).values
        ax.barh(
            smiles_y, smiles_values, height=bar_height,
            color=PlotConfig.SMILES_PRIMARY, edgecolor='black', 
            linewidth=0.5, xerr=smiles_xerr,
            error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}
        )
        
        # Plot CCD bars
        ccd_y = y_positions + ccd_offset
        ccd_values = df[ccd_mean_col].fillna(0).values
        ax.barh(
            ccd_y, ccd_values, height=bar_height,
            color=PlotConfig.CCD_PRIMARY, edgecolor='black', 
            linewidth=0.5, xerr=ccd_xerr,
            error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}
        )
        
        # Add threshold if requested
        if add_threshold and threshold_value is not None:
            ax.axvline(
                x=threshold_value, color='black', linestyle='--', 
                alpha=0.7, linewidth=1.0
            )
        
        # Set axis labels
        ax.set_xlabel(axis_label)
        if show_y_labels:
            ax.set_ylabel('PDB Identifier')
        
        # Set y-ticks and labels
        ax.set_yticks(y_positions)
        if show_y_labels:
            ax.set_yticklabels(pdb_labels)
        else:
            ax.set_yticklabels([])
        
        # Set axis limits
        ax.set_xlim(0.0)
        ax.set_ylim(-0.5, len(df) - 0.5)
    
    def _create_paginated_plots(self, df, category_title, metrics, 
                               add_threshold, threshold_values, 
                               width, height, bar_height, bar_spacing,
                               show_y_labels_on_all, save,
                               max_structures_per_plot, all_figures, all_axes):
        """
        Create paginated plots for a category with many structures.
        
        Args:
            df: DataFrame containing structures in this category
            category_title: Title for the plot
            metrics: List of tuples (smiles_col, ccd_col, label) to plot
            add_threshold: Whether to add threshold lines
            threshold_values: Values for threshold lines
            width, height: Figure dimensions
            bar_height, bar_spacing: Bar dimensions and spacing
            show_y_labels_on_all: Whether to show y-labels on all subplots
            save: Whether to save the figure
            max_structures_per_plot: Maximum number of structures per plot
            all_figures, all_axes: Lists to append the created figures and axes to
        """
        # If the number of structures is small, create a single plot
        if len(df) <= max_structures_per_plot:
            self._create_category_plot(
                df, category_title, metrics, 
                add_threshold, threshold_values, 
                width, height, bar_height, bar_spacing,
                show_y_labels_on_all, save,
                all_figures, all_axes
            )
            return
        
        # Split the dataframe into chunks
        num_plots = math.ceil(len(df) / max_structures_per_plot)
        
        # Create a plot for each chunk
        for i in range(num_plots):
            start_idx = i * max_structures_per_plot
            end_idx = min((i + 1) * max_structures_per_plot, len(df))
            
            # Get structures for this page
            page_df = df.iloc[start_idx:end_idx].copy()
            
            # For filenames only, create a version with page info
            page_title_with_info = f"{category_title} (Page {i+1} of {num_plots})"
            
            # Create the plot with the original title (no pagination info)
            self._create_category_plot(
                page_df, category_title, metrics, 
                add_threshold, threshold_values, 
                width, height, bar_height, bar_spacing,
                show_y_labels_on_all, save,
                all_figures, all_axes,
                filename_with_page=page_title_with_info  # Pass the page info for filename only
            )
    
    def _create_category_plot(self, category_df, category_title, metrics, 
                             add_threshold, threshold_values, 
                             width, height, bar_height, bar_spacing,
                             show_y_labels_on_all, save,
                             all_figures, all_axes,
                             filename_with_page=None):
        """
        Create a plot for a specific category of structures.
        
        Args:
            category_df: DataFrame containing only the structures in this category
            category_title: Title for the plot
            metrics: List of tuples (smiles_col, ccd_col, label) to plot
            add_threshold: Whether to add threshold lines
            threshold_values: Values for threshold lines
            width, height: Figure dimensions
            bar_height, bar_spacing: Bar dimensions and spacing
            show_y_labels_on_all: Whether to show y-labels on all subplots
            save: Whether to save the figure
            all_figures, all_axes: Lists to append the created figure and axes to
            filename_with_page: If provided, use this for generating filenames with page info
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
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(width, height), 
                                 sharey=not show_y_labels_on_all)
        
        # Ensure axes is a list even with one metric
        if n_metrics == 1:
            axes = [axes]
        
        # Y-axis positions
        y_positions = np.arange(n_pdbs)
        
        # Bar offsets
        smiles_offset = -bar_height/2 - bar_spacing/2
        ccd_offset = bar_height/2 + bar_spacing/2
        
        # Create legend handles
        smiles_handle = plt.Rectangle((0, 0), 1, 1, 
                                     facecolor=PlotConfig.SMILES_PRIMARY, 
                                     edgecolor='black', linewidth=0.5, 
                                     label='Ligand SMILES')
        ccd_handle = plt.Rectangle((0, 0), 1, 1, 
                                  facecolor=PlotConfig.CCD_PRIMARY, 
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
                        smiles_y[smiles_mask], smiles_values[smiles_mask], height=bar_height,
                        color=PlotConfig.SMILES_PRIMARY, edgecolor='black', 
                        linewidth=0.5, xerr=smiles_xerr[smiles_mask],
                        error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}
                    )
            else:
                ax.barh(
                    smiles_y, smiles_values, height=bar_height,
                    color=PlotConfig.SMILES_PRIMARY, edgecolor='black', 
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
                        ccd_y[ccd_mask], ccd_values[ccd_mask], height=bar_height,
                        color=PlotConfig.CCD_PRIMARY, edgecolor='black', 
                        linewidth=0.5, xerr=ccd_xerr[ccd_mask],
                        error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}
                    )
            else:
                ax.barh(
                    ccd_y, ccd_values, height=bar_height,
                    color=PlotConfig.CCD_PRIMARY, edgecolor='black', 
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
                            fontsize=PlotConfig.ANNOTATION_SIZE,
                            color=PlotConfig.GRAY
                        )
            
            # Set axis labels
            ax.set_xlabel(axis_label)
            if i == 0:
                ax.set_ylabel('PDB Identifier')
            
            # Set y-ticks and labels
            ax.set_yticks(y_positions)
            if i == 0 or show_y_labels_on_all:
                ax.set_yticklabels(pdb_labels)
            
            # Set axis limits
            ax.set_xlim(0.0)
            ax.set_ylim(-0.5, len(category_df) - 0.5)
        
        # Add legend to the last subplot
        axes[-1].legend(
            handles=legend_handles,
            loc='upper right',
            framealpha=0,
            edgecolor='none'
        )
        
        # Add category as title
        # For binary structure plots with RMSD metrics, add a note about missing DockQ metrics
        if is_binary_plot:
            # Use a modified title for binary structures
            fig.suptitle(f"Binary Structures - RMSD Values (No DockQ or LRMSD metrics)", 
                         fontsize=PlotConfig.TITLE_SIZE)
        else:
            # Check if this is a ternary structure plot
            if category_title.startswith("Ternary:"):
                # Extract the RMSD range from the title
                rmsd_range = category_title.replace("Ternary:", "").strip()
                fig.suptitle(f"Ternary RMSD: {rmsd_range}", fontsize=PlotConfig.TITLE_SIZE)
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