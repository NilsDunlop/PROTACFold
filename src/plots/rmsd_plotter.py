import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from base_plotter import BasePlotter
from config import PlotConfig
from data_loader import DataLoader
from utils import categorize_by_cutoffs, save_figure, distribute_structures_evenly

class RMSDPlotter(BasePlotter):
    """Class for creating RMSD, iRMSD, and LRMSD comparison plots."""
    
    def plot_rmsd_bars(self, df_agg, molecule_type="PROTAC", classification_cutoff=None,
                     add_threshold=False, threshold_value=2.0,
                     show_y_labels_on_all=False, width=12, height=14, 
                     bar_height=0.18, bar_spacing=0.08, save=False, 
                     max_structures_per_plot=20):
        """
        Create horizontal bar plots comparing RMSD, iRMSD, and LRMSD metrics for SMILES and CCD.
        
        Args:
            df_agg: Aggregated DataFrame with mean and std values
            molecule_type: Type of molecule to filter by (e.g., "PROTAC")
            classification_cutoff: List of cutoff values for categories
            add_threshold: Whether to add a threshold line
            threshold_value: Value for the threshold line
            show_y_labels_on_all: Whether to show y-labels on all subplots
            width, height: Figure dimensions
            bar_height, bar_spacing: Bar dimensions and spacing
            save: Whether to save the figure
            max_structures_per_plot: Maximum number of structures per plot
            
        Returns:
            Lists of created figures and axes
        """
        # Filter by molecule type and only include ternary structures
        df_filtered = df_agg[df_agg['TYPE'] == molecule_type].copy()
        
        # Identify binary structures
        df_filtered = DataLoader.identify_binary_structures(df_filtered)
        
        # Filter out binary structures since they don't have iRMSD and LRMSD values
        df_ternary = df_filtered[~df_filtered['is_binary']].copy()
        
        # Convert non-numeric values to NaN for all numeric columns
        for col in df_ternary.columns:
            if col.endswith('_mean') or col.endswith('_std'):
                df_ternary[col] = pd.to_numeric(df_ternary[col], errors='coerce')
        
        # Sort by release date
        if 'RELEASE_DATE' in df_ternary.columns:
            df_ternary['RELEASE_DATE'] = pd.to_datetime(df_ternary['RELEASE_DATE'])
            df_ternary = df_ternary.sort_values('RELEASE_DATE', ascending=True)
        
        # Default cutoffs based on molecule type if not provided
        if classification_cutoff is None:
            # Use the predefined cutoffs or calculate them
            ccd_rmsd_mean = df_ternary['CCD_RMSD_mean'].dropna()
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
        df_ternary = categorize_by_cutoffs(
            df_ternary, 'CCD_RMSD_mean', classification_cutoff, 'Category'
        )
        
        # Get category labels
        category_labels = [
            f"< {classification_cutoff[0]:.2f}",
            f"{classification_cutoff[0]:.2f} - {classification_cutoff[1]:.2f}",
            f"{classification_cutoff[1]:.2f} - {classification_cutoff[2]:.2f}",
            f"{classification_cutoff[2]:.2f} - {classification_cutoff[3]:.2f}",
            f"> {classification_cutoff[3]:.2f}"
        ]
        
        # Store all created figures and axes
        all_figures = []
        all_axes = []
        
        # Create plots for each category
        for category in category_labels:
            category_df = df_ternary[df_ternary['Category'] == category]
            
            if len(category_df) == 0:
                continue
                
            # Simple category title
            category_title = f"RMSD: {category}"
            
            # Create paginated plots for this category
            self._create_rmsd_plots(
                category_df, category_title, 
                add_threshold, threshold_value, 
                width, height, bar_height, bar_spacing,
                show_y_labels_on_all, save,
                max_structures_per_plot, all_figures, all_axes
            )
        
        return all_figures, all_axes
    
    def _create_rmsd_plots(self, df, category_title, 
                         add_threshold, threshold_value, 
                         width, height, bar_height, bar_spacing,
                         show_y_labels_on_all, save,
                         max_structures_per_plot, all_figures, all_axes):
        """
        Create RMSD, iRMSD, and LRMSD plots for a category of structures.
        Paginate the plots if there are too many structures.
        
        Args:
            df: DataFrame containing structures in this category
            category_title: Title for the plot
            add_threshold: Whether to add threshold lines
            threshold_value: Value for threshold line
            width, height: Figure dimensions
            bar_height, bar_spacing: Bar dimensions and spacing
            show_y_labels_on_all: Whether to show y-labels on all subplots
            save: Whether to save the figure
            max_structures_per_plot: Maximum number of structures per plot
            all_figures, all_axes: Lists to append the created figures and axes to
        """
        # Clean up category title - remove "Ternary: " prefix if present
        display_title = category_title
        if display_title.startswith("Ternary: "):
            display_title = "RMSD: " + display_title.replace("Ternary:", "").strip()
        elif display_title.startswith("RMSD Category: "):
            display_title = "RMSD: " + display_title.replace("RMSD Category:", "").strip()
        
        # If no structures, return immediately
        if len(df) == 0:
            return
            
        # Use the utility function for even distribution across pages
        pages, structures_per_page = distribute_structures_evenly(df, max_structures_per_plot)
        
        # Create plots with consistent number of structures per page
        for i, page_df in enumerate(pages):
            # Calculate plot dimensions based on number of structures
            plot_width, plot_height = self.calculate_plot_dimensions(len(page_df), width)
            
            # Create page filename info that includes pagination info
            page_filename = f"{category_title} (Page {i+1} of {len(pages)})"
            
            # Create the plot - don't include page numbers in the displayed title
            self._create_single_rmsd_plot(
                page_df, display_title,
                add_threshold, threshold_value, 
                plot_width, plot_height, bar_height, bar_spacing,
                show_y_labels_on_all, save,
                all_figures, all_axes,
                filename_with_page=page_filename  # Only used for filename, not display
            )
    
    def _create_single_rmsd_plot(self, df, category_title, 
                               add_threshold, threshold_value, 
                               width, height, bar_height, bar_spacing,
                               show_y_labels_on_all, save,
                               all_figures, all_axes,
                               filename_with_page=None):
        """
        Create a single RMSD plot with two panels (SMILES and CCD) comparing RMSD, iRMSD, and LRMSD.
        
        Args:
            df: DataFrame containing structures
            category_title: Title for the plot
            add_threshold: Whether to add threshold lines
            threshold_value: Value for threshold line
            width, height: Figure dimensions
            bar_height, bar_spacing: Bar dimensions and spacing
            show_y_labels_on_all: Whether to show y-labels on all subplots
            save: Whether to save the figure
            all_figures, all_axes: Lists to append the created figure and axes to
            filename_with_page: If provided, use this for generating filenames with page info
        """
        if len(df) == 0:
            return
        
        # Sort by release date (ascending)
        df_sorted = df.sort_values('RELEASE_DATE', ascending=True).reset_index(drop=True)
        
        # Create PDB ID labels with asterisk for newer structures
        pdb_labels = [
            f"{pdb}*" if date > PlotConfig.AF3_CUTOFF else pdb 
            for pdb, date in zip(df_sorted['PDB_ID'], df_sorted['RELEASE_DATE'])
        ]
        
        # Create figure with two subplots (SMILES on left, CCD on right)
        fig, (ax_smiles, ax_ccd) = plt.subplots(1, 2, figsize=(width, height), sharey=True)
        
        # Y-axis positions (one position per PDB)
        y_positions = np.arange(len(df_sorted))
        
        # Define metrics and their properties
        metrics = [
            # SMILES metrics - (column_name, color, label, axis)
            ('SMILES_RMSD_mean', PlotConfig.SMILES_PRIMARY, 'RMSD', ax_smiles, 2),
            ('SMILES_DOCKQ_iRMSD_mean', PlotConfig.SMILES_SECONDARY, 'iRMSD', ax_smiles, 1),
            ('SMILES_DOCKQ_LRMSD_mean', PlotConfig.SMILES_TERTIARY, 'LRMSD', ax_smiles, 0),
            # CCD metrics
            ('CCD_RMSD_mean', PlotConfig.CCD_PRIMARY, 'RMSD', ax_ccd, 2),
            ('CCD_DOCKQ_iRMSD_mean', PlotConfig.CCD_SECONDARY, 'iRMSD', ax_ccd, 1),
            ('CCD_DOCKQ_LRMSD_mean', PlotConfig.CCD_TERTIARY, 'LRMSD', ax_ccd, 0)
        ]
        
        # Create legend handles for each subplot
        legend_handles_smiles = []
        legend_handles_ccd = []
        
        # Define positions for each metric type
        positions = {
            'RMSD': 2,    # Will be the top bar
            'iRMSD': 1,   # Will be the middle bar
            'LRMSD': 0    # Will be the bottom bar
        }
        
        # Define exact positions for each bar type
        bar_positions = {}
        for metric_type, position in positions.items():
            bar_positions[metric_type] = (position - 1) * (bar_height + bar_spacing)
        
        # Plot each metric
        for col_name, color, label_type, ax, position in metrics:
            # Skip if column doesn't exist or has no data
            if col_name not in df_sorted.columns or df_sorted[col_name].isna().all():
                continue
            
            # Get corresponding std column if it exists
            std_col = col_name.replace('_mean', '_std')
            has_std = std_col in df_sorted.columns
            
            # Get position based on metric type
            bar_position = bar_positions[label_type]
            
            # For error bars, replace NaN with 0
            xerr = df_sorted[std_col].fillna(0).values if has_std else None
            
            # Plot the horizontal bar
            bars = ax.barh(
                y_positions + bar_position, 
                df_sorted[col_name].fillna(0).values, 
                height=bar_height,
                color=color, 
                edgecolor='black', 
                linewidth=PlotConfig.EDGE_WIDTH,
                xerr=xerr,
                error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1},
                label=f"{label_type}"
            )
            
            # Add to appropriate legend handles
            if ax == ax_smiles:
                legend_handles_smiles.append(bars)
            else:
                legend_handles_ccd.append(bars)
        
        # Add threshold if requested
        if add_threshold:
            # Add threshold line to SMILES plot
            threshold_line_smiles = ax_smiles.axvline(
                x=threshold_value, color='black', linestyle='--', 
                alpha=0.7, linewidth=1.0, label='Threshold'
            )
            
            # Add threshold line to CCD plot
            threshold_line_ccd = ax_ccd.axvline(
                x=threshold_value, color='black', linestyle='--', 
                alpha=0.7, linewidth=1.0, label='Threshold'
            )
            
            # Add threshold lines to legend handles
            legend_handles_smiles.append(threshold_line_smiles)
            legend_handles_ccd.append(threshold_line_ccd)
        
        # Set axis labels and titles
        ax_smiles.set_xlabel('Distance (Å)')
        ax_ccd.set_xlabel('Distance (Å)')
        ax_smiles.set_title('SMILES')  # Simplified title
        ax_ccd.set_title('CCD')  # Simplified title
        
        if not show_y_labels_on_all:
            ax_ccd.set_ylabel('')
        ax_smiles.set_ylabel('PDB Identifier')
        
        # Set y-ticks and labels
        ax_smiles.set_yticks(y_positions)
        ax_smiles.set_yticklabels(pdb_labels)
        if show_y_labels_on_all:
            ax_ccd.set_yticks(y_positions)
            ax_ccd.set_yticklabels(pdb_labels)
        
        # Set x-axis to start at 0
        ax_smiles.set_xlim(0)
        ax_ccd.set_xlim(0)
        
        # Set y-axis limits
        ax_smiles.set_ylim(-0.5, len(df_sorted) - 0.5)
        ax_ccd.set_ylim(-0.5, len(df_sorted) - 0.5)
        
        # Add legends
        if legend_handles_smiles:
            ax_smiles.legend(
                handles=legend_handles_smiles,
                loc='upper right',
                framealpha=0,
                edgecolor='none'
            )
        
        if legend_handles_ccd:
            ax_ccd.legend(
                handles=legend_handles_ccd,
                loc='upper right',
                framealpha=0,
                edgecolor='none'
            )
        
        # Add overall title without any page information
        fig.suptitle(category_title, fontsize=PlotConfig.TITLE_SIZE)
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            # Use the filename_with_page for the filename if provided
            save_title = filename_with_page if filename_with_page else category_title
            sanitized_category = save_title.replace('<', 'lt').replace('>', 'gt').replace(' ', '_').replace('-', 'to')
            
            # Handle pagination info in the filename
            if filename_with_page and "Page" in filename_with_page:
                base_category = filename_with_page.split(" (Page")[0]
                page_info = filename_with_page.split("(Page ")[1].split(")")[0].replace(" ", "_")
                sanitized_category = f"{base_category.replace('<', 'lt').replace('>', 'gt').replace(' ', '_').replace('-', 'to')}_page_{page_info}"
            
            filename = f"rmsd_plot_{sanitized_category}"
            save_figure(fig, filename)
        
        all_figures.append(fig)
        all_axes.append((ax_smiles, ax_ccd))
        
        return fig, (ax_smiles, ax_ccd) 