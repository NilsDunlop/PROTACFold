import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from base_plotter import BasePlotter
from config import PlotConfig
from data_loader import DataLoader
from utils import categorize_by_cutoffs, save_figure, distribute_structures_evenly
import logging
from typing import List, Tuple, Dict, Optional, Any, Union

class PTMPlotter(BasePlotter):
    """Class for creating pTM and ipTM comparison plots."""
    
    def plot_ptm_bars(self, df_agg, molecule_type="PROTAC", classification_cutoff=None,
                     add_threshold=False, threshold_value=0.5,
                     show_y_labels_on_all=True, width=12, height=14, 
                     bar_height=0.18, bar_spacing=0.08, save=False, 
                     max_structures_per_plot=17):
        """
        Create horizontal bar plots comparing pTM and ipTM metrics for SMILES and CCD.
        Only plots for categories below 0.8 and above 0.92.
        
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
        # Filter by molecule type
        df_filtered = df_agg[df_agg['TYPE'] == molecule_type].copy()
        
        # Convert non-numeric values to NaN for all numeric columns
        for col in df_filtered.columns:
            if col.endswith('_mean') or col.endswith('_std'):
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        
        # Sort by release date (ascending)
        if 'RELEASE_DATE' in df_filtered.columns:
            df_filtered['RELEASE_DATE'] = pd.to_datetime(df_filtered['RELEASE_DATE'])
            df_filtered = df_filtered.sort_values('RELEASE_DATE', ascending=False)  # Sort descending for most recent at top
        
        # Default cutoffs based on pTM values if not provided
        if classification_cutoff is None:
            # Use fixed cutoffs rather than percentiles to isolate specific ranges
            classification_cutoff = [0.8, 0.85, 0.9, 0.92]
        
        # Categorize data based on pTM score
        df_filtered = categorize_by_cutoffs(
            df_filtered, 'CCD_PTM_mean', classification_cutoff, 'Category'
        )
        
        # Get category labels
        category_labels = [
            f"< {classification_cutoff[0]:.2f}",
            f"{classification_cutoff[0]:.2f} - {classification_cutoff[1]:.2f}",
            f"{classification_cutoff[1]:.2f} - {classification_cutoff[2]:.2f}",
            f"{classification_cutoff[2]:.2f} - {classification_cutoff[3]:.2f}",
            f"> {classification_cutoff[3]:.2f}"
        ]
        
        # Only select the first category (below 0.8) and the last category (above 0.92)
        selected_categories = [category_labels[0], category_labels[-1]]
        
        # Store all created figures and axes
        all_figures = []
        all_axes = []
        
        # Create plots for selected categories only
        for category in selected_categories:
            category_df = df_filtered[df_filtered['Category'] == category]
            
            if len(category_df) == 0:
                continue
                
            # Simple category title
            category_title = f"pTM: {category}"
            
            # Create paginated plots for this category
            self._create_ptm_plots(
                category_df, category_title, 
                add_threshold, threshold_value, 
                width, height, bar_height, bar_spacing,
                show_y_labels_on_all, save,
                max_structures_per_plot, all_figures, all_axes
            )
        
        return all_figures, all_axes
    
    def _create_ptm_plots(self, df, category_title, 
                        add_threshold, threshold_value, 
                        width, height, bar_height, bar_spacing,
                        show_y_labels_on_all, save,
                        max_structures_per_plot, all_figures, all_axes):
        """
        Create pTM and ipTM plots for a category of structures.
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
        # Clean up category title
        display_title = category_title
        if display_title.startswith("pTM Category: "):
            display_title = "pTM: " + display_title.replace("pTM Category:", "").strip()
            
        # If the number of structures is relatively small, create a single plot with dynamic height
        if len(df) <= max_structures_per_plot:
            # Use the BasePlotter utility to calculate appropriate dimensions
            plot_width, plot_height = self.calculate_plot_dimensions(len(df), width)
            
            self._create_single_ptm_plot(
                df, display_title, 
                add_threshold, threshold_value, 
                plot_width, plot_height, bar_height, bar_spacing,
                show_y_labels_on_all, save,
                all_figures, all_axes
            )
            return
        
        # For larger datasets, use even distribution for consistent plots
        pages, structures_per_page = distribute_structures_evenly(df, max_structures_per_plot)
        
        # Create a plot for each page with consistent structure counts
        for i, page_df in enumerate(pages):
            # Calculate plot dimensions based on number of structures
            plot_width, plot_height = self.calculate_plot_dimensions(len(page_df), width)
            
            # Create page filename info that includes pagination info
            page_filename = f"{category_title} (Page {i+1} of {len(pages)})"
            
            # Create the plot - don't include page numbers in the displayed title
            self._create_single_ptm_plot(
                page_df, display_title,
                add_threshold, threshold_value, 
                plot_width, plot_height, bar_height, bar_spacing,
                show_y_labels_on_all, save,
                all_figures, all_axes,
                filename_with_page=page_filename  # Only used for filename, not display
            )
    
    def _create_single_ptm_plot(self, df, category_title, 
                              add_threshold, threshold_value, 
                              width, height, bar_height, bar_spacing,
                              show_y_labels_on_all, save,
                              all_figures, all_axes,
                              filename_with_page=None):
        """
        Create a single plot with two panels (pTM and ipTM) comparing SMILES and CCD metrics.
        
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
        if 'RELEASE_DATE' in df.columns:
            df['RELEASE_DATE'] = pd.to_datetime(df['RELEASE_DATE'])
            df = df.sort_values('RELEASE_DATE', ascending=False)  # Sort descending for most recent at top
        
        # Create PDB ID labels with asterisk for newer structures
        pdb_labels = [
            f"{pdb}*" if date > PlotConfig.AF3_CUTOFF else pdb 
            for pdb, date in zip(df['PDB_ID'], df['RELEASE_DATE'])
        ]
        
        # Create figure with two subplots (pTM on left, ipTM on right)
        fig, (ax_ptm, ax_iptm) = plt.subplots(1, 2, figsize=(width, height), sharey=True)
        
        # Y-axis positions (one position per PDB)
        y_positions = np.arange(len(df))
        
        # Define metrics to plot
        ptm_metrics = [
            # Column name, color, label, position offset
            ('CCD_PTM_mean', PlotConfig.CCD_PRIMARY, 'Ligand CCD', 0.5 * (bar_height + bar_spacing)),
            ('SMILES_PTM_mean', PlotConfig.SMILES_PRIMARY, 'Ligand SMILES', -0.5 * (bar_height + bar_spacing))
        ]
        
        iptm_metrics = [
            ('CCD_IPTM_mean', PlotConfig.CCD_PRIMARY, 'Ligand CCD', 0.5 * (bar_height + bar_spacing)),
            ('SMILES_IPTM_mean', PlotConfig.SMILES_PRIMARY, 'Ligand SMILES', -0.5 * (bar_height + bar_spacing))
        ]
        
        # Create legend handles for each subplot
        legend_handles_ptm = []
        legend_handles_iptm = []
        
        # Plot pTM metrics
        for col_name, color, label, offset in ptm_metrics:
            # Skip if column doesn't exist or has no data
            if col_name not in df.columns or df[col_name].isna().all():
                continue
            
            # Get corresponding std column if it exists
            std_col = col_name.replace('_mean', '_std')
            has_std = std_col in df.columns
            
            # For error bars, replace NaN with 0
            xerr = df[std_col].fillna(0).values if has_std else None
            
            # Plot the horizontal bar
            bars = ax_ptm.barh(
                y_positions + offset, 
                df[col_name].fillna(0).values, 
                height=bar_height,
                color=color, 
                edgecolor='black', 
                linewidth=PlotConfig.EDGE_WIDTH,
                xerr=xerr,
                error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1},
                label=label
            )
            
            # Add to legend handles
            legend_handles_ptm.append(bars)
        
        # Plot ipTM metrics
        for col_name, color, label, offset in iptm_metrics:
            # Skip if column doesn't exist or has no data
            if col_name not in df.columns or df[col_name].isna().all():
                continue
            
            # Get corresponding std column if it exists
            std_col = col_name.replace('_mean', '_std')
            has_std = std_col in df.columns
            
            # For error bars, replace NaN with 0
            xerr = df[std_col].fillna(0).values if has_std else None
            
            # Plot the horizontal bar
            bars = ax_iptm.barh(
                y_positions + offset, 
                df[col_name].fillna(0).values, 
                height=bar_height,
                color=color, 
                edgecolor='black', 
                linewidth=PlotConfig.EDGE_WIDTH,
                xerr=xerr,
                error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1},
                label=label
            )
            
            # Add to legend handles
            legend_handles_iptm.append(bars)
        
        # Add threshold if requested
        if add_threshold:
            # Add threshold line to pTM plot
            threshold_line_ptm = ax_ptm.axvline(
                x=threshold_value, color='black', linestyle='--', 
                alpha=0.7, linewidth=1.0, label='Threshold'
            )
            
            # Add threshold line to ipTM plot
            threshold_line_iptm = ax_iptm.axvline(
                x=threshold_value, color='black', linestyle='--', 
                alpha=0.7, linewidth=1.0, label='Threshold'
            )
            
            # Add threshold lines to legend handles
            legend_handles_ptm.append(threshold_line_ptm)
            legend_handles_iptm.append(threshold_line_iptm)
        
        # Set axis labels and titles
        ax_ptm.set_xlabel('pTM')
        ax_iptm.set_xlabel('ipTM')
        ax_ptm.set_title('pTM')  # Simplified title
        ax_iptm.set_title('ipTM')  # Simplified title
        
        # Reverse y-axis so earliest structures are at the top
        ax_ptm.invert_yaxis()
        ax_iptm.invert_yaxis()
        
        if not show_y_labels_on_all:
            ax_iptm.set_ylabel('')
        ax_ptm.set_ylabel('PDB Identifier')
        
        # Set y-ticks and labels
        ax_ptm.set_yticks(y_positions)
        ax_ptm.set_yticklabels(pdb_labels)
        if show_y_labels_on_all:
            ax_iptm.set_yticks(y_positions)
            ax_iptm.set_yticklabels(pdb_labels)
            # Ensure y-tick labels are visible on the right subplot
            plt.setp(ax_iptm.get_yticklabels(), visible=True)
            ax_iptm.tick_params(labelleft=True)
        
        # Set x-axis limits from 0 to 1 (for confidence scores)
        ax_ptm.set_xlim(0, 1.0)
        ax_iptm.set_xlim(0, 1.0)
        
        # Set y-axis limits
        ax_ptm.set_ylim(-0.5, len(df) - 0.5)
        ax_iptm.set_ylim(-0.5, len(df) - 0.5)
        
        # Add legends
        if legend_handles_ptm:
            ax_ptm.legend(
                handles=legend_handles_ptm,
                loc='lower right',
                framealpha=0,
                edgecolor='none'
            )
        
        if legend_handles_iptm:
            ax_iptm.legend(
                handles=legend_handles_iptm,
                loc='lower right',
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
            
            filename = f"ptm_plot_{sanitized_category}"
            save_figure(fig, filename)
        
        all_figures.append(fig)
        all_axes.append((ax_ptm, ax_iptm))
        
        return fig, (ax_ptm, ax_iptm)

    def plot_ptm_comparison(self, df, show_individual=True, save=True, add_threshold=False, threshold_value=0.8):
        """
        Create PTM (pTM and ipTM) comparison plots for different confidence level categories.
        Generates both individual structure plots and summary plots.
        
        Args:
            df: DataFrame containing the structures to plot
            show_individual: Whether to create plots for each individual category or just the summary
            save: Whether to save the figures
            add_threshold: Whether to add threshold lines
            threshold_value: Value for threshold line
        
        Returns:
            all_figures: List of generated figure objects
            all_axes: List of generated axes objects
        """
        all_figures = []
        all_axes = []
        
        # Create a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Get categories based on pTM values (SMILES or CCD)
        df['PTM_CATEGORY'] = self._categorize_ptm_values(df)
        
        # Define categories to plot (only plot below 0.8 and above 0.92)
        categories_to_plot = [
            {'range': (0, 0.8), 'title': 'PTM < 0.8', 'column': 'PTM_CATEGORY'},
            {'range': (0.92, 1.0), 'title': 'PTM > 0.92', 'column': 'PTM_CATEGORY'}
        ]
        
        # Dictionary to keep track of the number of plots in each category
        plot_counts = {cat['title']: 0 for cat in categories_to_plot}
        
        # Generate individual plots and count structures in each category
        if show_individual:
            logging.info("Creating individual PTM comparison plots by category")
            
            for category in categories_to_plot:
                # Filter structures based on the category
                category_df = df[(df[category['column']] >= category['range'][0]) & 
                                (df[category['column']] < category['range'][1])]
                
                # Only proceed if there are structures in this category
                if len(category_df) == 0:
                    continue
                
                plot_counts[category['title']] = len(category_df)
                
                # Use the utility function for even distribution across pages
                pages, structures_per_page = distribute_structures_evenly(
                    category_df, 
                    PlotConfig.MAX_STRUCTURES_PER_PTM_PLOT if hasattr(PlotConfig, 'MAX_STRUCTURES_PER_PTM_PLOT') else 12
                )
                
                # Create a plot for each page with consistent structure counts
                for i, page_df in enumerate(pages):
                    # Create title with page info
                    page_title = f"{category['title']} (Page {i+1} of {len(pages)})"
                    
                    # Create plot for this page
                    fig, ax = self._create_single_ptm_plot(
                        page_df, 
                        category['title'],  # Use original title for saving
                        add_threshold, 
                        threshold_value,
                        PlotConfig.PTM_PLOT_WIDTH if hasattr(PlotConfig, 'PTM_PLOT_WIDTH') else 14, 
                        min(PlotConfig.PTM_PLOT_HEIGHT if hasattr(PlotConfig, 'PTM_PLOT_HEIGHT') else 12, 
                            (PlotConfig.PTM_PLOT_HEIGHT_PER_STRUCTURE if hasattr(PlotConfig, 'PTM_PLOT_HEIGHT_PER_STRUCTURE') else 0.6) * len(page_df)),
                        PlotConfig.PTM_BAR_HEIGHT if hasattr(PlotConfig, 'PTM_BAR_HEIGHT') else 0.18,
                        PlotConfig.PTM_BAR_SPACING if hasattr(PlotConfig, 'PTM_BAR_SPACING') else 0.08,
                        show_y_labels_on_all=True,
                        save=save,
                        all_figures=all_figures,
                        all_axes=all_axes,
                        filename_with_page=page_title
                    )
                    
                    # Add plot title with page info
                    if fig is not None:
                        fig.suptitle(page_title, fontsize=PlotConfig.TITLE_SIZE)
                    
        # Create a summary plot with all categories
        logging.info("Creating summary PTM comparison plot across categories")
        
        # Create summary DataFrame with category counts
        summary_data = {
            'Category': [cat['title'] for cat in categories_to_plot],
            'Count': [plot_counts.get(cat['title'], 0) for cat in categories_to_plot]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Create summary figure if there's data to plot
        if summary_df['Count'].sum() > 0:
            # Create figure with one subplot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot horizontal bars for counts
            bars = ax.barh(
                summary_df['Category'],
                summary_df['Count'],
                color=PlotConfig.SUMMARY_BAR_COLOR,
                edgecolor='black',
                linewidth=PlotConfig.EDGE_WIDTH
            )
            
            # Add count labels at the end of each bar
            for bar in bars:
                width = bar.get_width()
                if width > 0:  # Only add label if count > 0
                    ax.text(
                        width + 1,  # Offset from end of bar
                        bar.get_y() + bar.get_height()/2,
                        f'{int(width)}',
                        va='center',
                        fontsize=10
                    )
            
            # Set plot title and labels
            ax.set_title('Number of Structures by PTM Category', fontsize=PlotConfig.TITLE_SIZE)
            ax.set_xlabel('Number of Structures', fontsize=PlotConfig.AXIS_LABEL_SIZE)
            ax.set_ylabel('PTM Category', fontsize=PlotConfig.AXIS_LABEL_SIZE)
            
            # Set y-axis limits to accommodate all categories
            ax.set_ylim(-0.5, len(summary_df) - 0.5)
            
            # Add grid lines for x-axis only
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure if requested
            if save:
                save_figure(fig, 'ptm_category_summary')
            
            all_figures.append(fig)
            all_axes.append(ax)
        
        return all_figures, all_axes 