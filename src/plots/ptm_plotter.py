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

# --- Plotting Constants ---
PTM_FIG_WIDTH = 8.0       # Default width for the main PTM/ipTM comparison figure (inches)
PTM_INITIAL_MAX_FIG_HEIGHT = 14.0 # Initial max height for figures in plot_ptm_bars before dynamic calculation (inches)
PTM_BAR_HEIGHT_DATA = 0.30   # 'height' parameter for plt.barh (in data coordinates)
PTM_BAR_SPACING_DATA = 0.02  # Spacing factor (in data coordinates) used for offsetting bars

# Font Sizes
PTM_AXIS_LABEL_FONTSIZE = 12  # For ax.set_xlabel(), ax.set_ylabel()
PTM_TICK_LABEL_FONTSIZE = 10  # For tick labels (ax.tick_params labelsize)
PTM_LEGEND_FONTSIZE = 10      # For legend text

# For paginated plots in plot_ptm_comparison:
PTM_PAGE_WIDTH = PTM_FIG_WIDTH # Use the same width for these pages
PTM_PAGE_MAX_HEIGHT = 8.0     # Max height for a plot page (inches)
PTM_PAGE_HEIGHT_PER_STRUCTURE = 0.6 # Inches of plot height allocated per PDB ID on a page.
                                    # Figure height for page = min(PTM_PAGE_MAX_HEIGHT, PTM_PAGE_HEIGHT_PER_STRUCTURE * num_items)

class PTMPlotter(BasePlotter):
    """Class for creating pTM and ipTM comparison plots."""
    
    def plot_ptm_bars(self, df_agg, molecule_type="PROTAC", data_source="af3",
                     add_threshold=False, ptm_threshold_value=0.5, iptm_threshold_value=0.6,
                     show_y_labels_on_all=True, width=PTM_FIG_WIDTH, height=PTM_INITIAL_MAX_FIG_HEIGHT, 
                     bar_height=PTM_BAR_HEIGHT_DATA, bar_spacing=PTM_BAR_SPACING_DATA, save=False, 
                     max_structures_per_plot=17):
        """
        Create horizontal bar plots comparing pTM and ipTM metrics for SMILES and CCD
        for all structures of a given molecule_type. Plots are paginated if necessary.
        Sorted by CCD_PTM_mean (descending).
        
        Args:
            df_agg: Aggregated DataFrame with mean and std values
            molecule_type: Type of molecule to filter by (e.g., "PROTAC")
            data_source: Source of the data (e.g., "af3", "boltz1"), used for color and filename.
            add_threshold: Whether to add a threshold line
            ptm_threshold_value: Value for the pTM threshold line
            iptm_threshold_value: Value for the ipTM threshold line
            show_y_labels_on_all: Whether to show y-labels on all subplots
            width, height: Overall figure dimensions parameters (height may be adjusted dynamically)
            bar_height, bar_spacing: Bar dimensions and spacing for individual bars
            save: Whether to save the figure
            max_structures_per_plot: Maximum number of structures per plot page
            
        Returns:
            Lists of created figures and axes
        """
        # Filter by molecule type
        df_filtered = df_agg[df_agg['TYPE'] == molecule_type].copy()
        
        # Convert non-numeric values to NaN for all numeric columns that will be used for sorting or plotting
        cols_to_convert = ['CCD_PTM_mean', 'SMILES_PTM_mean', 'CCD_IPTM_mean', 'SMILES_IPTM_mean']
        for col_end in ['_std']:
            cols_to_convert.extend([c.replace('_mean', col_end) for c in cols_to_convert if '_mean' in c])
        
        for col in df_filtered.columns:
            if col.endswith('_mean') or col.endswith('_std'): # General conversion for all such columns
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        
        # Primary sort: CCD_PTM_mean (user has changed this to ascending in their local copy)
        if 'CCD_PTM_mean' in df_filtered.columns:
            df_filtered = df_filtered.sort_values('CCD_PTM_mean', ascending=True)
        else:
            logging.warning("'CCD_PTM_mean' column not found. Cannot sort by it. Plots may not be ordered as expected.")
            if 'RELEASE_DATE' in df_filtered.columns: 
                 df_filtered['RELEASE_DATE'] = pd.to_datetime(df_filtered['RELEASE_DATE'])
                 df_filtered = df_filtered.sort_values('RELEASE_DATE', ascending=False) # Fallback to date sort (recent first)

        if len(df_filtered) == 0:
            logging.warning(f"No data to plot for {data_source} {molecule_type} after filtering and sorting attempts.")
            return [], []
            
        all_figures = []
        all_axes = []
        
        plot_title_base = f"{data_source}_{molecule_type.replace(' ', '_').lower()}_ptm_iptm"
        if 'CCD_PTM_mean' in df_filtered.columns:
            plot_title_base += "_sorted_by_ccd_ptm"
        elif 'RELEASE_DATE' in df_filtered.columns:
            plot_title_base += "_sorted_by_date"
        
        self._create_ptm_plots(
            df_filtered, plot_title_base, data_source, 
            add_threshold, ptm_threshold_value, iptm_threshold_value, 
            width, height, bar_height, bar_spacing,
            show_y_labels_on_all, save,
            max_structures_per_plot, all_figures, all_axes
        )
        
        return all_figures, all_axes
    
    def _create_ptm_plots(self, df, plot_title_base, data_source,
                        add_threshold, ptm_threshold_value, iptm_threshold_value, 
                        width, height, bar_height, bar_spacing,
                        show_y_labels_on_all, save,
                        max_structures_per_plot, all_figures, all_axes):
        """
        Create pTM and ipTM plots for a category of structures.
        Paginate the plots if there are too many structures.
        
        Args:
            df: DataFrame containing structures in this category
            plot_title_base: Base for filename generation.
            data_source: Source of the data (e.g., "af3", "boltz1") for color selection.
            add_threshold: Whether to add threshold lines
            ptm_threshold_value: Value for threshold line on pTM plot
            iptm_threshold_value: Value for threshold line on ipTM plot
            width, height: Figure dimensions
            bar_height, bar_spacing: Bar dimensions and spacing
            show_y_labels_on_all: Whether to show y-labels on all subplots
            save: Whether to save the figure
            max_structures_per_plot: Maximum number of structures per plot
            all_figures, all_axes: Lists to append the created figures and axes to
        """
        display_title_for_filename = plot_title_base # This is now for filename, no visual title
            
        if len(df) <= max_structures_per_plot:
            plot_width, plot_height = self.calculate_plot_dimensions(len(df), width)
            self._create_single_ptm_plot(
                df, display_title_for_filename, data_source, 
                add_threshold, ptm_threshold_value, iptm_threshold_value, 
                plot_width, plot_height, bar_height, bar_spacing,
                show_y_labels_on_all, save,
                all_figures, all_axes
            )
            return
        
        pages, structures_per_page = distribute_structures_evenly(df, max_structures_per_plot)
        for i, page_df in enumerate(pages):
            plot_width, plot_height = self.calculate_plot_dimensions(len(page_df), width)
            page_filename = f"{display_title_for_filename} (Page {i+1} of {len(pages)})"
            self._create_single_ptm_plot(
                page_df, display_title_for_filename, data_source, 
                add_threshold, ptm_threshold_value, iptm_threshold_value, 
                plot_width, plot_height, bar_height, bar_spacing,
                show_y_labels_on_all, save,
                all_figures, all_axes,
                filename_with_page=page_filename
            )
    
    def _create_single_ptm_plot(self, df, filename_base, data_source, 
                              add_threshold, ptm_threshold_value, iptm_threshold_value, 
                              width, height, bar_height, bar_spacing,
                              show_y_labels_on_all, save,
                              all_figures, all_axes,
                              filename_with_page=None):
        """
        Create a single plot with two panels (pTM and ipTM) comparing SMILES and CCD metrics.
        
        Args:
            df: DataFrame containing structures
            filename_base: Base for filename generation (no visual title is used).
            data_source: Source of the data (e.g., "af3", "boltz1") for color selection.
            add_threshold: Whether to add threshold lines
            ptm_threshold_value: Value for threshold line on pTM plot
            iptm_threshold_value: Value for threshold line on ipTM plot
            width, height: Figure dimensions
            bar_height, bar_spacing: Bar dimensions and spacing
            show_y_labels_on_all: Whether to show y-labels on all subplots
            save: Whether to save the figure
            all_figures, all_axes: Lists to append the created figure and axes to
            filename_with_page: If provided, use this for generating filenames with page info
        """
        if len(df) == 0:
            return
        
        if 'RELEASE_DATE' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['RELEASE_DATE']):
            df['RELEASE_DATE'] = pd.to_datetime(df['RELEASE_DATE'], errors='coerce')

        pdb_labels = []
        if 'RELEASE_DATE' in df.columns and 'PDB_ID' in df.columns:
            pdb_labels = [
                f"{pdb}*" if pd.notna(date) and date > PlotConfig.AF3_CUTOFF else pdb 
                for pdb, date in zip(df['PDB_ID'], df['RELEASE_DATE'])
            ]
        elif 'PDB_ID' in df.columns:
            pdb_labels = df['PDB_ID'].tolist()
        else:
            pdb_labels = [f"Struct {i+1}" for i in range(len(df))] # Fallback labels

        fig, (ax_ptm, ax_iptm) = plt.subplots(1, 2, figsize=(width, height), sharey=True)
        y_positions = np.arange(len(df))
        
        # Determine colors based on data_source
        if data_source == "boltz1":
            ccd_color = PlotConfig.BOLTZ1_CCD_COLOR
            smiles_color = PlotConfig.BOLTZ1_SMILES_COLOR
        else: # Default to AF3 colors
            ccd_color = PlotConfig.CCD_PRIMARY
            smiles_color = PlotConfig.SMILES_PRIMARY

        ptm_metrics = [
            ('CCD_PTM_mean', ccd_color, 'Ligand CCD', 0.5 * (bar_height + bar_spacing)),
            ('SMILES_PTM_mean', smiles_color, 'Ligand SMILES', -0.5 * (bar_height + bar_spacing))
        ]
        iptm_metrics = [
            ('CCD_IPTM_mean', ccd_color, 'Ligand CCD', 0.5 * (bar_height + bar_spacing)),
            ('SMILES_IPTM_mean', smiles_color, 'Ligand SMILES', -0.5 * (bar_height + bar_spacing))
        ]
        
        legend_handles_ptm = []
        legend_handles_iptm = []
        
        for col_name, color, label, offset in ptm_metrics:
            if col_name not in df.columns or df[col_name].isna().all():
                continue
            std_col = col_name.replace('_mean', '_std')
            has_std = std_col in df.columns
            xerr = df[std_col].fillna(0).values if has_std else None
            bars = ax_ptm.barh(
                y_positions + offset, df[col_name].fillna(0).values, height=bar_height,
                color=color, edgecolor='black', linewidth=PlotConfig.EDGE_WIDTH,
                xerr=xerr, error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}, label=label
            )
            legend_handles_ptm.append(bars)
        
        for col_name, color, label, offset in iptm_metrics:
            if col_name not in df.columns or df[col_name].isna().all():
                continue
            std_col = col_name.replace('_mean', '_std')
            has_std = std_col in df.columns
            xerr = df[std_col].fillna(0).values if has_std else None
            bars = ax_iptm.barh(
                y_positions + offset, df[col_name].fillna(0).values, height=bar_height,
                color=color, edgecolor='black', linewidth=PlotConfig.EDGE_WIDTH,
                xerr=xerr, error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}, label=label
            )
            legend_handles_iptm.append(bars)
        
        if add_threshold:
            if ptm_threshold_value is not None:
                threshold_line_ptm = ax_ptm.axvline(x=ptm_threshold_value, color='gray', linestyle='--', alpha=0.7, linewidth=1.0, label='Threshold')
                legend_handles_ptm.append(threshold_line_ptm)
            if iptm_threshold_value is not None:
                threshold_line_iptm = ax_iptm.axvline(x=iptm_threshold_value, color='gray', linestyle='--', alpha=0.7, linewidth=1.0, label='Threshold')
                legend_handles_iptm.append(threshold_line_iptm)
        
        ax_ptm.set_xlabel('pTM', fontsize=PTM_AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax_iptm.set_xlabel('ipTM', fontsize=PTM_AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax_ptm.invert_yaxis()
        ax_iptm.invert_yaxis()
        ax_ptm.set_ylabel('PDB Identifier', fontsize=PTM_AXIS_LABEL_FONTSIZE, fontweight='bold')
        if not show_y_labels_on_all:
            ax_iptm.set_ylabel('')
        
        ax_ptm.set_yticks(y_positions)
        ax_ptm.set_yticklabels(pdb_labels, fontsize=PTM_TICK_LABEL_FONTSIZE)
        ax_ptm.tick_params(axis='x', labelsize=PTM_TICK_LABEL_FONTSIZE)
        if show_y_labels_on_all:
            ax_iptm.set_yticks(y_positions)
            ax_iptm.set_yticklabels(pdb_labels, fontsize=PTM_TICK_LABEL_FONTSIZE)
            plt.setp(ax_iptm.get_yticklabels(), visible=True)
            ax_iptm.tick_params(labelleft=True)
        ax_iptm.tick_params(axis='x', labelsize=PTM_TICK_LABEL_FONTSIZE)
        ax_iptm.tick_params(axis='y', labelsize=PTM_TICK_LABEL_FONTSIZE)
        
        ax_ptm.set_xlim(0, 1.0)
        ax_iptm.set_xlim(0, 1.0)
        ax_ptm.set_ylim(-0.5, len(df) - 0.5)
        ax_iptm.set_ylim(-0.5, len(df) - 0.5)
        
        combined_handles = legend_handles_ptm + legend_handles_iptm
        final_legend_dict = {}
        for handle in combined_handles:
            label = handle.get_label()
            if label not in final_legend_dict:
                final_legend_dict[label] = handle
        final_handles = list(final_legend_dict.values())
        final_labels = list(final_legend_dict.keys())
        if final_handles:
            ax_iptm.legend(final_handles, final_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0, edgecolor='none', fontsize=PTM_LEGEND_FONTSIZE)
        
        plt.tight_layout()
        
        if save:
            save_title = filename_with_page if filename_with_page else filename_base
            sanitized_filename = save_title.replace('<', 'lt').replace('>', 'gt').replace(' ', '_').replace('-', 'to')
            if filename_with_page and "Page" in filename_with_page: # Ensure paged files are distinct
                base_part = filename_base.replace('<', 'lt').replace('>', 'gt').replace(' ', '_').replace('-', 'to')
                page_info_part = filename_with_page.split("(Page ")[-1].split(")")[0].replace(" ", "_")
                sanitized_filename = f"{base_part}_page_{page_info_part}"
            
            filename = f"ptm_plot_{sanitized_filename}.png" # Added .png extension
            save_figure(fig, filename)
        
        all_figures.append(fig)
        all_axes.append((ax_ptm, ax_iptm))
        return fig, (ax_ptm, ax_iptm)

    def plot_ptm_comparison(self, df, show_individual=True, save=True, 
                            add_threshold=False, ptm_threshold_value=0.8, iptm_threshold_value=0.6):
        """
        Create PTM (pTM and ipTM) comparison plots for different confidence level categories.
        Generates both individual structure plots.
        
        Args:
            df: DataFrame containing the structures to plot
            show_individual: Whether to create plots for each individual category or just the summary
            save: Whether to save the figures
            add_threshold: Whether to add threshold lines
            ptm_threshold_value: Value for threshold line on pTM plot
            iptm_threshold_value: Value for threshold line on ipTM plot
        
        Returns:
            all_figures: List of generated figure objects
            all_axes: List of generated axes objects
        """
        all_figures = []
        all_axes = []
        df = df.copy()
        df['PTM_CATEGORY'] = self._categorize_ptm_values(df)
        categories_to_plot = [
            {'range': (0, 0.8), 'title': 'PTM < 0.8', 'column': 'PTM_CATEGORY'},
            {'range': (0.92, 1.0), 'title': 'PTM > 0.92', 'column': 'PTM_CATEGORY'}
        ]
        plot_counts = {cat['title']: 0 for cat in categories_to_plot}
        
        if show_individual:
            logging.info("Creating individual PTM comparison plots by category")
            for category in categories_to_plot:
                category_df = df[(df[category['column']] >= category['range'][0]) & (df[category['column']] < category['range'][1])]
                if len(category_df) == 0:
                    continue
                plot_counts[category['title']] = len(category_df)
                pages, structures_per_page = distribute_structures_evenly(category_df, PlotConfig.MAX_STRUCTURES_PER_PTM_PLOT if hasattr(PlotConfig, 'MAX_STRUCTURES_PER_PTM_PLOT') else 12)
                for i, page_df in enumerate(pages):
                    page_title_for_filename = f"{category['title']} (Page {i+1} of {len(pages)})"
                    page_plot_height = min(PTM_PAGE_MAX_HEIGHT, PTM_PAGE_HEIGHT_PER_STRUCTURE * len(page_df))
                    page_plot_height = max(min(PTM_PAGE_MAX_HEIGHT, PTM_PAGE_HEIGHT_PER_STRUCTURE), page_plot_height)
                    fig, ax = self._create_single_ptm_plot(
                        page_df, 
                        category['title'], # Base for filename, no visual title
                        "af3", # Defaulting to af3 for plot_ptm_comparison, can be parameterized if needed
                        add_threshold, ptm_threshold_value, iptm_threshold_value,
                        PTM_PAGE_WIDTH, page_plot_height,
                        PTM_BAR_HEIGHT_DATA, PTM_BAR_SPACING_DATA,
                        show_y_labels_on_all=True, save=save,
                        all_figures=all_figures, all_axes=all_axes,
                        filename_with_page=page_title_for_filename
                    )
        return all_figures, all_axes 