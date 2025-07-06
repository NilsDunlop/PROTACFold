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
    
    # Plot dimensions
    FIG_WIDTH = 8.0
    INITIAL_MAX_HEIGHT = 14.0
    PAGE_MAX_HEIGHT = 8.0
    PAGE_HEIGHT_PER_STRUCTURE = 0.6
    
    # Bar styling
    BAR_HEIGHT = 0.30
    BAR_SPACING = 0.02
    
    # Font sizes
    AXIS_LABEL_FONTSIZE = 12
    TICK_LABEL_FONTSIZE = 10
    LEGEND_FONTSIZE = 10
    
    # Default thresholds
    DEFAULT_PTM_THRESHOLD = 0.8
    DEFAULT_IPTM_THRESHOLD = 0.6
    
    def __init__(self):
        """Initialize the PTM plotter."""
        super().__init__()

    def _filter_and_prepare_data(self, df_agg, molecule_type):
        """Filter data by molecule type and prepare numeric columns."""
        df_filtered = df_agg[df_agg['TYPE'] == molecule_type].copy()
        
        # Convert to numeric for all metric columns
        metric_columns = ['CCD_PTM_mean', 'SMILES_PTM_mean', 'CCD_IPTM_mean', 'SMILES_IPTM_mean']
        for col in metric_columns:
            if col in df_filtered.columns:
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        
        # Convert std columns
        for col in df_filtered.columns:
            if col.endswith('_std'):
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        
        return df_filtered

    def _sort_data(self, df_filtered):
        """Sort data by CCD_PTM_mean or fallback to RELEASE_DATE."""
        if 'CCD_PTM_mean' in df_filtered.columns:
            return df_filtered.sort_values('CCD_PTM_mean', ascending=True)
        else:
            logging.warning("'CCD_PTM_mean' column not found. Cannot sort by it.")
            if 'RELEASE_DATE' in df_filtered.columns:
                df_filtered['RELEASE_DATE'] = pd.to_datetime(df_filtered['RELEASE_DATE'])
                return df_filtered.sort_values('RELEASE_DATE', ascending=False)
            return df_filtered

    def _get_plot_title_base(self, data_source, molecule_type, df_filtered):
        """Generate base title for plot filename."""
        title_base = f"{data_source}_{molecule_type.replace(' ', '_').lower()}_ptm_iptm"
        
        if 'CCD_PTM_mean' in df_filtered.columns:
            title_base += "_sorted_by_ccd_ptm"
        elif 'RELEASE_DATE' in df_filtered.columns:
            title_base += "_sorted_by_date"
        
        return title_base

    def _get_colors_for_data_source(self, data_source):
        """Get appropriate colors based on data source."""
        if data_source == "boltz1":
            return PlotConfig.BOLTZ1_CCD_COLOR, PlotConfig.BOLTZ1_SMILES_COLOR
        else:
            return PlotConfig.CCD_PRIMARY, PlotConfig.SMILES_PRIMARY

    def _generate_pdb_labels(self, df):
        """Generate PDB labels with date indicators."""
        if 'RELEASE_DATE' in df.columns:
            df['RELEASE_DATE'] = pd.to_datetime(df['RELEASE_DATE'], errors='coerce')
        
        if 'RELEASE_DATE' in df.columns and 'PDB_ID' in df.columns:
            return [f"{pdb}*" if pd.notna(date) and date > PlotConfig.AF3_CUTOFF else pdb 
                   for pdb, date in zip(df['PDB_ID'], df['RELEASE_DATE'])]
        elif 'PDB_ID' in df.columns:
            return df['PDB_ID'].tolist()
        else:
            return [f"Struct {i+1}" for i in range(len(df))]

    def _plot_metric_bars(self, ax, df, metric_configs, y_positions, bar_height):
        """Plot bars for a specific metric (pTM or ipTM)."""
        legend_handles = []
        
        for col_name, color, label, offset in metric_configs:
            if col_name not in df.columns or df[col_name].isna().all():
                continue
                
            std_col = col_name.replace('_mean', '_std')
            xerr = df[std_col].fillna(0).values if std_col in df.columns else None
            
            bars = ax.barh(
                y_positions + offset, df[col_name].fillna(0).values, height=bar_height,
                color=color, edgecolor='black', linewidth=PlotConfig.EDGE_WIDTH,
                xerr=xerr, error_kw={'ecolor': 'black', 'capsize': 3, 'capthick': 1}, 
                label=label
            )
            legend_handles.append(bars)
        
        return legend_handles

    def _add_threshold_lines(self, ax_ptm, ax_iptm, add_threshold, ptm_threshold, iptm_threshold):
        """Add threshold lines to plots if requested."""
        legend_handles = []
        
        if add_threshold:
            if ptm_threshold is not None:
                threshold_line_ptm = ax_ptm.axvline(
                    x=ptm_threshold, color='gray', linestyle='--', alpha=0.7, 
                    linewidth=1.0, label='Threshold'
                )
                legend_handles.append(threshold_line_ptm)
            
            if iptm_threshold is not None:
                threshold_line_iptm = ax_iptm.axvline(
                    x=iptm_threshold, color='gray', linestyle='--', alpha=0.7, 
                    linewidth=1.0, label='Threshold'
                )
                legend_handles.append(threshold_line_iptm)
        
        return legend_handles

    def _setup_axes(self, ax_ptm, ax_iptm, y_positions, pdb_labels, show_y_labels_on_all):
        """Configure axis labels, ticks, and limits."""
        ax_ptm.set_xlabel('pTM', fontsize=self.AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax_iptm.set_xlabel('ipTM', fontsize=self.AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax_ptm.set_ylabel('PDB Identifier', fontsize=self.AXIS_LABEL_FONTSIZE, fontweight='bold')
        
        if not show_y_labels_on_all:
            ax_iptm.set_ylabel('')
        
        ax_ptm.invert_yaxis()
        ax_iptm.invert_yaxis()
        
        ax_ptm.set_yticks(y_positions)
        ax_ptm.set_yticklabels(pdb_labels, fontsize=self.TICK_LABEL_FONTSIZE)
        ax_ptm.tick_params(axis='x', labelsize=self.TICK_LABEL_FONTSIZE)
        
        if show_y_labels_on_all:
            ax_iptm.set_yticks(y_positions)
            ax_iptm.set_yticklabels(pdb_labels, fontsize=self.TICK_LABEL_FONTSIZE)
            plt.setp(ax_iptm.get_yticklabels(), visible=True)
            ax_iptm.tick_params(labelleft=True)
        
        ax_iptm.tick_params(axis='x', labelsize=self.TICK_LABEL_FONTSIZE)
        ax_iptm.tick_params(axis='y', labelsize=self.TICK_LABEL_FONTSIZE)
        
        # Set axis limits
        ax_ptm.set_xlim(0, 1.0)
        ax_iptm.set_xlim(0, 1.0)
        ax_ptm.set_ylim(-0.5, len(y_positions) - 0.5)
        ax_iptm.set_ylim(-0.5, len(y_positions) - 0.5)

    def _create_combined_legend(self, ax_iptm, ptm_handles, iptm_handles, threshold_handles):
        """Create combined legend for both plots."""
        all_handles = ptm_handles + iptm_handles + threshold_handles
        
        # Remove duplicates while preserving order
        legend_dict = {}
        for handle in all_handles:
            label = handle.get_label()
            if label not in legend_dict:
                legend_dict[label] = handle
        
        handles = list(legend_dict.values())
        labels = list(legend_dict.keys())
        
        if handles:
            ax_iptm.legend(
                handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                framealpha=0, edgecolor='none', fontsize=self.LEGEND_FONTSIZE
            )

    def _generate_save_filename(self, filename_base, filename_with_page=None):
        """Generate appropriate filename for saving."""
        if filename_with_page:
            if "Page" in filename_with_page:
                base_part = filename_base.replace('<', 'lt').replace('>', 'gt').replace(' ', '_').replace('-', 'to')
                page_info = filename_with_page.split("(Page ")[-1].split(")")[0].replace(" ", "_")
                return f"ptm_plot_{base_part}_page_{page_info}.png"
            else:
                save_title = filename_with_page
        else:
            save_title = filename_base
        
        sanitized = save_title.replace('<', 'lt').replace('>', 'gt').replace(' ', '_').replace('-', 'to')
        return f"ptm_plot_{sanitized}.png"

    def plot_ptm_bars(self, df_agg, molecule_type="PROTAC", data_source="af3",
                     add_threshold=False, ptm_threshold_value=0.5, iptm_threshold_value=0.6,
                     show_y_labels_on_all=True, width=None, height=None, 
                     bar_height=None, bar_spacing=None, save=False, 
                     max_structures_per_plot=17):
        """
        Create horizontal bar plots comparing pTM and ipTM metrics for SMILES and CCD.
        
        Args:
            df_agg: Aggregated DataFrame with mean and std values
            molecule_type: Type of molecule to filter by
            data_source: Source of the data for color and filename selection
            add_threshold: Whether to add threshold lines
            ptm_threshold_value: Value for the pTM threshold line
            iptm_threshold_value: Value for the ipTM threshold line
            show_y_labels_on_all: Whether to show y-labels on all subplots
            width, height: Figure dimensions (uses defaults if None)
            bar_height, bar_spacing: Bar dimensions (uses defaults if None)
            save: Whether to save the figure
            max_structures_per_plot: Maximum number of structures per plot page
            
        Returns:
            Lists of created figures and axes
        """
        # Use defaults if not provided
        width = width or self.FIG_WIDTH
        height = height or self.INITIAL_MAX_HEIGHT
        bar_height = bar_height or self.BAR_HEIGHT
        bar_spacing = bar_spacing or self.BAR_SPACING
        
        df_filtered = self._filter_and_prepare_data(df_agg, molecule_type)
        df_filtered = self._sort_data(df_filtered)
        
        if len(df_filtered) == 0:
            logging.warning(f"No data to plot for {data_source} {molecule_type}")
            return [], []
        
        plot_title_base = self._get_plot_title_base(data_source, molecule_type, df_filtered)
        
        all_figures = []
        all_axes = []
        
        self._create_ptm_plots(
            df_filtered, plot_title_base, data_source, 
            add_threshold, ptm_threshold_value, iptm_threshold_value, 
            width, height, bar_height, bar_spacing,
            show_y_labels_on_all, save, max_structures_per_plot, 
            all_figures, all_axes
        )
        
        return all_figures, all_axes

    def _create_ptm_plots(self, df, plot_title_base, data_source,
                         add_threshold, ptm_threshold_value, iptm_threshold_value, 
                         width, height, bar_height, bar_spacing,
                         show_y_labels_on_all, save, max_structures_per_plot, 
                         all_figures, all_axes):
        """Create pTM and ipTM plots, paginating if necessary."""
        if len(df) <= max_structures_per_plot:
            plot_width, plot_height = self.calculate_plot_dimensions(len(df), width)
            self._create_single_ptm_plot(
                df, plot_title_base, data_source, 
                add_threshold, ptm_threshold_value, iptm_threshold_value, 
                plot_width, plot_height, bar_height, bar_spacing,
                show_y_labels_on_all, save, all_figures, all_axes
            )
            return
        
        pages, structures_per_page = distribute_structures_evenly(df, max_structures_per_plot)
        for i, page_df in enumerate(pages):
            plot_width, plot_height = self.calculate_plot_dimensions(len(page_df), width)
            page_filename = f"{plot_title_base} (Page {i+1} of {len(pages)})"
            self._create_single_ptm_plot(
                page_df, plot_title_base, data_source, 
                add_threshold, ptm_threshold_value, iptm_threshold_value, 
                plot_width, plot_height, bar_height, bar_spacing,
                show_y_labels_on_all, save, all_figures, all_axes,
                filename_with_page=page_filename
            )

    def _create_single_ptm_plot(self, df, filename_base, data_source, 
                               add_threshold, ptm_threshold_value, iptm_threshold_value, 
                               width, height, bar_height, bar_spacing,
                               show_y_labels_on_all, save, all_figures, all_axes,
                               filename_with_page=None):
        """Create a single plot with two panels (pTM and ipTM)."""
        if len(df) == 0:
            return
        
        pdb_labels = self._generate_pdb_labels(df)
        fig, (ax_ptm, ax_iptm) = plt.subplots(1, 2, figsize=(width, height), sharey=True)
        y_positions = np.arange(len(df))
        
        ccd_color, smiles_color = self._get_colors_for_data_source(data_source)
        
        # Define metric configurations
        ptm_metrics = [
            ('CCD_PTM_mean', ccd_color, 'Ligand CCD', 0.5 * (bar_height + bar_spacing)),
            ('SMILES_PTM_mean', smiles_color, 'Ligand SMILES', -0.5 * (bar_height + bar_spacing))
        ]
        iptm_metrics = [
            ('CCD_IPTM_mean', ccd_color, 'Ligand CCD', 0.5 * (bar_height + bar_spacing)),
            ('SMILES_IPTM_mean', smiles_color, 'Ligand SMILES', -0.5 * (bar_height + bar_spacing))
        ]
        
        # Plot bars
        ptm_handles = self._plot_metric_bars(ax_ptm, df, ptm_metrics, y_positions, bar_height)
        iptm_handles = self._plot_metric_bars(ax_iptm, df, iptm_metrics, y_positions, bar_height)
        
        # Add threshold lines
        threshold_handles = self._add_threshold_lines(
            ax_ptm, ax_iptm, add_threshold, ptm_threshold_value, iptm_threshold_value
        )
        
        # Setup axes
        self._setup_axes(ax_ptm, ax_iptm, y_positions, pdb_labels, show_y_labels_on_all)
        
        # Create legend
        self._create_combined_legend(ax_iptm, ptm_handles, iptm_handles, threshold_handles)
        
        plt.tight_layout()
        
        if save:
            filename = self._generate_save_filename(filename_base, filename_with_page)
            save_figure(fig, filename)
        
        all_figures.append(fig)
        all_axes.append((ax_ptm, ax_iptm))
        return fig, (ax_ptm, ax_iptm)

    def _categorize_ptm_values(self, df):
        """Categorize PTM values for comparison plots."""
        # This is a placeholder - implement based on your categorization logic
        if 'CCD_PTM_mean' in df.columns:
            return df['CCD_PTM_mean'].fillna(0)
        return pd.Series([0] * len(df))

    def plot_ptm_comparison(self, df, show_individual=True, save=True, 
                           add_threshold=False, ptm_threshold_value=None, iptm_threshold_value=None):
        """
        Create PTM comparison plots for different confidence level categories.
        
        Args:
            df: DataFrame containing the structures to plot
            show_individual: Whether to create plots for each individual category
            save: Whether to save the figures
            add_threshold: Whether to add threshold lines
            ptm_threshold_value: Value for threshold line on pTM plot
            iptm_threshold_value: Value for threshold line on ipTM plot
        
        Returns:
            Lists of generated figures and axes
        """
        ptm_threshold_value = ptm_threshold_value or self.DEFAULT_PTM_THRESHOLD
        iptm_threshold_value = iptm_threshold_value or self.DEFAULT_IPTM_THRESHOLD
        
        all_figures = []
        all_axes = []
        
        if not show_individual:
            return all_figures, all_axes
        
        df = df.copy()
        df['PTM_CATEGORY'] = self._categorize_ptm_values(df)
        
        categories = [
            {'range': (0, 0.8), 'title': 'PTM < 0.8', 'column': 'PTM_CATEGORY'},
            {'range': (0.92, 1.0), 'title': 'PTM > 0.92', 'column': 'PTM_CATEGORY'}
        ]
        
        logging.info("Creating individual PTM comparison plots by category")
        
        for category in categories:
            category_df = df[
                (df[category['column']] >= category['range'][0]) & 
                (df[category['column']] < category['range'][1])
            ]
            
            if len(category_df) == 0:
                continue
            
            max_structures = getattr(PlotConfig, 'MAX_STRUCTURES_PER_PTM_PLOT', 12)
            pages, structures_per_page = distribute_structures_evenly(category_df, max_structures)
            
            for i, page_df in enumerate(pages):
                page_title = f"{category['title']} (Page {i+1} of {len(pages)})"
                page_height = min(
                    self.PAGE_MAX_HEIGHT, 
                    max(self.PAGE_HEIGHT_PER_STRUCTURE, self.PAGE_HEIGHT_PER_STRUCTURE * len(page_df))
                )
                
                self._create_single_ptm_plot(
                    page_df, category['title'], "af3",
                    add_threshold, ptm_threshold_value, iptm_threshold_value,
                    self.FIG_WIDTH, page_height, self.BAR_HEIGHT, self.BAR_SPACING,
                    show_y_labels_on_all=True, save=save,
                    all_figures=all_figures, all_axes=all_axes,
                    filename_with_page=page_title
                )
        
        return all_figures, all_axes 