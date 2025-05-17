import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.lines import Line2D

from base_plotter import BasePlotter
from config import PlotConfig
from data_loader import DataLoader
from utils import save_figure, distribute_pdb_ids, categorize_by_cutoffs

class RMSDComplexIsolatedPlotter(BasePlotter):
    """
    Generates plots for RMSD of the complex, POI, and E3 ligase,
    separately for CCD and SMILES inputs, and for AlphaFold3 and Boltz1 models.
    """

    # General Plot dimensions & appearance
    TITLE_FONT_SIZE = 16 # Kept for per-PDB plots 
    AXIS_LABEL_FONT_SIZE = 12 # General axis label size, used by aggregated plot y-axis

    # Specific for Aggregated Plot
    AGGREGATED_PLOT_WIDTH = 4
    AGGREGATED_PLOT_HEIGHT = 4
    AGGREGATED_TICK_LABEL_FONT_SIZE = 11
    AGGREGATED_LEGEND_FONT_SIZE = 9.5
    AGGREGATED_BAR_WIDTH = 0.6
    
    # Specific for Per-PDB Plot
    PER_PDB_PLOT_WIDTH = 5
    # PER_PDB_PLOT_HEIGHT will be dynamic
    PER_PDB_AXIS_LABEL_FONT_SIZE = 13 # New specific axis label size for per-PDB
    PER_PDB_TICK_LABEL_FONT_SIZE = 13
    PER_PDB_LEGEND_FONT_SIZE = 12
    BAR_SPACING_HORIZONTAL = 0 # Space between bars WITHIN a PDB group (Complex, POI, E3)
    BAR_HEIGHT_HORIZONTAL = 0.30 
    PDB_INTER_GROUP_SPACING = 0.2 # Space BETWEEN PDB groups

    MAX_STRUCTURES_PER_HORIZONTAL_PLOT = 20
    DEFAULT_RMSD_THRESHOLD = 4.0

    def __init__(self):
        """Initialize the plotter."""
        super().__init__()

    def _get_rmsd_component_columns_and_colors(self, model_type, input_type):
        """Helper to get column names and colors for RMSD components."""
        if input_type.upper() == 'CCD':
            columns = ('CCD_RMSD', 'CCD_POI_RMSD', 'CCD_E3_RMSD')
            if model_type == 'AlphaFold3':
                colors = (PlotConfig.AF3_CCD_COLOR, PlotConfig.AF3_CCD_COLOR_POI, PlotConfig.AF3_CCD_COLOR_E3)
            else: # Boltz1
                colors = (PlotConfig.BOLTZ1_CCD_COLOR, PlotConfig.BOLTZ1_CCD_COLOR_POI, PlotConfig.BOLTZ1_CCD_COLOR_E3)
        elif input_type.upper() == 'SMILES':
            columns = ('SMILES_RMSD', 'SMILES_POI_RMSD', 'SMILES_E3_RMSD')
            if model_type == 'AlphaFold3':
                colors = (PlotConfig.AF3_SMILES_COLOR, PlotConfig.AF3_SMILES_COLOR_POI, PlotConfig.AF3_SMILES_COLOR_E3)
            else: # Boltz1
                colors = (PlotConfig.BOLTZ1_SMILES_COLOR, PlotConfig.BOLTZ1_SMILES_COLOR_POI, PlotConfig.BOLTZ1_SMILES_COLOR_E3)
        else:
            raise ValueError(f"Invalid input_type: {input_type}")
        
        labels = ('Complex RMSD', 'POI RMSD', 'E3 RMSD')
        return columns, colors, labels

    def plot_aggregated_rmsd_components(self, df, model_type, molecule_type, 
                                        input_type='CCD', add_threshold=True, 
                                        threshold_value=None, save=True):
        """
        Generates a bar plot showing mean RMSD for Complex, POI, and E3.
        """
        if df is None or df.empty:
            print(f"Error: DataFrame is empty. Cannot plot aggregated RMSD for {model_type} {input_type}.")
            return None, None

        # Filter by molecule_type
        molecule_type_col = 'MOLECULE_TYPE' if 'MOLECULE_TYPE' in df.columns else 'TYPE'
        if molecule_type_col in df.columns:
            df_filtered = df[df[molecule_type_col] == molecule_type].copy()
        else:
            df_filtered = df.copy() # No molecule type filtering if column not present
        
        # Filter by model_type
        df_model = df_filtered[df_filtered['MODEL_TYPE'] == model_type].copy()

        if df_model.empty:
            print(f"No data for {model_type}, {molecule_type}, {input_type} in aggregated plot.")
            return None, None

        component_cols, component_colors, component_labels = self._get_rmsd_component_columns_and_colors(model_type, input_type)
        
        means = []
        errors = []
        
        for col in component_cols:
            if col in df_model.columns:
                values = df_model[col].dropna()
                if len(values) > 0:
                    means.append(values.mean())
                    errors.append(values.std() / np.sqrt(len(values)) if len(values) > 1 else 0)
                else:
                    means.append(0)
                    errors.append(0)
            else:
                print(f"Warning: Column {col} not found in DataFrame for aggregated plot.")
                means.append(0)
                errors.append(0)

        fig, ax = plt.subplots(figsize=(self.AGGREGATED_PLOT_WIDTH, self.AGGREGATED_PLOT_HEIGHT))
        
        x_pos = np.arange(len(component_labels))
        
        ax.bar(x_pos, means, yerr=errors, width=self.AGGREGATED_BAR_WIDTH,
               color=component_colors, capsize=5, edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel('RMSD (Å)', fontsize=self.AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_xticks([]) # Remove x-ticks
        ax.set_xticklabels([]) # Remove x-tick labels
        
        # Adjust y-axis limit
        max_val_with_error = max((m + e) for m, e in zip(means, errors)) if means else 0
        current_threshold_value = threshold_value if threshold_value is not None else self.DEFAULT_RMSD_THRESHOLD
        
        legend_handles = []
        # Create legend items for each bar component
        for i, label in enumerate(component_labels):
            legend_handles.append(plt.Rectangle((0,0),1,1, facecolor=component_colors[i], edgecolor='black', linewidth=0.5))

        if add_threshold:
            ax.axhline(y=current_threshold_value, color='gray', linestyle='--', linewidth=1.0) # No label here, add to custom handles
            legend_handles.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=1.0))
            component_labels_for_legend = list(component_labels) + [f'Threshold']
            ax.legend(legend_handles, component_labels_for_legend, loc='best', fontsize=self.AGGREGATED_LEGEND_FONT_SIZE, frameon=False)
            ax.set_ylim(0, 6) # Fixed y-axis range
        else:
            ax.legend(legend_handles, component_labels, loc='best', fontsize=self.AGGREGATED_LEGEND_FONT_SIZE, frameon=False)
            ax.set_ylim(0, 6) # Fixed y-axis range

        # Apply specific tick label size for y-axis
        ax.tick_params(axis='y', labelsize=self.AGGREGATED_TICK_LABEL_FONT_SIZE)

        ax.grid(axis='y', linestyle='--', alpha=0.2)
        plt.tight_layout() # Removed rect to allow tight_layout to manage spacing without suptitle

        if save:
            filename = f"{model_type.lower()}_{input_type.lower()}_{molecule_type.lower().replace(' ', '_')}_agg_rmsd_components.png"
            save_figure(fig, filename)
            
        return fig, ax

    def plot_per_pdb_rmsd_components(self, df, model_type, molecule_type, 
                                     input_type='CCD', add_threshold=True, 
                                     threshold_value=None, save=True):
        """
        Generates horizontal bar plots for Complex, POI, and E3 RMSD per PDB ID.
        """
        if df is None or df.empty:
            print(f"Error: DataFrame is empty. Cannot plot per-PDB RMSD for {model_type} {input_type}.")
            return [], []

        # Filter by molecule_type
        molecule_type_col = 'MOLECULE_TYPE' if 'MOLECULE_TYPE' in df.columns else 'TYPE'
        if molecule_type_col in df.columns:
            df_filtered = df[df[molecule_type_col] == molecule_type].copy()
        else:
            df_filtered = df.copy()
        
        # Filter by model_type
        df_model = df_filtered[df_filtered['MODEL_TYPE'] == model_type].copy()

        if df_model.empty:
            print(f"No data for {model_type}, {molecule_type}, {input_type} in per-PDB plot.")
            return [], []
        
        # Sort by release date - this should apply to the whole df_model before categorization
        if 'RELEASE_DATE' in df_model.columns:
            df_model['RELEASE_DATE'] = pd.to_datetime(df_model['RELEASE_DATE'])
            # df_model = df_model.sort_values('RELEASE_DATE', ascending=True) # Initial sort by date can be removed or done before merge if preferred
        
        component_cols, component_colors, component_labels = self._get_rmsd_component_columns_and_colors(model_type, input_type)

        # --- Categorization Logic --- 
        categorization_metric_col = component_cols[0] # e.g., CCD_RMSD or SMILES_RMSD
        categories_to_plot = []
        category_column_name_in_df = 'RMSD_Category_Label' # Define what the new category column will be named

        if categorization_metric_col not in df_model.columns:
            print(f"Warning: Categorization column '{categorization_metric_col}' not found for {model_type} {input_type}. Plotting all as 'Uncategorized'.")
            df_model[category_column_name_in_df] = 'Uncategorized'
            categories_to_plot = ['Uncategorized']
        else:
            # Ensure one value per PDB for categorization_metric_col if multiple seeds exist
            pdb_cat_values_df = df_model.groupby('PDB_ID', observed=True)[categorization_metric_col].mean().reset_index()
            pdb_cat_values_for_percentiles = pdb_cat_values_df[categorization_metric_col].dropna()

            if len(pdb_cat_values_for_percentiles) > 4: # Need enough data points for percentiles
                classification_cutoff = [
                    np.percentile(pdb_cat_values_for_percentiles, 20),
                    np.percentile(pdb_cat_values_for_percentiles, 40),
                    np.percentile(pdb_cat_values_for_percentiles, 60),
                    np.percentile(pdb_cat_values_for_percentiles, 80)
                ]
                category_labels_list = [f"< {classification_cutoff[0]:.1f} Å"] + \
                                   [f"{classification_cutoff[i]:.1f} - {classification_cutoff[i+1]:.1f} Å" for i in range(len(classification_cutoff)-1)] + \
                                   [f"> {classification_cutoff[-1]:.1f} Å"]
            else: # Fallback if not enough data for robust percentiles
                print(f"Warning: Not enough unique PDBs with {categorization_metric_col} data for percentile-based categorization. Using broader categories.")
                # Simplified cutoffs if too few PDBs with the metric
                median_val = pdb_cat_values_for_percentiles.median() if not pdb_cat_values_for_percentiles.empty else 4.0
                classification_cutoff = [median_val / 2, median_val, median_val * 1.5, median_val * 2]
                category_labels_list = [f"< {classification_cutoff[0]:.1f} Å", 
                                   f"{classification_cutoff[0]:.1f} - {classification_cutoff[1]:.1f} Å",
                                   f"{classification_cutoff[1]:.1f} - {classification_cutoff[2]:.1f} Å",
                                   f"{classification_cutoff[2]:.1f} - {classification_cutoff[3]:.1f} Å",
                                   f"> {classification_cutoff[3]:.1f} Å"]
            
            # Merge these PDB-level mean categorization values back to the main df_model
            temp_cat_col_name = f"__{categorization_metric_col}_for_cat_mean"
            df_model = pd.merge(df_model, pdb_cat_values_df.rename(columns={categorization_metric_col: temp_cat_col_name}), on='PDB_ID', how='left')

            # Use categorize_by_cutoffs (assuming it's updated or we use pd.cut manually)
            # For now, using pd.cut directly as in thought process for precise label control
            df_model[category_column_name_in_df] = pd.cut(
                df_model[temp_cat_col_name],
                bins=[-np.inf] + classification_cutoff + [np.inf],
                labels=category_labels_list,
                right=False # Intervals are [a, b)
            ).astype(str).fillna('Undefined') # Ensure string and handle any NaNs from cut

            df_model = df_model.drop(columns=[temp_cat_col_name]) # Clean up temporary column
            # Ensure categories_to_plot respects the order of category_labels_list
            categories_to_plot = [label for label in category_labels_list if label in df_model[category_column_name_in_df].unique()]
            if 'Undefined' in df_model[category_column_name_in_df].unique() and 'Undefined' not in categories_to_plot:
                 categories_to_plot.append('Undefined') # Add if it exists
        
        all_figs = []
        all_axes = []

        if not categories_to_plot:
            print(f"No categories generated to plot for {model_type} {input_type}.")
            return [], []

        for category_label_text in categories_to_plot:
            df_category_filtered = df_model[df_model[category_column_name_in_df] == category_label_text].copy() # Use .copy()
            
            # --- Sorting within category by RMSD (categorization_metric_col PDB-level mean) ---
            # Ensure the PDB-level mean used for categorization is available for sorting
            # Re-calculate or ensure temp_cat_col_name_for_sort exists on df_category_filtered if needed.
            # For simplicity, let's assume the PDB-level mean was already added as temp_cat_col_name earlier
            # and we can sort based on that, or re-merge it just for sorting this subset.

            # To sort by the PDB-level mean of the categorization metric:
            # 1. Calculate PDB-level mean for the current category_filtered_df
            pdb_level_means_for_sort = df_category_filtered.groupby('PDB_ID', observed=True)[categorization_metric_col].mean().reset_index()
            pdb_level_means_for_sort = pdb_level_means_for_sort.rename(columns={categorization_metric_col: '__sort_metric_pdb_mean'})
            
            # 2. Merge these means back to df_category_filtered for sorting
            df_category_filtered_sorted = pd.merge(df_category_filtered, pdb_level_means_for_sort, on='PDB_ID', how='left')
            
            # 3. Sort: Ascending order of the metric, so when y-axis is inverted, highest RMSD is at top.
            #    Also, use PDB_ID as a secondary sort key for stable sorting if RMSD values are identical.
            #    NaNs (if any from merge or original data) should be handled (e.g. to bottom or top).
            df_category_filtered_sorted = df_category_filtered_sorted.sort_values(
                by=['__sort_metric_pdb_mean', 'PDB_ID'], 
                ascending=[False, True], # Corrected: False for __sort_metric_pdb_mean (descending)
                na_position='last' # Put PDBs with no sort metric at the bottom
            )
            
            category_pdb_ids = df_category_filtered_sorted['PDB_ID'].unique()
            # --- End Sorting --- 

            if len(category_pdb_ids) == 0:
                continue
            
            paginated_pdb_ids_for_category = distribute_pdb_ids(category_pdb_ids, self.MAX_STRUCTURES_PER_HORIZONTAL_PLOT)

            for page_num, page_pdb_ids_list in enumerate(paginated_pdb_ids_for_category, 1):
                page_df = df_category_filtered_sorted[df_category_filtered_sorted['PDB_ID'].isin(page_pdb_ids_list)].copy() # Use sorted df
                page_df['PDB_ID'] = pd.Categorical(page_df['PDB_ID'], categories=page_pdb_ids_list, ordered=True)
                page_df = page_df.sort_values('PDB_ID')
                
                # Check if any essential component columns are missing for all PDBs on this page
                all_cols_present_for_all_pdbs = True
                for col in component_cols:
                    if col not in page_df.columns:
                         all_cols_present_for_all_pdbs = False
                         print(f"Warning: Column {col} not found for per-PDB plot page {page_num}.")
                         break
                if not all_cols_present_for_all_pdbs:
                    continue # Skip this page if essential columns are missing

                # Group by PDB_ID and take the mean if multiple entries exist (e.g. different seeds)
                # Keep RELEASE_DATE for labelling by taking the first
                agg_operations = {'mean', 'std', 'count'}
                agg_dict = {}
                for col in component_cols:
                    if col in page_df.columns:
                        for op in agg_operations:
                            agg_dict[f"{col}_{op}"] = (col, op)
                
                if 'RELEASE_DATE' in page_df.columns:
                     agg_dict['RELEASE_DATE'] = ('RELEASE_DATE', 'first') # Keep original name for simplicity
                
                if not any(op == 'mean' for key_op in agg_dict for op_tuple in [agg_dict[key_op]] for op in [op_tuple[1]]): # Check if any mean op exists
                    print(f"Warning: No valid RMSD component columns found for PDBs on page {page_num} to calculate mean.")
                    continue

                page_data_agg = page_df.groupby('PDB_ID', as_index=False, observed=True).agg(**agg_dict)
                # Flatten multi-index columns if pandas version creates them, or adjust naming
                new_cols = {}
                for old_col_tuple in page_data_agg.columns:
                    if isinstance(old_col_tuple, tuple):
                        if old_col_tuple[1] == '' or old_col_tuple[1] == 'first': # Handle RELEASE_DATE and PDB_ID
                            new_cols[old_col_tuple] = old_col_tuple[0]
                        else:
                            new_cols[old_col_tuple] = f"{old_col_tuple[0]}_{old_col_tuple[1]}"
                    else: # PDB_ID is not a tuple
                        new_cols[old_col_tuple] = old_col_tuple
                page_data_agg.columns = page_data_agg.columns.map(new_cols)

                # Reorder page_data_agg to match page_pdb_ids_list
                page_data_agg['PDB_ID'] = pd.Categorical(page_data_agg['PDB_ID'], categories=page_pdb_ids_list, ordered=True)
                page_data_agg = page_data_agg.sort_values('PDB_ID').reset_index(drop=True)


                if page_data_agg.empty:
                    continue

                n_pdbs_on_page = len(page_data_agg)
                
                # Dynamic height calculation considering inter-group spacing
                num_components = len(component_cols)
                height_of_one_pdb_bar_group = num_components * self.BAR_HEIGHT_HORIZONTAL + (num_components - 1) * self.BAR_SPACING_HORIZONTAL
                total_bar_group_height_all_pdbs = n_pdbs_on_page * height_of_one_pdb_bar_group
                total_inter_group_spacing = (n_pdbs_on_page - 1) * self.PDB_INTER_GROUP_SPACING if n_pdbs_on_page > 0 else 0
                plot_height = max(5, 1.5 + total_bar_group_height_all_pdbs + total_inter_group_spacing) # 1.5 for margins

                fig, ax = plt.subplots(figsize=(self.PER_PDB_PLOT_WIDTH, plot_height))

                y_ticks_pdb_labels = []
                if 'RELEASE_DATE' in page_data_agg.columns:
                     y_ticks_pdb_labels = [f"{pdb}*" if date > PlotConfig.AF3_CUTOFF else pdb 
                                       for pdb, date in zip(page_data_agg['PDB_ID'], page_data_agg['RELEASE_DATE'])]
                else:
                     y_ticks_pdb_labels = page_data_agg['PDB_ID'].tolist()
                
                # Calculate y_pos_base considering PDB_INTER_GROUP_SPACING
                # Each step includes the height of a PDB bar group and the inter-group spacing
                step_per_pdb = height_of_one_pdb_bar_group + self.PDB_INTER_GROUP_SPACING
                y_pos_base = np.arange(n_pdbs_on_page) * step_per_pdb 
                # y_pos_base needs to respect the true order for plotting, which PDB_ID (Categorical) ensures if page_df is sorted by it.
                # The PDB labels (y_ticks_pdb_labels) should be generated from page_data_agg in its final PDB_ID sorted order.

                max_rmsd_on_page = 0

                for i, (col, color, label) in enumerate(zip(component_cols, component_colors, component_labels)):
                    # y_offsets are relative to y_pos_base, using BAR_SPACING_HORIZONTAL for within-group spacing
                    y_offsets = y_pos_base + i * (self.BAR_HEIGHT_HORIZONTAL + self.BAR_SPACING_HORIZONTAL)
                    
                    mean_col_name = f"{col}_mean"
                    std_col_name = f"{col}_std"
                    count_col_name = f"{col}_count"
                    
                    values = page_data_agg[mean_col_name].fillna(0).values if mean_col_name in page_data_agg else np.zeros(n_pdbs_on_page)
                    stds = page_data_agg[std_col_name].fillna(0).values if std_col_name in page_data_agg else np.zeros(n_pdbs_on_page)
                    counts = page_data_agg[count_col_name].fillna(1).values if count_col_name in page_data_agg else np.ones(n_pdbs_on_page)
                    
                    # Calculate standard error: std / sqrt(count). Ensure counts are not zero.
                    # np.errstate handles division by zero if any count is 0 (though fillna(1) avoids this for count).
                    with np.errstate(divide='ignore', invalid='ignore'):
                        x_errors = stds / np.sqrt(counts)
                    x_errors = np.nan_to_num(x_errors) # Replace NaN (e.g. from count=1 -> std=NaN) with 0

                    ax.barh(y_offsets, values, height=self.BAR_HEIGHT_HORIZONTAL, xerr=x_errors,
                            facecolor=color, edgecolor='black', linewidth=0.5, label=label if page_num ==1 and i==0 else None, 
                            error_kw={'ecolor': 'black', 'capsize': 2, 'alpha':0.7})
                    
                    current_max = np.nanmax(values + x_errors) if len(values) > 0 else 0 # Consider error for max value
                    if current_max > max_rmsd_on_page:
                        max_rmsd_on_page = current_max
                
                ax.set_xlabel(f'RMSD (Å)', fontsize=self.PER_PDB_AXIS_LABEL_FONT_SIZE, fontweight='bold')
                ax.set_ylabel('PDB Identifier', fontsize=self.PER_PDB_AXIS_LABEL_FONT_SIZE, fontweight='bold')
                # Center ticks within each PDB bar group
                tick_positions = y_pos_base + (height_of_one_pdb_bar_group - self.BAR_HEIGHT_HORIZONTAL) / 2.0
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(y_ticks_pdb_labels, fontsize=self.PER_PDB_TICK_LABEL_FONT_SIZE)
                ax.invert_yaxis() # PDBs sorted by custom criteria, this makes 0-index (highest RMSD) appear at top

                # Legend
                handles, labels_legend = [], []
                for i, (col, color, label_text) in enumerate(zip(component_cols, component_colors, component_labels)):
                     handles.append(plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', linewidth=0.5))
                     labels_legend.append(label_text)

                current_threshold_value = threshold_value if threshold_value is not None else self.DEFAULT_RMSD_THRESHOLD
                if add_threshold:
                    ax.axvline(x=current_threshold_value, color='gray', linestyle='--', linewidth=1.0)
                    handles.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=1.0))
                    labels_legend.append(f'Threshold')
                    ax.set_xlim(0, max(max_rmsd_on_page * 1.1, current_threshold_value * 1.1, 1.0))
                else:
                    ax.set_xlim(0, max(max_rmsd_on_page * 1.1, 1.0))

                ax.legend(handles, labels_legend, loc='best', fontsize=self.PER_PDB_LEGEND_FONT_SIZE, frameon=False)
                
                page_info = f" (Page {page_num} of {len(paginated_pdb_ids_for_category)})" if len(paginated_pdb_ids_for_category) > 1 else ""
                
                # Simplified title construction
                if category_label_text == 'Uncategorized' or category_label_text == 'Undefined':
                    plot_title = f'{category_label_text}{page_info}'
                else:
                    # Remove " Å" suffix from category_label_text for cleaner title if present
                    cleaned_label_text = category_label_text.replace(" Å", "")
                    plot_title = f'RMSD: {cleaned_label_text}{page_info}'
                
                fig.suptitle(plot_title, fontsize=self.TITLE_FONT_SIZE) # Removed fontweight='bold'
                
                ax.grid(axis='x', linestyle='--', alpha=0.2)
                plt.tight_layout() # Adjusted top of rect from 0.94 to 0.97

                if save:
                    page_suffix = f"_page{page_num}" if len(paginated_pdb_ids_for_category) > 1 else ""
                    category_filename_part = f"_cat_{category_label_text.replace(' ', '_').replace('<', 'lt').replace('>', 'gt').replace('-', 'to').replace('.', 'p').replace('Å','A')}" if category_label_text != 'Uncategorized' else ""
                    filename = f"{model_type.lower()}_{input_type.lower()}_{molecule_type.lower().replace(' ', '_')}_perpdb_rmsd_components{category_filename_part}{page_suffix}.png"
                    save_figure(fig, filename)

                all_figs.append(fig)
                all_axes.append(ax)
                
        return all_figs, all_axes 