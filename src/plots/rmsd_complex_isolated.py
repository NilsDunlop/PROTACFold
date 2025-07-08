import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from base_plotter import BasePlotter
from config import PlotConfig
from utils import save_figure, distribute_pdb_ids, categorize_by_cutoffs, create_plot_filename

class RMSDComplexIsolatedPlotter(BasePlotter):
    """Generates plots for RMSD of complex, POI, and E3 ligase components."""

    # Constants now imported from PlotConfig

    def __init__(self):
        """Initialize the plotter."""
        super().__init__()

    def _get_rmsd_component_columns_and_colors(self, model_type, input_type):
        """Get column names and colors for RMSD components."""
        input_type = input_type.upper()
        
        if input_type == 'CCD':
            columns = ('CCD_RMSD', 'CCD_POI_RMSD', 'CCD_E3_RMSD')
            colors = self._get_ccd_colors(model_type)
        elif input_type == 'SMILES':
            columns = ('SMILES_RMSD', 'SMILES_POI_RMSD', 'SMILES_E3_RMSD')
            colors = self._get_smiles_colors(model_type)
        else:
            raise ValueError(f"Invalid input_type: {input_type}")
        
        labels = ('Complex RMSD', 'POI RMSD', 'E3 RMSD')
        return columns, colors, labels

    def _get_ccd_colors(self, model_type):
        """Get CCD colors based on model type."""
        if model_type == 'AlphaFold3':
            return (PlotConfig.AF3_CCD_COLOR, PlotConfig.AF3_CCD_COLOR_POI, PlotConfig.AF3_CCD_COLOR_E3)
        else:
            return (PlotConfig.BOLTZ1_CCD_COLOR, PlotConfig.BOLTZ1_CCD_COLOR_POI, PlotConfig.BOLTZ1_CCD_COLOR_E3)

    def _get_smiles_colors(self, model_type):
        """Get SMILES colors based on model type."""
        if model_type == 'AlphaFold3':
            return (PlotConfig.AF3_SMILES_COLOR, PlotConfig.AF3_SMILES_COLOR_POI, PlotConfig.AF3_SMILES_COLOR_E3)
        else:
            return (PlotConfig.BOLTZ1_SMILES_COLOR, PlotConfig.BOLTZ1_SMILES_COLOR_POI, PlotConfig.BOLTZ1_SMILES_COLOR_E3)

    def _filter_data(self, df, model_type, molecule_type):
        """Filter data by model and molecule type."""
        if df is None or df.empty:
            return df
            
        molecule_type_col = 'MOLECULE_TYPE' if 'MOLECULE_TYPE' in df.columns else 'TYPE'
        
        if molecule_type_col in df.columns:
            df_filtered = df[df[molecule_type_col] == molecule_type].copy()
        else:
            df_filtered = df.copy()
        
        return df_filtered[df_filtered['MODEL_TYPE'] == model_type].copy()

    def _calculate_component_stats(self, df, component_cols):
        """Calculate mean and standard error for each component."""
        means, errors = [], []
        
        for col in component_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    means.append(values.mean())
                    errors.append(values.std() / np.sqrt(len(values)) if len(values) > 1 else 0)
                else:
                    means.append(0)
                    errors.append(0)
            else:
                print(f"Warning: Column {col} not found in DataFrame.")
                means.append(0)
                errors.append(0)
        
        return means, errors

    def _setup_aggregated_plot_axes(self, ax, means, errors, component_colors, component_labels, 
                                   add_threshold, threshold_value):
        """Configure axes for aggregated plot."""
        x_pos = np.arange(len(component_labels))
        
        ax.bar(x_pos, means, yerr=errors, width=PlotConfig.RMSD_AGGREGATED_BAR_WIDTH,
               color=component_colors, capsize=PlotConfig.ERROR_BAR_CAPSIZE, 
               edgecolor='black', linewidth=PlotConfig.BAR_EDGE_WIDTH,
               error_kw={'elinewidth': PlotConfig.ERROR_BAR_THICKNESS, 'alpha': PlotConfig.ERROR_BAR_ALPHA})
        
        ax.set_ylabel('RMSD (Å)', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        if add_threshold:
            threshold_val = threshold_value if threshold_value is not None else PlotConfig.DEFAULT_RMSD_THRESHOLD
            ax.axhline(y=threshold_val, color='gray', linestyle='--', 
                      linewidth=PlotConfig.THRESHOLD_LINE_WIDTH, alpha=PlotConfig.THRESHOLD_LINE_ALPHA)
        
        ax.set_ylim(0, 6)
        ax.tick_params(axis='y', labelsize=PlotConfig.TICK_LABEL_SIZE)
        ax.grid(axis='y', linestyle=PlotConfig.GRID_LINESTYLE, alpha=PlotConfig.GRID_ALPHA)

    def plot_aggregated_rmsd_components(self, df, model_type, molecule_type, 
                                        input_type='CCD', add_threshold=True, 
                                        threshold_value=None, classification_cutoff=None, save=True):
        """Generate aggregated bar plot showing mean RMSD for Complex, POI, and E3."""
        if df is None or df.empty:
            print(f"Error: DataFrame is empty. Cannot plot aggregated RMSD for {model_type} {input_type}.")
            return None, None

        df_model = self._filter_data(df, model_type, molecule_type)
        if df_model.empty:
            print(f"No data for {model_type}, {molecule_type}, {input_type} in aggregated plot.")
            return None, None

        component_cols, component_colors, component_labels = self._get_rmsd_component_columns_and_colors(model_type, input_type)
        means, errors = self._calculate_component_stats(df_model, component_cols)

        fig, ax = plt.subplots(figsize=(PlotConfig.RMSD_AGGREGATED_WIDTH, PlotConfig.RMSD_AGGREGATED_HEIGHT))
        self._setup_aggregated_plot_axes(ax, means, errors, component_colors, component_labels, 
                                        add_threshold, threshold_value)
        
        plt.tight_layout()

        if save:
            filename = create_plot_filename(
                'rmsd_agg', model_type=model_type, 
                input_type=input_type, molecule_type=molecule_type
            )
            save_figure(fig, filename)
            
        return fig, ax

    

    def _sort_data_within_category(self, df, sorting_col):
        """Sort data within category by sorting column."""
        if sorting_col not in df.columns:
            print(f"Warning: Sorting column '{sorting_col}' not found. Using default order.")
            return df['PDB_ID'].unique()
        
        pdb_means = df.groupby('PDB_ID', observed=True)[sorting_col].mean().reset_index()
        pdb_means = pdb_means.rename(columns={sorting_col: '__sort_metric_pdb_mean'})
        
        df_sorted = pd.merge(df, pdb_means, on='PDB_ID', how='left')
        df_sorted = df_sorted.sort_values(
            by=['__sort_metric_pdb_mean', 'PDB_ID'], 
            ascending=[False, True],
            na_position='last'
        )
        
        return df_sorted['PDB_ID'].unique(), df_sorted

    def _calculate_plot_height(self, n_pdbs):
        """Calculate dynamic plot height based on number of PDBs."""
        num_bar_groups_per_pdb = 2
        height_per_pdb = num_bar_groups_per_pdb * PlotConfig.RMSD_BAR_HEIGHT_HORIZONTAL + (num_bar_groups_per_pdb - 1) * PlotConfig.RMSD_BAR_SPACING_HORIZONTAL
        total_height = n_pdbs * height_per_pdb
        total_spacing = (n_pdbs - 1) * PlotConfig.RMSD_PDB_INTER_GROUP_SPACING if n_pdbs > 0 else 0
        return max(5, 1.5 + total_height + total_spacing)

    def _aggregate_page_data(self, page_df, component_cols):
        """Aggregate page data by PDB_ID."""
        agg_dict = {}
        for col in component_cols:
            if col in page_df.columns:
                for op in ['mean', 'std', 'count']:
                    agg_dict[f"{col}_{op}"] = (col, op)
        
        if 'RELEASE_DATE' in page_df.columns:
            agg_dict['RELEASE_DATE'] = ('RELEASE_DATE', 'first')
        
        return page_df.groupby('PDB_ID', as_index=False, observed=True).agg(**agg_dict)

    def _generate_pdb_labels(self, page_data, cutoff_date=None):
        """Generate PDB labels with date indicators."""
        if 'RELEASE_DATE' not in page_data.columns:
            return page_data['PDB_ID'].tolist()
        
        cutoff = cutoff_date or PlotConfig.AF3_CUTOFF
        return [f"{pdb}*" if date > cutoff else pdb 
                for pdb, date in zip(page_data['PDB_ID'], page_data['RELEASE_DATE'])]

    def _calculate_bar_positions(self, n_pdbs):
        """Calculate bar positions for plotting."""
        height_per_pdb = 2 * PlotConfig.RMSD_BAR_HEIGHT_HORIZONTAL + PlotConfig.RMSD_BAR_SPACING_HORIZONTAL
        step_per_pdb = height_per_pdb + PlotConfig.RMSD_PDB_INTER_GROUP_SPACING
        y_pos_base = np.arange(n_pdbs) * step_per_pdb
        
        y_pos_complex = y_pos_base
        y_pos_stacked = y_pos_base + (PlotConfig.RMSD_BAR_HEIGHT_HORIZONTAL + PlotConfig.RMSD_BAR_SPACING_HORIZONTAL)
        
        return y_pos_complex, y_pos_stacked, y_pos_base

    def _extract_component_values(self, page_data, component_cols):
        """Extract values for all three components."""
        n_pdbs = len(page_data)
        values = {}
        
        for i, col in enumerate(component_cols):
            component_name = ['complex', 'poi', 'e3'][i]
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"
            count_col = f"{col}_count"
            
            values[f'{component_name}_values'] = page_data[mean_col].fillna(0).values if mean_col in page_data else np.zeros(n_pdbs)
            values[f'{component_name}_stds'] = page_data[std_col].fillna(0).values if std_col in page_data else np.zeros(n_pdbs)
            values[f'{component_name}_counts'] = page_data[count_col].fillna(1).values if count_col in page_data else np.ones(n_pdbs)
        
        return values

    def _calculate_standard_errors(self, values):
        """Calculate standard errors for all components."""
        errors = {}
        for component in ['complex', 'poi', 'e3']:
            stds = values[f'{component}_stds']
            counts = values[f'{component}_counts']
            
            with np.errstate(divide='ignore', invalid='ignore'):
                errors[f'{component}_errors'] = stds / np.sqrt(counts)
            errors[f'{component}_errors'] = np.nan_to_num(errors[f'{component}_errors'])
        
        return errors

    def _plot_rmsd_bars(self, ax, y_pos_complex, y_pos_stacked, values, errors, colors, labels, page_num):
        """Plot the RMSD bars for all components."""
        # Complex RMSD (single bar)
        ax.barh(y_pos_complex, values['complex_values'], height=PlotConfig.RMSD_BAR_HEIGHT_HORIZONTAL, 
                xerr=errors['complex_errors'], facecolor=colors[0], edgecolor='black', linewidth=0.5, 
                label=labels[0] if page_num == 1 else None,
                error_kw={'ecolor': 'black', 'capsize': 2, 'alpha': 0.7})

        # POI RMSD (base of stacked bar)
        ax.barh(y_pos_stacked, values['poi_values'], height=PlotConfig.RMSD_BAR_HEIGHT_HORIZONTAL, 
                xerr=errors['poi_errors'], facecolor=colors[1], edgecolor='black', linewidth=0.5, 
                label=labels[1] if page_num == 1 else None,
                error_kw={'ecolor': 'black', 'capsize': 2, 'alpha': 0.7})

        # E3 RMSD (stacked on POI)
        ax.barh(y_pos_stacked, values['e3_values'], left=values['poi_values'], 
                height=PlotConfig.RMSD_BAR_HEIGHT_HORIZONTAL, xerr=errors['e3_errors'], 
                facecolor=colors[2], edgecolor='black', linewidth=0.5, 
                label=labels[2] if page_num == 1 else None,
                error_kw={'ecolor': 'black', 'capsize': 2, 'alpha': 0.7})

    def _setup_per_pdb_axes(self, ax, y_pos_base, pdb_labels, max_rmsd, add_threshold, threshold_value):
        """Setup axes for per-PDB plot."""
        ax.set_xlabel('RMSD (Å)', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('PDB Identifier', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
        
        height_per_pdb = 2 * PlotConfig.RMSD_BAR_HEIGHT_HORIZONTAL + PlotConfig.RMSD_BAR_SPACING_HORIZONTAL
        tick_positions = y_pos_base + (height_per_pdb - PlotConfig.RMSD_BAR_HEIGHT_HORIZONTAL) / 2.0
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(pdb_labels, fontsize=PlotConfig.TICK_LABEL_SIZE)
        ax.invert_yaxis()

        threshold_val = threshold_value if threshold_value is not None else PlotConfig.DEFAULT_RMSD_THRESHOLD
        if add_threshold:
            ax.axvline(x=threshold_val, color='gray', linestyle='--', linewidth=1.0)
            ax.set_xlim(0, max(max_rmsd * 1.1, threshold_val * 1.1, 1.0))
        else:
            ax.set_xlim(0, max(max_rmsd * 1.1, 1.0))
        
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=2))
        ax.grid(axis='x', linestyle='--', alpha=0.2)

    def _create_per_pdb_legend(self, colors, labels, add_threshold):
        """Create legend for per-PDB plot."""
        handles = []
        legend_labels = []
        
        for color, label in zip(colors, labels):
            handles.append(plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', linewidth=0.5))
            legend_labels.append(label)
        
        if add_threshold:
            handles.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=1.0))
            legend_labels.append('Threshold')
        
        return handles, legend_labels

    def _generate_plot_title(self, category_label, page_num, total_pages):
        """Generate plot title based on category and page info."""
        page_info = f" (Page {page_num} of {total_pages})" if total_pages > 1 else ""
        
        if category_label in ['Uncategorized', 'Undefined']:
            return f'{category_label}{page_info}'
        else:
            cleaned_label = category_label.replace(" Å", "")
            return f'RMSD: {cleaned_label}{page_info}'

    def plot_per_pdb_rmsd_components(self, df, model_type, molecule_type, 
                                     input_type='CCD', add_threshold=True, 
                                     threshold_value=None, classification_cutoff=None, save=True):
        """Generate horizontal bar plots for Complex, POI, and E3 RMSD per PDB ID."""
        if df is None or df.empty:
            print(f"Error: DataFrame is empty. Cannot plot per-PDB RMSD for {model_type} {input_type}.")
            return [], []

        df_model = self._filter_data(df, model_type, molecule_type)
        if df_model.empty:
            print(f"No data for {model_type}, {molecule_type}, {input_type} in per-PDB plot.")
            return [], []
        
        if 'RELEASE_DATE' in df_model.columns:
            df_model['RELEASE_DATE'] = pd.to_datetime(df_model['RELEASE_DATE'])
        
        component_cols, component_colors, component_labels = self._get_rmsd_component_columns_and_colors(model_type, input_type)
        df_model = categorize_by_cutoffs(df_model, component_cols[0], classification_cutoff, 'RMSD_Category_Label')
        categories = df_model['RMSD_Category_Label'].unique().tolist()
        
        all_figs, all_axes = [], []

        for category_label in categories:
            df_category = df_model[df_model['RMSD_Category_Label'] == category_label].copy()
            
            sorting_col = component_cols[0]
            category_pdb_ids, df_category_sorted = self._sort_data_within_category(df_category, sorting_col)
            
            if len(category_pdb_ids) == 0:
                continue
            
            paginated_pdb_ids = distribute_pdb_ids(category_pdb_ids, PlotConfig.MAX_STRUCTURES_PER_HORIZONTAL_PLOT)

            for page_num, page_pdb_ids in enumerate(paginated_pdb_ids, 1):
                page_df = df_category_sorted[df_category_sorted['PDB_ID'].isin(page_pdb_ids)].copy()
                page_df['PDB_ID'] = pd.Categorical(page_df['PDB_ID'], categories=page_pdb_ids, ordered=True)
                page_df = page_df.sort_values('PDB_ID')
                
                if not all(col in page_df.columns for col in component_cols):
                    continue

                page_data = self._aggregate_page_data(page_df, component_cols)
                
                # Handle column naming from aggregation
                new_cols = {}
                for col in page_data.columns:
                    if isinstance(col, tuple):
                        if col[1] in ['', 'first']:
                            new_cols[col] = col[0]
                        else:
                            new_cols[col] = f"{col[0]}_{col[1]}"
                    else:
                        new_cols[col] = col
                page_data.columns = page_data.columns.map(new_cols)

                page_data['PDB_ID'] = pd.Categorical(page_data['PDB_ID'], categories=page_pdb_ids, ordered=True)
                page_data = page_data.sort_values('PDB_ID').reset_index(drop=True)

                if page_data.empty:
                    continue

                n_pdbs = len(page_data)
                plot_height = self._calculate_plot_height(n_pdbs)
                
                fig, ax = plt.subplots(figsize=(PlotConfig.RMSD_PER_PDB_WIDTH, plot_height))
                
                pdb_labels = self._generate_pdb_labels(page_data)
                y_pos_complex, y_pos_stacked, y_pos_base = self._calculate_bar_positions(n_pdbs)
                
                values = self._extract_component_values(page_data, component_cols)
                errors = self._calculate_standard_errors(values)
                
                self._plot_rmsd_bars(ax, y_pos_complex, y_pos_stacked, values, errors, 
                                   component_colors, component_labels, page_num)
                
                max_rmsd = max(
                    np.nanmax(values['complex_values'] + errors['complex_errors']) if len(values['complex_values']) > 0 else 0,
                    np.nanmax((values['poi_values'] + values['e3_values']) + 
                             np.sqrt(errors['poi_errors']**2 + errors['e3_errors']**2)) if len(values['poi_values']) > 0 else 0
                )
                
                self._setup_per_pdb_axes(ax, y_pos_base, pdb_labels, max_rmsd, add_threshold, threshold_value)
                
                handles, legend_labels = self._create_per_pdb_legend(component_colors, component_labels, add_threshold)
                ax.legend(handles, legend_labels, loc='best', fontsize=PlotConfig.LEGEND_TEXT_SIZE, frameon=False)
                
                plot_title = self._generate_plot_title(category_label, page_num, len(paginated_pdb_ids))
                fig.suptitle(plot_title, fontsize=PlotConfig.TITLE_SIZE)
                
                plt.tight_layout(rect=[0, 0, 1, 0.99])

                if save:
                    page_suffix = f"_page{page_num}" if len(paginated_pdb_ids) > 1 else ""
                    category_part = f"_cat_{category_label.replace(' ', '_').replace('<', 'lt').replace('>', 'gt').replace('-', 'to').replace('.', 'p').replace('Å','A')}" if category_label != 'Uncategorized' else ""
                    
                    filename = create_plot_filename(
                        'rmsd_perpdb', model_type=model_type,
                        input_type=input_type, molecule_type=molecule_type,
                        category=category_part, page=page_suffix
                    )
                    save_figure(fig, filename)

                all_figs.append(fig)
                all_axes.append(ax)
                
        return all_figs, all_axes

    def create_vertical_legend(self, model_type, input_type, add_threshold=True, width=2, height=4, save=True, filename=None):
        """Create a standalone vertical legend figure for RMSD complex/isolated plots."""
        fig, ax = plt.subplots(figsize=(width, height))
        
        component_cols, component_colors, component_labels = self._get_rmsd_component_columns_and_colors(model_type, input_type)
        
        legend_handles = []
        for label, color in zip(component_labels, component_colors):
            patch = plt.Rectangle(
                (0, 0), 1, 1,
                facecolor=color,
                edgecolor='black',
                linewidth=PlotConfig.BAR_EDGE_WIDTH,
                label=label
            )
            legend_handles.append(patch)
        
        if add_threshold:
            threshold_line = plt.Line2D(
                [0, 1], [0, 0],
                color='gray',
                linestyle='--',
                linewidth=PlotConfig.THRESHOLD_LINE_WIDTH,
                label='Threshold'
            )
            legend_handles.append(threshold_line)
        
        legend = ax.legend(
            handles=legend_handles,
            loc='center',
            ncol=1,
            frameon=False,
            fontsize=PlotConfig.LEGEND_TEXT_SIZE,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=1.0
        )
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        
        if filename is None:
            filename = create_plot_filename(
                'rmsd_legend_vertical', model_type=model_type,
                input_type=input_type, threshold=add_threshold
            )
        
        if save:
            save_figure(fig, filename)
            
        return fig 