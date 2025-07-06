import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from base_plotter import BasePlotter
from config import PlotConfig
from data_loader import DataLoader
from matplotlib.gridspec import GridSpec

class PropertyPlotter(BasePlotter):
    """Class for plotting LRMSD vs molecular properties."""
    
    # Styling parameters
    # Figure dimensions for combined property grid (2×3 layout)
    DEFAULT_COMBINED_WIDTH = 15
    DEFAULT_COMBINED_HEIGHT = 7
    
    # Font size settings
    AXIS_LABEL_SIZE = 14
    TICK_LABEL_SIZE = 13
    LEGEND_FONT_SIZE = 15
    
    # Y-axis limits for different molecule types
    Y_AXIS_LIMITS = {
        "MOLECULAR GLUE": (0, 55),
        "PROTAC": (0, 50),
        "DEFAULT": (0, 50)
    }
    
    # Error region transparency
    ERROR_ALPHA = 0.2
    
    # Bar width and spacing settings
    BAR_ALPHA = 1
    BAR_EDGE_COLOR = 'black'
    BAR_EDGE_WIDTH = 0.5
    NORMALIZED_WIDTH_FACTOR = 0.05  # 5% of axis width
    
    # Grid settings (matching comparison_plotter.py for consistent design)
    GRID_STYLE = '--'
    GRID_ALPHA = 0.2
    
    # Legend settings
    LEGEND_FRAME = False  # No border for legend
    
    # Subplot layout settings
    SUBPLOT_SPACING = 0.25
    SUBPLOT_WSPACE = 0.15  # Reduced from 0.35 to reduce horizontal spacing
    SUBPLOT_HSPACE = 0.3
    SUBPLOT_ROW_OFFSET = 0.05
    
    # Constants for models and colors
    MODEL_TYPES = ['AlphaFold3', 'Boltz1']
    MODEL_COLORS = {
        'AlphaFold3': PlotConfig.SMILES_PRIMARY,
        'Boltz1': PlotConfig.CCD_SECONDARY
    }
    
    # Constants for molecule types
    MOLECULE_TYPES = ["PROTAC", "MOLECULAR GLUE"]
    
    # Constants for property names and labels
    PROPERTY_LABELS = {
        'Rotatable_Bond_Count': 'Rotatable Bond Count',
        'Heavy_Atom_Count': 'Heavy Atom Count',
        'Molecular_Weight': 'Molecular Weight',
        'LogP': 'LogP',
        'HBD_Count': 'H-Bond Donors',
        'HBA_Count': 'H-Bond Acceptors'
    }
    
    # Default bin sizes for different properties
    DEFAULT_BIN_SIZES = {
        "Molecular_Weight": 50,
        "Heavy_Atom_Count": 10,
        "Rotatable_Bond_Count": 10,
        "LogP": 1,
        "HBD_Count": 1,
        "HBA_Count": 2
    }
    
    # Properties to display in combined plot
    COMBINED_PROPERTIES = [
        # Top row
        "Molecular_Weight", "Heavy_Atom_Count", "Rotatable_Bond_Count",
        # Bottom row
        "LogP", "HBD_Count", "HBA_Count"
    ]
    
    def _setup_plot_environment(self):
        plt.close('all')
        plt.rcdefaults()
        PlotConfig.apply_style()
        if hasattr(self, '_cached_bin_settings'):
            delattr(self, '_cached_bin_settings')
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['grid.alpha'] = self.GRID_ALPHA

    def _prepare_data_and_bins(self, df, molecule_type, bin_settings):
        if molecule_type in self.MOLECULE_TYPES:
            df_filtered = df[df['TYPE'] == molecule_type].copy()
        else:
            df_filtered = df[df['TYPE'].isin(self.MOLECULE_TYPES)].copy()

        final_bin_settings = self.DEFAULT_BIN_SIZES.copy()
        if bin_settings:
            final_bin_settings.update({k: v for k, v in bin_settings.items() if v is not None})
        
        return df_filtered, final_bin_settings
    
    def _get_bin_edges_and_centers(self, df_filtered, prop, bin_size):
        max_prop_val = df_filtered[prop].max()
        if prop == "LogP":
            min_prop_val = df_filtered[prop].min()
            bin_start = math.floor(min_prop_val / bin_size) * bin_size
            bin_end = math.ceil(max_prop_val / bin_size) * bin_size
            edges = list(np.arange(bin_start, bin_end + bin_size, bin_size))
        else:
            edges = list(range(0, int(max_prop_val) + bin_size + 1, bin_size))
        
        centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
        return edges, centers

    def _bin_data(self, df_filtered, prop, bin_edges, bin_centers):
        property_bins = {center: {model: {'smiles': [], 'ccd': []} for model in self.MODEL_TYPES} for center in bin_centers}
        
        for _, row_data in df_filtered.iterrows():
            prop_val = row_data.get(prop)
            model_type = row_data.get('MODEL_TYPE')

            if model_type not in self.MODEL_TYPES or pd.isna(prop_val):
                continue
            
            for i, center in enumerate(bin_centers):
                if bin_edges[i] <= prop_val < bin_edges[i+1]:
                    if pd.notna(row_data.get('SMILES_DOCKQ_LRMSD')):
                        property_bins[center][model_type]['smiles'].append(row_data['SMILES_DOCKQ_LRMSD'])
                    if pd.notna(row_data.get('CCD_DOCKQ_LRMSD')):
                        property_bins[center][model_type]['ccd'].append(row_data['CCD_DOCKQ_LRMSD'])
                    break
        return property_bins

    def _calculate_stats(self, property_bins, bin_centers):
        stats = {'x_values': []}
        for model in self.MODEL_TYPES:
            stats[model] = {'means': [], 'errors': []}

        for center in bin_centers:
            has_data = False
            model_data = {}
            for model in self.MODEL_TYPES:
                data = property_bins[center][model]['smiles'] + property_bins[center][model]['ccd']
                model_data[model] = data
                if data:
                    has_data = True
            
            if has_data:
                stats['x_values'].append(center)
                for model in self.MODEL_TYPES:
                    data = model_data[model]
                    if data:
                        stats[model]['means'].append(np.mean(data))
                        stats[model]['errors'].append(np.std(data) / np.sqrt(len(data)) if len(data) > 1 else 0)
                    else:
                        stats[model]['means'].append(np.nan)
                        stats[model]['errors'].append(np.nan)

        for model in self.MODEL_TYPES:
            stats[model]['means'] = np.array(stats[model]['means'])
            stats[model]['errors'] = np.array(stats[model]['errors'])
        stats['x_values'] = np.array(stats['x_values'])
        
        return stats

    def _plot_subplot(self, ax, x_values, model_stats, model, color):
        means = model_stats['means']
        errors = model_stats['errors']
        ax.plot(x_values, means, 'o-', color=color, linewidth=2.0, markersize=6, label=model)
        ax.fill_between(x_values, means - errors, means + errors, color=color, alpha=self.ERROR_ALPHA, where=~np.isnan(means))

    def _configure_axes(self, ax, prop, col_index, molecule_type, x_values, bin_size):
        ax.set_xlabel(self.PROPERTY_LABELS.get(prop, prop), fontsize=self.AXIS_LABEL_SIZE, fontweight='bold')
        if col_index == 0:
            ax.set_ylabel('LRMSD (Å)', fontsize=self.AXIS_LABEL_SIZE, fontweight='bold')
        
        ax.set_ylim(self.Y_AXIS_LIMITS.get(molecule_type, self.Y_AXIS_LIMITS["DEFAULT"]))
        ax.tick_params(axis='both', which='major', labelsize=self.TICK_LABEL_SIZE)
        ax.grid(axis='y', linestyle=self.GRID_STYLE, alpha=self.GRID_ALPHA)

        if len(x_values) > 0:
            padding = bin_size * 0.5
            ax.set_xlim(min(x_values) - padding, max(x_values) + padding)

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}" if x == int(x) else f"{x:.1f}"))
        if prop in ["HBD_Count", "HBA_Count"]:
            min_lim, max_lim = ax.get_xlim()
            if prop == "HBA_Count":
                start = math.ceil(min_lim)
                if start % 2 != 0: start += 1
                ax.set_xticks(np.arange(start, math.floor(max_lim) + 1, 2))
            else:
                ax.set_xticks(np.arange(math.ceil(min_lim), math.floor(max_lim) + 1))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
            
    def _add_figure_legend(self, fig):
        dummy_lines = [plt.Line2D([0], [0], color=self.MODEL_COLORS[model], linewidth=2, marker='o', markersize=6) for model in self.MODEL_TYPES]
        labels = ['AlphaFold3', 'Boltz-1']
        fig.legend(handles=dummy_lines, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                   ncol=2, fontsize=self.LEGEND_FONT_SIZE, frameon=self.LEGEND_FRAME)
    
    def plot_combined_properties(self, df, molecule_type="PROTAC", 
                               width=DEFAULT_COMBINED_WIDTH, height=None, save=False, 
                               output_path=None, bin_settings=None):
        """
        Plot six properties with standard error shown as shaded regions around each line.
        Arranged in a 2×3 grid (2 rows, 3 columns).
        
        Args:
            df: DataFrame containing the data
            molecule_type: Type of molecule to filter by ("PROTAC" or "MOLECULAR GLUE")
            width: Overall figure width
            height: Overall figure height (calculated automatically if None)
            save: Whether to save the figure
            output_path: Path to save the figure
            bin_settings: Dictionary with custom bin sizes for each property type
                          e.g., {"Molecular_Weight": 100, "Heavy_Atom_Count": 10, "Rotatable_Bond_Count": 2}
            
        Returns:
            fig, axes: Created figure and axes
        """
        self._setup_plot_environment()
        
        df_filtered, final_bin_settings = self._prepare_data_and_bins(df, molecule_type, bin_settings)
        
        if height is None:
            height = self.DEFAULT_COMBINED_HEIGHT
        
        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(nrows=2, ncols=3, hspace=self.SUBPLOT_HSPACE, wspace=self.SUBPLOT_WSPACE)
        plt.rcParams['figure.constrained_layout.use'] = True
        
        all_axes = []
        
        for i, prop in enumerate(self.COMBINED_PROPERTIES):
            row, col = i // 3, i % 3
            ax = fig.add_subplot(gs[row, col])
            all_axes.append(ax)
            
            if prop not in df_filtered.columns:
                ax.text(0.5, 0.5, f"{prop} not found", ha='center', va='center', transform=ax.transAxes)
                continue
            
            bin_size = final_bin_settings.get(prop, self.DEFAULT_BIN_SIZES[prop])
            bin_edges, bin_centers = self._get_bin_edges_and_centers(df_filtered, prop, bin_size)
            property_bins = self._bin_data(df_filtered, prop, bin_edges, bin_centers)
            stats = self._calculate_stats(property_bins, bin_centers)

            if len(stats['x_values']) > 0:
                self._plot_subplot(ax, stats['x_values'], stats['AlphaFold3'], 'AlphaFold3', self.MODEL_COLORS['AlphaFold3'])
                self._plot_subplot(ax, stats['x_values'], stats['Boltz1'], 'Boltz-1', self.MODEL_COLORS['Boltz1'])
            
            self._configure_axes(ax, prop, col, molecule_type, stats['x_values'], bin_size)
        
        self._add_figure_legend(fig)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        if save:
            bin_suffix = "".join([f"_{key.split('_')[0][0]}{val}" for key, val in final_bin_settings.items()])
            filename = f'{molecule_type.lower()}_lrmsd_combined_properties{bin_suffix}'.lower()
            self.save_plot(fig, filename, save_path=output_path)
        
        return fig, all_axes

    def get_property_ranges(self, df, molecule_type="PROTAC"):
        """
        Calculate the range (min/max) of each property in the dataset for a specific molecule type.
        
        Args:
            df: DataFrame containing the data
            molecule_type: Type of molecule to filter by ("PROTAC" or "MOLECULAR GLUE")
            
        Returns:
            Dictionary with property names as keys and (min, max) tuples as values
        """
        df_filtered, _ = self._prepare_data_and_bins(df, molecule_type, None)
        
        ranges = {}
        for prop in self.COMBINED_PROPERTIES:
            if prop in df_filtered.columns:
                ranges[prop] = (df_filtered[prop].min(), df_filtered[prop].max())
        
        return ranges