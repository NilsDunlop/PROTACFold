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
    # Figure dimensions (used in function parameters)
    DEFAULT_WIDTH = 12
    DEFAULT_HEIGHT = 20
    DEFAULT_COMBINED_WIDTH = 10
    DEFAULT_COMBINED_HEIGHT = 12
    
    # Font size settings
    AXIS_LABEL_SIZE = 14
    TICK_LABEL_SIZE = 13
    LEGEND_FONT_SIZE = 13
    
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
    
    # Grid settings
    GRID_STYLE = '--'
    GRID_ALPHA = 0.3
    
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
    
    def plot_property_vs_lrmsd(self, df, property_type, molecule_type="PROTAC", 
                            width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, save=False, 
                            output_path=None):
        """
        Plot LRMSD vs molecular property for AF3 and Boltz-1 models.
        
        Args:
            df: DataFrame containing the data
            property_type: Type of property to analyze ("Rotatable_Bond_Count", 
                          "Heavy_Atom_Count", or "Molecular_Weight")
            molecule_type: Type of molecule to filter by ("PROTAC" or "MOLECULAR GLUE")
            width, height: Figure dimensions
            save: Whether to save the figure
            output_path: Path to save the figure
            
        Returns:
            fig, ax: Created figure and axis
        """
        # Apply configured style settings
        PlotConfig.apply_style()
        
        # Filter for molecule type
        if molecule_type in self.MOLECULE_TYPES:
            df_filtered = df[df['TYPE'] == molecule_type].copy()
        else:
            # If invalid molecule type, use both
            df_filtered = df[(df['TYPE'] == "PROTAC") | (df['TYPE'] == "MOLECULAR GLUE")].copy()
        
        # Check if property exists in data
        if property_type not in df_filtered.columns:
            print(f"Error: Property {property_type} not found in data")
            return None, None
        
        # Create figure
        fig, ax = self.create_figure(width, height)
        
        # Group by property value and model type
        property_bins = {}
        max_property_value = df_filtered[property_type].max()
        
        # Define bin ranges based on property type
        if property_type == "Rotatable_Bond_Count":
            bin_ranges = range(0, int(max_property_value) + 2, 2)  # Every 2 rotatable bonds
        elif property_type == "Heavy_Atom_Count":
            bin_ranges = range(0, int(max_property_value) + 10, 10)  # Every 10 heavy atoms
        elif property_type == "Molecular_Weight":
            bin_ranges = range(0, int(max_property_value) + 100, 100)  # Every 100 in molecular weight
        else:
            bin_ranges = range(0, int(max_property_value) + 5, 5)  # Default binning
        
        # Create empty data containers for plotting
        bin_centers = []
        for i in range(len(bin_ranges) - 1):
            bin_min = bin_ranges[i]
            bin_max = bin_ranges[i+1]
            bin_center = (bin_min + bin_max) / 2
            bin_centers.append(bin_center)
            property_bins[bin_center] = {model: {'smiles': [], 'ccd': []} for model in self.MODEL_TYPES}
        
        # Group data by property bins
        for _, row in df_filtered.iterrows():
            property_value = row[property_type]
            model_type = row['MODEL_TYPE']
            
            # Skip if model type not of interest or missing LRMSD values
            if model_type not in self.MODEL_TYPES:
                continue
            
            # Find appropriate bin
            for i in range(len(bin_ranges) - 1):
                bin_min = bin_ranges[i]
                bin_max = bin_ranges[i+1]
                if bin_min <= property_value < bin_max:
                    bin_center = (bin_min + bin_max) / 2
                    
                    # Add LRMSD values to the respective bin
                    if pd.notna(row['SMILES_DOCKQ_LRMSD']):
                        property_bins[bin_center][model_type]['smiles'].append(row['SMILES_DOCKQ_LRMSD'])
                    if pd.notna(row['CCD_DOCKQ_LRMSD']):
                        property_bins[bin_center][model_type]['ccd'].append(row['CCD_DOCKQ_LRMSD'])
                    
                    break
        
        # Calculate means for each bin
        x_values = []
        af3_means = []
        boltz_means = []
        
        for bin_center in bin_centers:
            # Only include bins with data
            if len(property_bins[bin_center]['AlphaFold3']['smiles']) > 0 or \
               len(property_bins[bin_center]['AlphaFold3']['ccd']) > 0 or \
               len(property_bins[bin_center]['Boltz1']['smiles']) > 0 or \
               len(property_bins[bin_center]['Boltz1']['ccd']) > 0:
                
                x_values.append(bin_center)
                
                # AlphaFold3 (combine SMILES and CCD LRMSD values)
                af3_lrmsd = property_bins[bin_center]['AlphaFold3']['smiles'] + \
                          property_bins[bin_center]['AlphaFold3']['ccd']
                
                if len(af3_lrmsd) > 0:
                    af3_means.append(np.mean(af3_lrmsd))
                else:
                    af3_means.append(np.nan)
                
                # Boltz1 (combine SMILES and CCD LRMSD values)
                boltz_lrmsd = property_bins[bin_center]['Boltz1']['smiles'] + \
                            property_bins[bin_center]['Boltz1']['ccd']
                
                if len(boltz_lrmsd) > 0:
                    boltz_means.append(np.mean(boltz_lrmsd))
                else:
                    boltz_means.append(np.nan)
        
        # Convert to numpy arrays for easier handling of NaN values
        x_values = np.array(x_values)
        af3_means = np.array(af3_means)
        boltz_means = np.array(boltz_means)
        
        # Plot the data - line plots with no error bars
        ax.plot(x_values, af3_means, 'o-', color=self.MODEL_COLORS['AlphaFold3'], 
              linewidth=2, label='AlphaFold3')
        ax.plot(x_values, boltz_means, 'o-', color=self.MODEL_COLORS['Boltz1'], 
              linewidth=2, label='Boltz-1')
        
        # Set axis labels
        property_label = self.PROPERTY_LABELS.get(property_type, property_type)
        
        ax.set_xlabel(property_label, fontsize=self.AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('LRMSD (Å)', fontsize=self.AXIS_LABEL_SIZE, fontweight='bold')
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Add grid
        ax.grid(True, linestyle=self.GRID_STYLE, alpha=self.GRID_ALPHA)
        
        # Set tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=self.TICK_LABEL_SIZE)
        
        # Add legend with no frame
        ax.legend(fontsize=self.LEGEND_FONT_SIZE, frameon=self.LEGEND_FRAME)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save:
            property_short = property_type.replace('_', '')
            filename = f'{molecule_type.lower()}_lrmsd_vs_{property_short}'.lower()
            self.save_plot(fig, filename, save_path=output_path)
            
        return fig, ax
    
    def plot_combined_properties(self, df, molecule_type="PROTAC", 
                               width=DEFAULT_COMBINED_WIDTH, height=None, save=False, 
                               output_path=None, bin_settings=None):
        """
        Plot six properties with standard error shown as shaded regions around each line.
        Arranged in a 3×2 grid (3 rows, 2 columns).
        
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
        # Set up plotting environment
        plt.close('all')
        plt.rcdefaults()
        
        # Apply configured style settings
        PlotConfig.apply_style()
        
        # Clear any cached variables
        if hasattr(self, '_cached_bin_settings'):
            delattr(self, '_cached_bin_settings')
        
        # Filter for molecule type
        if molecule_type in self.MOLECULE_TYPES:
            df_filtered = df[df['TYPE'] == molecule_type].copy()
        else:
            # If invalid molecule type, use both
            df_filtered = df[(df['TYPE'] == "PROTAC") | (df['TYPE'] == "MOLECULAR GLUE")].copy()
        
        # Use provided bin settings or defaults
        if bin_settings is None:
            bin_settings = self.DEFAULT_BIN_SIZES.copy()
        else:
            # Create a new dictionary for bin_settings
            bin_settings_copy = {}
            for prop, default_size in self.DEFAULT_BIN_SIZES.items():
                if prop in bin_settings and bin_settings[prop] is not None:
                    bin_settings_copy[prop] = bin_settings[prop]
                else:
                    bin_settings_copy[prop] = default_size
            bin_settings = bin_settings_copy
        
        # Calculate figure dimensions
        if height is None:
            # Use default height for 3 rows of plots
            height = self.DEFAULT_COMBINED_HEIGHT
        
        # Create a new figure
        fig = plt.figure(figsize=(width, height))
        
        # Create grid layout with 3 rows and 2 columns with reduced spacing
        gs = fig.add_gridspec(nrows=3, ncols=2, 
                           hspace=self.SUBPLOT_HSPACE, 
                           wspace=self.SUBPLOT_WSPACE)
        
        # Maximize figure size to use all available space
        plt.rcParams['figure.constrained_layout.use'] = True
        
        # Set common style - uses PlotConfig.apply_style() for font settings
        # These settings complement the font settings from PlotConfig
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['grid.alpha'] = self.GRID_ALPHA
        
        # Colors for each model
        af3_color = self.MODEL_COLORS['AlphaFold3']
        boltz_color = self.MODEL_COLORS['Boltz1']
        
        # Create arrays to store axes
        all_axes = []
        
        # Create all subplots and populate with data
        for i in range(len(self.COMBINED_PROPERTIES)):
            # Calculate grid position (for 3×2 grid)
            row = i // 2   # Integer division: 0,1 → 0; 2,3 → 1; 4,5 → 2
            col = i % 2    # Remainder: 0,2,4 → 0; 1,3,5 → 1
            
            # Create main plot (no separate error plot)
            ax = fig.add_subplot(gs[row, col])
            all_axes.append(ax)
            
            # Check if property exists in data
            if self.COMBINED_PROPERTIES[i] not in df_filtered.columns:
                print(f"Error: Property {self.COMBINED_PROPERTIES[i]} not found in data")
                ax.text(0.5, 0.5, f"{self.COMBINED_PROPERTIES[i]} not found", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax.transAxes)
                continue
            
            # Group by property value and model type
            property_bins = {}
            max_property_value = df_filtered[self.COMBINED_PROPERTIES[i]].max()
            
            # Get bin size for this property
            bin_size = bin_settings[self.COMBINED_PROPERTIES[i]]
            
            # Create bins for this property
            bin_edges = list(range(0, int(max_property_value) + bin_size + 1, bin_size))
            
            # Special handling for LogP which can be negative
            if self.COMBINED_PROPERTIES[i] == "LogP":
                min_property_value = df_filtered[self.COMBINED_PROPERTIES[i]].min()
                bin_start = math.floor(min_property_value / bin_size) * bin_size
                bin_end = math.ceil(max_property_value / bin_size) * bin_size
                bin_edges = list(np.arange(bin_start, bin_end + bin_size, bin_size))
            
            # Calculate bin centers
            bin_centers = [(bin_edges[j] + bin_edges[j+1]) / 2 for j in range(len(bin_edges)-1)]
            
            # Initialize data containers
            for bin_center in bin_centers:
                property_bins[bin_center] = {model: {'smiles': [], 'ccd': []} for model in self.MODEL_TYPES}
            
            # Group data by property bins
            for _, row in df_filtered.iterrows():
                property_value = row[self.COMBINED_PROPERTIES[i]]
                model_type = row['MODEL_TYPE']
                
                # Skip if model type not of interest or missing property value
                if model_type not in self.MODEL_TYPES or pd.isna(property_value):
                    continue
                
                # Find appropriate bin
                for j in range(len(bin_edges)-1):
                    bin_min = bin_edges[j]
                    bin_max = bin_edges[j+1]
                    if bin_min <= property_value < bin_max:
                        bin_center = bin_centers[j]
                        
                        # Add LRMSD values to the respective bin
                        if pd.notna(row['SMILES_DOCKQ_LRMSD']):
                            property_bins[bin_center][model_type]['smiles'].append(row['SMILES_DOCKQ_LRMSD'])
                        if pd.notna(row['CCD_DOCKQ_LRMSD']):
                            property_bins[bin_center][model_type]['ccd'].append(row['CCD_DOCKQ_LRMSD'])
                        
                        break
            
            # Calculate means and standard errors for each bin
            x_values = []
            af3_means = []
            af3_errors = []
            boltz_means = []
            boltz_errors = []
            
            for bin_center in bin_centers:
                # Only include bins with data
                af3_data = property_bins[bin_center]['AlphaFold3']['smiles'] + property_bins[bin_center]['AlphaFold3']['ccd']
                boltz_data = property_bins[bin_center]['Boltz1']['smiles'] + property_bins[bin_center]['Boltz1']['ccd']
                
                if len(af3_data) > 0 or len(boltz_data) > 0:
                    x_values.append(bin_center)
                    
                    # AlphaFold3 stats
                    if len(af3_data) > 0:
                        af3_means.append(np.mean(af3_data))
                        af3_errors.append(np.std(af3_data)/np.sqrt(len(af3_data)) if len(af3_data) > 1 else 0)
                    else:
                        af3_means.append(np.nan)
                        af3_errors.append(0)
                    
                    # Boltz1 stats
                    if len(boltz_data) > 0:
                        boltz_means.append(np.mean(boltz_data))
                        boltz_errors.append(np.std(boltz_data)/np.sqrt(len(boltz_data)) if len(boltz_data) > 1 else 0)
                    else:
                        boltz_means.append(np.nan)
                        boltz_errors.append(0)
            
            # Convert to numpy arrays
            x_values = np.array(x_values)
            af3_means = np.array(af3_means)
            af3_errors = np.array(af3_errors)
            boltz_means = np.array(boltz_means)
            boltz_errors = np.array(boltz_errors)
            
            if len(x_values) > 0:
                # Plot AlphaFold3 data with error regions
                ax.plot(x_values, af3_means, 'o-', color=af3_color, 
                     linewidth=2.0, markersize=6, label='AlphaFold3')
                
                # Add shaded error region for AlphaFold3
                ax.fill_between(x_values, 
                             af3_means - af3_errors, 
                             af3_means + af3_errors, 
                             color=af3_color, alpha=self.ERROR_ALPHA)
                
                # Plot Boltz1 data with error regions
                ax.plot(x_values, boltz_means, 'o-', color=boltz_color, 
                     linewidth=2.0, markersize=6, label='Boltz-1')
                
                # Add shaded error region for Boltz1
                ax.fill_between(x_values, 
                             boltz_means - boltz_errors, 
                             boltz_means + boltz_errors, 
                             color=boltz_color, alpha=self.ERROR_ALPHA)
                
                # Set x-axis limits
                padding = bin_size * 0.5
                x_min = min(x_values) - padding
                x_max = max(x_values) + padding
                ax.set_xlim(x_min, x_max)
            
            # Set axis labels
            property_label = self.PROPERTY_LABELS.get(self.COMBINED_PROPERTIES[i], self.COMBINED_PROPERTIES[i])
            ax.set_xlabel(property_label, fontsize=self.AXIS_LABEL_SIZE, fontweight='bold')
            
            # Add y-label to first column plots 
            if col == 0:
                ax.set_ylabel('LRMSD (Å)', fontsize=self.AXIS_LABEL_SIZE, fontweight='bold')
            
            # Set y-axis limits based on molecule type
            ax.set_ylim(self.Y_AXIS_LIMITS.get(molecule_type, self.Y_AXIS_LIMITS["DEFAULT"]))
            
            # Set tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=self.TICK_LABEL_SIZE)
            
            # Add grid
            ax.grid(True, linestyle=self.GRID_STYLE, alpha=self.GRID_ALPHA)
            
            # Add legend to first plot only with no frame
            if i == 0:
                ax.legend(fontsize=self.LEGEND_FONT_SIZE, frameon=self.LEGEND_FRAME, loc='upper left')
            
            # Format tick labels for integers vs decimals
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x) if x == int(x) else f"{x:.1f}"))

            # Special formatting for integer-based properties
            if self.COMBINED_PROPERTIES[i] in ["HBD_Count", "HBA_Count"]:
                current_min, current_max = ax.get_xlim()
                integer_ticks = np.arange(math.ceil(current_min), math.floor(current_max) + 1)
                ax.set_xticks(integer_ticks)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
                
                # Special handling for HBA_Count to use even-numbered ticks
                if self.COMBINED_PROPERTIES[i] == "HBA_Count":
                    current_min, current_max = ax.get_xlim()
                    start = math.ceil(current_min)
                    if start % 2 != 0:
                        start += 1
                    end = math.floor(current_max)
                    if end % 2 != 0:
                        end -= 1
                    even_ticks = np.arange(start, end + 1, 2)
                    ax.set_xticks(even_ticks)
        
        # Apply tight layout
        #plt.tight_layout()
        
        # Save if requested
        if save:
            bin_suffix = (f"_MW{bin_settings['Molecular_Weight']}_HAC{bin_settings['Heavy_Atom_Count']}"
                         f"_RBC{bin_settings['Rotatable_Bond_Count']}_LP{bin_settings['LogP']}"
                         f"_HBD{bin_settings['HBD_Count']}_HBA{bin_settings['HBA_Count']}")
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
        # Apply configured style settings
        PlotConfig.apply_style()
        
        # Filter for molecule type
        if molecule_type in self.MOLECULE_TYPES:
            df_filtered = df[df['TYPE'] == molecule_type].copy()
        else:
            # If invalid molecule type, use both
            df_filtered = df[(df['TYPE'] == "PROTAC") | (df['TYPE'] == "MOLECULAR GLUE")].copy()
        
        # Calculate ranges
        ranges = {}
        for prop in self.COMBINED_PROPERTIES:
            if prop in df_filtered.columns:
                min_val = df_filtered[prop].min()
                max_val = df_filtered[prop].max()
                ranges[prop] = (min_val, max_val)
        
        return ranges