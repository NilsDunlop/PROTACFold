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
    
    def plot_property_vs_lrmsd(self, df, property_type, molecule_type="PROTAC", 
                            width=10, height=8, save=False, 
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
        # Filter for molecule type
        if molecule_type in ["PROTAC", "MOLECULAR GLUE"]:
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
        
        # Set up colors for models
        model_colors = {
            'AlphaFold3': 'red',
            'Boltz1': 'blue'
        }
        
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
            property_bins[bin_center] = {'AlphaFold3': {'smiles': [], 'ccd': []}, 
                                       'Boltz1': {'smiles': [], 'ccd': []}}
        
        # Group data by property bins
        for _, row in df_filtered.iterrows():
            property_value = row[property_type]
            model_type = row['MODEL_TYPE']
            
            # Skip if model type not of interest or missing LRMSD values
            if model_type not in ['AlphaFold3', 'Boltz1']:
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
        ax.plot(x_values, af3_means, 'o-', color='red', 
              linewidth=2, label='AlphaFold3')
        ax.plot(x_values, boltz_means, 'o-', color='blue', 
              linewidth=2, label='Boltz-1')
        
        # Set axis labels
        property_labels = {
            'Rotatable_Bond_Count': 'Rotatable Bond Count',
            'Heavy_Atom_Count': 'Heavy Atom Count',
            'Molecular_Weight': 'Molecular Weight',
            'LogP': 'LogP',
            'HBD_Count': 'H-Bond Donors',
            'HBA_Count': 'H-Bond Acceptors'
        }
        
        property_label = property_labels.get(property_type, property_type)
        
        ax.set_xlabel(property_label, fontsize=PlotConfig.AXIS_LABEL_SIZE)
        ax.set_ylabel('LRMSD (Å)', fontsize=PlotConfig.AXIS_LABEL_SIZE)
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(fontsize=PlotConfig.LEGEND_TEXT_SIZE)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save:
            property_short = property_type.replace('_', '')
            filename = f'{molecule_type.lower()}_lrmsd_vs_{property_short}'.lower()
            self.save_plot(fig, filename, save_path=output_path)
            
        return fig, ax
    
    def plot_combined_properties(self, df, molecule_type="PROTAC", 
                               width=15, height=None, save=False, 
                               output_path=None, bin_settings=None):
        """
        Plot six properties in a 3×2 grid, with separate error bars above each plot.
        Top row: Molecular Weight, Heavy Atom Count, Rotatable Bond Count
        Bottom row: LogP, HBD_Count, HBA_Count
        
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
        
        # Clear any cached variables
        if hasattr(self, '_cached_bin_settings'):
            delattr(self, '_cached_bin_settings')
        
        # Filter for molecule type
        if molecule_type in ["PROTAC", "MOLECULAR GLUE"]:
            df_filtered = df[df['TYPE'] == molecule_type].copy()
        else:
            # If invalid molecule type, use both
            df_filtered = df[(df['TYPE'] == "PROTAC") | (df['TYPE'] == "MOLECULAR GLUE")].copy()
        
        # Property types to plot in two rows
        property_types = [
            # Top row
            "Molecular_Weight", "Heavy_Atom_Count", "Rotatable_Bond_Count",
            # Bottom row
            "LogP", "HBD_Count", "HBA_Count"
        ]
        
        property_labels = {
            'Rotatable_Bond_Count': 'Rotatable Bond Count',
            'Heavy_Atom_Count': 'Heavy Atom Count',
            'Molecular_Weight': 'Molecular Weight',
            'LogP': 'LogP',
            'HBD_Count': 'H-Bond Donors',
            'HBA_Count': 'H-Bond Acceptors'
        }
        
        # Default bin sizes if not provided
        default_bin_sizes = {
            "Molecular_Weight": 100,
            "Heavy_Atom_Count": 10,
            "Rotatable_Bond_Count": 10,
            "LogP": 1,
            "HBD_Count": 1,
            "HBA_Count": 2
        }
        
        # Use provided bin settings or defaults
        if bin_settings is None:
            bin_settings = default_bin_sizes.copy()
        else:
            # Create a new dictionary for bin_settings
            bin_settings_copy = {}
            for prop, default_size in default_bin_sizes.items():
                if prop in bin_settings and bin_settings[prop] is not None:
                    bin_settings_copy[prop] = bin_settings[prop]
                else:
                    bin_settings_copy[prop] = default_size
            bin_settings = bin_settings_copy
        
        # Calculate figure dimensions
        orig_height = 7
        subplot_height = orig_height / 2
        num_rows = 2
        new_height = subplot_height * num_rows * 1.1
        
        # Create a new figure
        fig = plt.figure(figsize=(width, new_height))
        
        # Create grid layout with specific spacing
        gs = fig.add_gridspec(nrows=4, ncols=3, 
                           height_ratios=[2.5, 4, 2.5, 4],
                           hspace=0.2, wspace=0.35)
        
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.35)
        
        # Set common style
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['grid.alpha'] = 0.3
        
        # Colors for each model
        af3_color = 'red'
        boltz_color = 'blue'
        
        # Create arrays to store axes and data
        main_axes = []
        error_axes = []
        all_x_values = []
        all_bin_sizes = []
        all_x_ranges = []
        
        # Create all subplots and populate with data
        for i in range(len(property_types)):
            # Calculate grid position
            row_offset = 0 if i < 3 else 2
            col = i % 3
            
            # Create error plot
            ax_error = fig.add_subplot(gs[row_offset, col])
            
            # Create main plot
            ax_main = fig.add_subplot(gs[row_offset + 1, col])
            
            # Hide x-tick labels on error plots
            ax_error.tick_params(axis='x', labelbottom=False)
            
            # Store references
            error_axes.append(ax_error)
            main_axes.append(ax_main)
            
            # Check if property exists in data
            if property_types[i] not in df_filtered.columns:
                print(f"Error: Property {property_types[i]} not found in data")
                ax_main.text(0.5, 0.5, f"{property_types[i]} not found", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax_main.transAxes)
                continue
            
            # Group by property value and model type
            property_bins = {}
            max_property_value = df_filtered[property_types[i]].max()
            
            # Get bin size for this property
            bin_size = bin_settings[property_types[i]]
            all_bin_sizes.append(bin_size)
            
            # Create bins for this property
            bin_edges = list(range(0, int(max_property_value) + bin_size + 1, bin_size))
            
            # Special handling for LogP which can be negative
            if property_types[i] == "LogP":
                min_property_value = df_filtered[property_types[i]].min()
                bin_start = math.floor(min_property_value / bin_size) * bin_size
                bin_end = math.ceil(max_property_value / bin_size) * bin_size
                bin_edges = list(np.arange(bin_start, bin_end + bin_size, bin_size))
            
            # Calculate bin centers
            bin_centers = [(bin_edges[j] + bin_edges[j+1]) / 2 for j in range(len(bin_edges)-1)]
            
            # Initialize data containers
            for bin_center in bin_centers:
                property_bins[bin_center] = {
                    'AlphaFold3': {'smiles': [], 'ccd': []}, 
                    'Boltz1': {'smiles': [], 'ccd': []}
                }
            
            # Group data by property bins
            for _, row in df_filtered.iterrows():
                property_value = row[property_types[i]]
                model_type = row['MODEL_TYPE']
                
                # Skip if model type not of interest or missing property value
                if model_type not in ['AlphaFold3', 'Boltz1'] or pd.isna(property_value):
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
            all_x_values.append(x_values)
            af3_means = np.array(af3_means)
            af3_errors = np.array(af3_errors)
            boltz_means = np.array(boltz_means)
            boltz_errors = np.array(boltz_errors)
            
            # Calculate the x-axis range
            if len(x_values) > 0:
                x_min = min(x_values) - bin_size * 0.5
                x_max = max(x_values) + bin_size * 0.5
                all_x_ranges.append((x_min, x_max))
            else:
                all_x_ranges.append((0, 1))
                continue

            # Plot error bars in top subplot
            if len(x_values) > 0:
                ax_error.bar(x_values - bin_size/6, af3_errors, 
                          width=bin_size/3, color=af3_color, alpha=0.7,
                          edgecolor='black', linewidth=0.5, label='AF3 StdErr')
                
                ax_error.bar(x_values + bin_size/6, boltz_errors, 
                          width=bin_size/3, color=boltz_color, alpha=0.7,
                          edgecolor='black', linewidth=0.5, label='Boltz-1 StdErr')
                
                # Set y-axis label for error plots
                if i == 0:
                    ax_error.set_ylabel('StdErr', fontsize=PlotConfig.AXIS_LABEL_SIZE-1)
                elif i == 3:
                    ax_error.set_ylabel('StdErr', fontsize=PlotConfig.AXIS_LABEL_SIZE-1)
                
                ax_error.tick_params(axis='x', labelbottom=False)
                ax_error.grid(axis='y', linestyle='--', alpha=0.3)
                
                # Store error values for consistent scaling later
                if not hasattr(self, '_max_error_values'):
                    self._max_error_values = []
                self._max_error_values.append(max(np.nanmax(af3_errors) if len(af3_errors) > 0 else 0,
                                              np.nanmax(boltz_errors) if len(boltz_errors) > 0 else 0))
                
                # Set x-axis limits
                padding = bin_size * 0.5
                x_min = min(x_values) - padding
                x_max = max(x_values) + padding
                ax_error.set_xlim(x_min, x_max)
            
            # Plot main data in bottom subplot
            if len(x_values) > 0:
                # Plot lines
                ax_main.plot(x_values, af3_means, 'o-', color=af3_color, 
                          linewidth=2.0, markersize=6, label='AlphaFold3')
                ax_main.plot(x_values, boltz_means, 'o-', color=boltz_color, 
                          linewidth=2.0, markersize=6, label='Boltz-1')
                
                # Set x-axis limits
                padding = bin_size * 0.5
                x_min = min(x_values) - padding
                x_max = max(x_values) + padding
                ax_main.set_xlim(x_min, x_max)
            
            # Set axis labels for main plot
            property_label = property_labels.get(property_types[i], property_types[i])
            ax_main.set_xlabel(property_label, fontsize=PlotConfig.AXIS_LABEL_SIZE)
            
            # Add y-label to first column plots in each row
            if i == 0 or i == 3:
                ax_main.set_ylabel('LRMSD (Å)', fontsize=PlotConfig.AXIS_LABEL_SIZE)
            
            # Set y-axis limits based on molecule type
            if molecule_type == "MOLECULAR GLUE":
                ax_main.set_ylim(bottom=0, top=55)
            else:
                ax_main.set_ylim(bottom=0, top=45)
            
            # Set tick label sizes
            ax_main.tick_params(axis='both', which='major', labelsize=10)
            ax_error.tick_params(axis='both', which='major', labelsize=8)
            
            # Add grid
            ax_main.grid(True, linestyle='--', alpha=0.3)
            
            # Add legend to first plot only
            if i == 0:
                ax_main.legend(fontsize=10, loc='upper left')
            
            # Format tick labels for integers vs decimals
            ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x) if x == int(x) else f"{x:.1f}"))

            # Special formatting for integer-based properties
            if property_types[i] in ["HBD_Count", "HBA_Count"]:
                current_min, current_max = ax_main.get_xlim()
                integer_ticks = np.arange(math.ceil(current_min), math.floor(current_max) + 1)
                ax_main.set_xticks(integer_ticks)
                ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
                
                # Special handling for HBA_Count to use even-numbered ticks
                if property_types[i] == "HBA_Count":
                    current_min, current_max = ax_main.get_xlim()
                    start = math.ceil(current_min)
                    if start % 2 != 0:
                        start += 1
                    end = math.floor(current_max)
                    if end % 2 != 0:
                        end -= 1
                    even_ticks = np.arange(start, end + 1, 2)
                    ax_main.set_xticks(even_ticks)
        
        # Calculate consistent visual bar widths for error plots
        normalized_widths = []
        for i, (ax_error, ax_main) in enumerate(zip(error_axes, main_axes)):
            if i < len(all_x_ranges):
                x_min, x_max = all_x_ranges[i]
                x_range = x_max - x_min
                normalized_widths.append(x_range * 0.05)  # 5% of axis width
        
        # Redraw error bars with consistent visual width
        for i, (ax_error, ax_main) in enumerate(zip(error_axes, main_axes)):
            if i < len(all_x_values) and len(all_x_values[i]) > 0:
                # Clear previous bars
                ax_error.clear()
                
                x_values = all_x_values[i]
                
                # Get the corresponding data
                af3_errors = []
                boltz_errors = []
                
                # Recalculate standard errors
                for bin_center in x_values:
                    if i < len(property_types):
                        prop_type = property_types[i]
                        af3_data = []
                        boltz_data = []
                        
                        # Collect data for each bin
                        for _, row in df_filtered.iterrows():
                            property_value = row[prop_type]
                            model_type = row['MODEL_TYPE']
                            bin_size = all_bin_sizes[i]
                            
                            if model_type not in ['AlphaFold3', 'Boltz1'] or pd.isna(property_value):
                                continue
                            
                            # Check if property value is in this bin
                            if abs(property_value - bin_center) <= bin_size/2:
                                if model_type == 'AlphaFold3':
                                    if pd.notna(row['SMILES_DOCKQ_LRMSD']):
                                        af3_data.append(row['SMILES_DOCKQ_LRMSD'])
                                    if pd.notna(row['CCD_DOCKQ_LRMSD']):
                                        af3_data.append(row['CCD_DOCKQ_LRMSD'])
                                elif model_type == 'Boltz1':
                                    if pd.notna(row['SMILES_DOCKQ_LRMSD']):
                                        boltz_data.append(row['SMILES_DOCKQ_LRMSD'])
                                    if pd.notna(row['CCD_DOCKQ_LRMSD']):
                                        boltz_data.append(row['CCD_DOCKQ_LRMSD'])
                        
                        # Calculate standard errors
                        if len(af3_data) > 0:
                            af3_errors.append(np.std(af3_data)/np.sqrt(len(af3_data)) if len(af3_data) > 1 else 0)
                        else:
                            af3_errors.append(0)
                            
                        if len(boltz_data) > 0:
                            boltz_errors.append(np.std(boltz_data)/np.sqrt(len(boltz_data)) if len(boltz_data) > 1 else 0)
                        else:
                            boltz_errors.append(0)
                
                # Use a consistent visual bar width
                bar_width = normalized_widths[i] if i < len(normalized_widths) else normalized_widths[0]
                bar_offset = bar_width / 2
                
                # Plot error bars with consistent width
                if len(x_values) > 0 and len(af3_errors) == len(x_values) and len(boltz_errors) == len(x_values):
                    # Plot AF3 error bars
                    ax_error.bar(x_values - bar_offset, af3_errors, 
                              width=bar_width, color=af3_color, alpha=0.7,
                              edgecolor='black', linewidth=0.5, label='AF3 StdErr')
                    
                    # Plot Boltz-1 error bars
                    ax_error.bar(x_values + bar_offset, boltz_errors, 
                              width=bar_width, color=boltz_color, alpha=0.7,
                              edgecolor='black', linewidth=0.5, label='Boltz-1 StdErr')
                    
                    # Set y-axis label for error plots
                    if i == 0:
                        ax_error.set_ylabel('StdErr', fontsize=PlotConfig.AXIS_LABEL_SIZE-1)
                    elif i == 3:
                        ax_error.set_ylabel('StdErr', fontsize=PlotConfig.AXIS_LABEL_SIZE-1)
                    
                    ax_error.tick_params(axis='x', labelbottom=False)
                    ax_error.grid(axis='y', linestyle='--', alpha=0.3)
                    
                    # Set consistent y limit for error plots
                    max_error = max(np.nanmax(af3_errors) if len(af3_errors) > 0 else 0,
                                  np.nanmax(boltz_errors) if len(boltz_errors) > 0 else 0)
                    if max_error > 0:
                        ax_error.set_ylim(0, max_error * 1.2)
                    
                    # Set x-axis limits
                    ax_error.set_xlim(all_x_ranges[i])
        
        # Apply consistent formatting to all plots
        for ax_error in error_axes:
            ax_error.set_ylim(0, 8)
            ax_error.set_yticks([0, 2, 4, 6, 8])
            ax_error.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
        
        for i, ax_main in enumerate(main_axes):
            ax_main.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
            
            # Apply tick formatting based on property type
            if i < len(property_types):
                prop_type = property_types[i % len(property_types)]
                
                # Special handling for integer properties
                if prop_type == "HBD_Count":
                    current_min, current_max = ax_main.get_xlim()
                    integer_ticks = np.arange(math.ceil(current_min), math.floor(current_max) + 1)
                    ax_main.set_xticks(integer_ticks)
                    ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
                elif prop_type == "HBA_Count":
                    current_min, current_max = ax_main.get_xlim()
                    start = math.ceil(current_min)
                    if start % 2 != 0:
                        start += 1
                    end = math.floor(current_max)
                    if end % 2 != 0:
                        end -= 1
                    even_ticks = np.arange(start, end + 1, 2)
                    ax_main.set_xticks(even_ticks)
                    ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
                else:
                    ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x) if x == int(x) else f"{x:.1f}"))
        
        # Adjust vertical spacing to ensure labels are visible
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.25, wspace=0.35)
        
        # Set consistent spacing between rows
        if len(error_axes) >= 3 and len(main_axes) >= 3:
            # Reference height from first error plot
            ref_height = error_axes[0].get_position().height
            
            # Make all error plots consistent height
            for ax_error in error_axes[1:]:
                pos = ax_error.get_position()
                ax_error.set_position([pos.x0, pos.y0, pos.width, ref_height])
            
            # Add spacing between top and bottom rows
            y_offset = 0.05
            
            # Apply spacing for six-plot layout
            if len(error_axes) >= 6 and len(main_axes) >= 6:
                # Make error plots slightly taller
                for i, ax_error in enumerate(error_axes):
                    pos = ax_error.get_position()
                    ax_error.set_position([pos.x0, pos.y0, pos.width, pos.height * 1.1])
                
                # Add space between top and bottom rows
                for i in range(3, 6):
                    if i < len(error_axes):
                        # Move error plots down
                        ax_error = error_axes[i]
                        pos = ax_error.get_position()
                        new_y0 = pos.y0 - y_offset
                        ax_error.set_position([pos.x0, new_y0, pos.width, pos.height])
                        
                        # Move main plots down in sync
                        if i < len(main_axes):
                            ax_main = main_axes[i]
                            main_pos = ax_main.get_position()
                            new_main_y0 = main_pos.y0 - y_offset
                            ax_main.set_position([main_pos.x0, new_main_y0, main_pos.width, main_pos.height])
        
        # Final update to ensure all plots are visible
        plt.draw()
        
        # Save if requested
        if save:
            bin_suffix = (f"_MW{bin_settings['Molecular_Weight']}_HAC{bin_settings['Heavy_Atom_Count']}"
                         f"_RBC{bin_settings['Rotatable_Bond_Count']}_LP{bin_settings['LogP']}"
                         f"_HBD{bin_settings['HBD_Count']}_HBA{bin_settings['HBA_Count']}")
            filename = f'{molecule_type.lower()}_lrmsd_combined_properties{bin_suffix}'.lower()
            self.save_plot(fig, filename, save_path=output_path)
        
        # Return axes in proper order
        all_axes = []
        for ax_error, ax_main in zip(error_axes, main_axes):
            all_axes.append(ax_error)
            all_axes.append(ax_main)
        
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
        # Filter for molecule type
        if molecule_type in ["PROTAC", "MOLECULAR GLUE"]:
            df_filtered = df[df['TYPE'] == molecule_type].copy()
        else:
            # If invalid molecule type, use both
            df_filtered = df[(df['TYPE'] == "PROTAC") | (df['TYPE'] == "MOLECULAR GLUE")].copy()
        
        # Property types to calculate ranges for
        property_types = [
            "Molecular_Weight", 
            "Heavy_Atom_Count", 
            "Rotatable_Bond_Count",
            "LogP", 
            "HBD_Count", 
            "HBA_Count"
        ]
        
        # Calculate ranges
        ranges = {}
        for prop in property_types:
            if prop in df_filtered.columns:
                min_val = df_filtered[prop].min()
                max_val = df_filtered[prop].max()
                ranges[prop] = (min_val, max_val)
        
        return ranges