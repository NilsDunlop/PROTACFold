import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from base_plotter import BasePlotter
from config import PlotConfig
from data_loader import DataLoader

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
            'Molecular_Weight': 'Molecular Weight'
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
                               width=15, height=5, save=False, 
                               output_path=None):
        """
        Plot three properties (Molecular Weight, Heavy Atom Count, Rotatable Bond Count)
        side by side in a single figure.
        
        Args:
            df: DataFrame containing the data
            molecule_type: Type of molecule to filter by ("PROTAC" or "MOLECULAR GLUE")
            width, height: Overall figure dimensions
            save: Whether to save the figure
            output_path: Path to save the figure
            
        Returns:
            fig, axes: Created figure and axes
        """
        # Filter for molecule type
        if molecule_type in ["PROTAC", "MOLECULAR GLUE"]:
            df_filtered = df[df['TYPE'] == molecule_type].copy()
        else:
            # If invalid molecule type, use both
            df_filtered = df[(df['TYPE'] == "PROTAC") | (df['TYPE'] == "MOLECULAR GLUE")].copy()
        
        # Property types to plot in order: Molecular Weight, Heavy Atom Count, Rotatable Bond Count
        property_types = ["Molecular_Weight", "Heavy_Atom_Count", "Rotatable_Bond_Count"]
        property_labels = {
            'Rotatable_Bond_Count': 'Rotatable Bond Count',
            'Heavy_Atom_Count': 'Heavy Atom Count',
            'Molecular_Weight': 'Molecular Weight'
        }
        
        # Create a figure with three subplots side by side
        fig, axes = plt.subplots(1, 3, figsize=(width, height))
        plt.subplots_adjust(wspace=0.35)  # Add spacing between subplots
        
        # Plot each property
        for i, prop_type in enumerate(property_types):
            ax = axes[i]
            
            # Check if property exists in data
            if prop_type not in df_filtered.columns:
                print(f"Error: Property {prop_type} not found in data")
                ax.text(0.5, 0.5, f"{prop_type} not found", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax.transAxes)
                continue
            
            # Group by property value and model type
            property_bins = {}
            max_property_value = df_filtered[prop_type].max()
            
            # Define bin ranges based on property type
            if prop_type == "Rotatable_Bond_Count":
                bin_ranges = range(0, int(max_property_value) + 2, 2)  # Every 2 rotatable bonds
            elif prop_type == "Heavy_Atom_Count":
                bin_ranges = range(0, int(max_property_value) + 10, 10)  # Every 10 heavy atoms
            elif prop_type == "Molecular_Weight":
                bin_ranges = range(0, int(max_property_value) + 100, 100)  # Every 100 in molecular weight
            else:
                bin_ranges = range(0, int(max_property_value) + 5, 5)  # Default binning
            
            # Create empty data containers for plotting
            bin_centers = []
            for j in range(len(bin_ranges) - 1):
                bin_min = bin_ranges[j]
                bin_max = bin_ranges[j+1]
                bin_center = (bin_min + bin_max) / 2
                bin_centers.append(bin_center)
                property_bins[bin_center] = {'AlphaFold3': {'smiles': [], 'ccd': []}, 
                                           'Boltz1': {'smiles': [], 'ccd': []}}
            
            # Group data by property bins
            for _, row in df_filtered.iterrows():
                property_value = row[prop_type]
                model_type = row['MODEL_TYPE']
                
                # Skip if model type not of interest or missing LRMSD values
                if model_type not in ['AlphaFold3', 'Boltz1']:
                    continue
                
                # Find appropriate bin
                for j in range(len(bin_ranges) - 1):
                    bin_min = bin_ranges[j]
                    bin_max = bin_ranges[j+1]
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
                  linewidth=2.5, markersize=8, label='AF3')
            ax.plot(x_values, boltz_means, 'o-', color='blue', 
                  linewidth=2.5, markersize=8, label='Boltz-1')
            
            # Set axis labels
            property_label = property_labels.get(prop_type, prop_type)
            ax.set_xlabel(property_label, fontsize=PlotConfig.AXIS_LABEL_SIZE)
            
            # Only add y-label for the first subplot
            if i == 0:
                ax.set_ylabel('LRMSD (Å)', fontsize=PlotConfig.AXIS_LABEL_SIZE)
            
            # Set same y-axis limits for all plots based on molecule type
            if molecule_type == "MOLECULAR GLUE":
                ax.set_ylim(bottom=0, top=55)  # Higher y-axis limit for molecular glue
            else:
                ax.set_ylim(bottom=0, top=45)  # For PROTAC or other types
            
            # Increase tick label size
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend only to the first subplot
            if i == 0:
                ax.legend(fontsize=12, loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save:
            filename = f'{molecule_type.lower()}_lrmsd_combined_properties'.lower()
            self.save_plot(fig, filename, save_path=output_path)
            
        return fig, axes 