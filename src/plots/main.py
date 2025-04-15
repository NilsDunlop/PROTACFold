import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
from config import PlotConfig
from data_loader import DataLoader
from horizontal_bars import HorizontalBarPlotter
from rmsd_plotter import RMSDPlotter
from ptm_plotter import PTMPlotter
from comparison_plotter import ComparisonPlotter
from utils import categorize_by_cutoffs

class PlottingApp:
    """Main application for generating and saving plots."""
    
    def __init__(self):
        """Initialize the application."""
        # Configure non-interactive backend
        plt.switch_backend('agg')
        
        # Apply style settings
        PlotConfig.apply_style()
        
        # Data holders
        self.df_af3 = None
        self.df_af3_agg = None
        self.df_boltz1 = None
        
        # Initialize plotters
        self.bar_plotter = HorizontalBarPlotter()
        self.rmsd_plotter = RMSDPlotter()
        self.ptm_plotter = PTMPlotter()
        self.comparison_plotter = ComparisonPlotter()
        
        # Set up cutoffs (will be calculated after data load)
        self.cutoff_protac = None
        self.cutoff_molecular_glue = None
        self.cutoff_ptm_protac = None
        self.cutoff_ptm_molecular_glue = None
        
        # Set up output directory
        self.output_dir = os.path.abspath('../../data/plots')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Plots will be saved to: {self.output_dir}")
        
    def load_data(self):
        """Load and preprocess all datasets."""
        print("\nLoading data files...")
        
        # Load AF3 results
        try:
            self.df_af3 = DataLoader.load_data('../../data/af3_results/af3_results.csv')
            self.df_af3_agg = DataLoader.aggregate_by_pdb_id(self.df_af3)
            print("✓ AF3 data loaded successfully")
            
            # Calculate cutoffs for PROTACs
            protac_mask = self.df_af3_agg['TYPE'] == 'PROTAC'
            ccd_rmsd_mean_protac = self.df_af3_agg.loc[protac_mask, 'CCD_RMSD_mean'].dropna()
            
            if len(ccd_rmsd_mean_protac) > 0:
                self.cutoff_protac = [
                    np.percentile(ccd_rmsd_mean_protac, 20),
                    np.percentile(ccd_rmsd_mean_protac, 40),
                    np.percentile(ccd_rmsd_mean_protac, 60),
                    np.percentile(ccd_rmsd_mean_protac, 80)
                ]
            
            # Calculate pTM cutoffs for PROTACs
            ccd_ptm_mean_protac = self.df_af3_agg.loc[protac_mask, 'CCD_PTM_mean'].dropna()
            
            if len(ccd_ptm_mean_protac) > 0:
                self.cutoff_ptm_protac = [
                    np.percentile(ccd_ptm_mean_protac, 20),
                    np.percentile(ccd_ptm_mean_protac, 40),
                    np.percentile(ccd_ptm_mean_protac, 60),
                    np.percentile(ccd_ptm_mean_protac, 80)
                ]
            
            # Calculate cutoffs for Molecular Glues
            molecular_glue_mask = self.df_af3_agg['TYPE'] == 'Molecular Glue'
            ccd_rmsd_mean_molecular_glue = self.df_af3_agg.loc[molecular_glue_mask, 'CCD_RMSD_mean'].dropna()
            
            if len(ccd_rmsd_mean_molecular_glue) > 0:
                self.cutoff_molecular_glue = [
                    np.percentile(ccd_rmsd_mean_molecular_glue, 20),
                    np.percentile(ccd_rmsd_mean_molecular_glue, 40),
                    np.percentile(ccd_rmsd_mean_molecular_glue, 60),
                    np.percentile(ccd_rmsd_mean_molecular_glue, 80)
                ]
            
            # Calculate pTM cutoffs for Molecular Glues
            ccd_ptm_mean_molecular_glue = self.df_af3_agg.loc[molecular_glue_mask, 'CCD_PTM_mean'].dropna()
            
            if len(ccd_ptm_mean_molecular_glue) > 0:
                self.cutoff_ptm_molecular_glue = [
                    np.percentile(ccd_ptm_mean_molecular_glue, 20),
                    np.percentile(ccd_ptm_mean_molecular_glue, 40),
                    np.percentile(ccd_ptm_mean_molecular_glue, 60),
                    np.percentile(ccd_ptm_mean_molecular_glue, 80)
                ]
                
        except Exception as e:
            print(f"Error loading AF3 data: {e}")
        
        # Load Boltz1 results
        try:
            self.df_boltz1 = DataLoader.load_data('../../data/boltz1_results/boltz1_results.csv')
            print("✓ Boltz1 data loaded successfully")
            
            # Also try to load combined results if available
            try:
                self.df_combined = DataLoader.load_data('../../data/af3_results/combined_results.csv')
                print("✓ Combined results data loaded successfully")
            except Exception as e:
                self.df_combined = None
                print(f"Note: Combined results data not loaded: {e}")
                
        except Exception as e:
            print(f"Error loading Boltz1 data: {e}")
    
    def save_plots(self, figs, base_name):
        """Save all figures with a common base name and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []
        
        for i, fig in enumerate(figs):
            # Create a descriptive filename
            filename = f"{base_name}_{i+1}of{len(figs)}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the figure
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_paths.append(filepath)
            
            # Close the figure to free memory
            plt.close(fig)
        
        print(f"Saved {len(saved_paths)} plots to output directory")
        
        return saved_paths
    
    def open_output_folder(self):
        """Open the output folder in the system file explorer."""
        try:
            # Different commands for different operating systems
            if os.name == 'nt':  # Windows
                os.startfile(self.output_dir)
            elif os.name == 'posix':  # macOS, Linux
                if os.uname().sysname == 'Darwin':  # macOS
                    subprocess.call(['open', self.output_dir])
                else:  # Linux
                    subprocess.call(['xdg-open', self.output_dir])
            print(f"Opened output folder: {self.output_dir}")
        except Exception as e:
            print(f"Could not open output folder: {e}")
    
    def plot_horizontal_bars(self):
        """Generate horizontal bar plots with mean and standard deviation."""
        if self.df_af3_agg is None:
            print("Error: AF3 aggregated data not loaded. Please load data first.")
            return
        
        # Get user input for plot parameters
        print("\nHorizontal Bar Plot Settings:")
        molecule_type = input("Molecule type (PROTAC/Molecular Glue) [PROTAC]: ").strip() or "PROTAC"
        
        # Select cutoffs based on molecule type
        cutoffs = self.cutoff_protac if molecule_type == "PROTAC" else self.cutoff_molecular_glue
        
        # Ask for threshold display
        add_threshold = input("Add threshold lines? (y/n) [y]: ").strip().lower() != 'n'
        
        print("Generating horizontal bar plots...")
        try:
            figs, axes = self.bar_plotter.plot_bars(
                self.df_af3_agg,
                molecule_type=molecule_type,
                classification_cutoff=cutoffs,
                add_threshold=add_threshold,
                threshold_values=[0.23, 4, 4],
                show_y_labels_on_all=True,
                max_structures_per_plot=20,  # Fixed at 20 structures per plot
                save=False  # We'll handle saving ourselves
            )
            
            # Save the plots with appropriate naming
            self.save_plots(figs, f"horizontal_bars_{molecule_type}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    def plot_rmsd_horizontal_bars(self):
        """Generate RMSD, iRMSD, LRMSD horizontal bar plots."""
        if self.df_af3_agg is None:
            print("Error: AF3 aggregated data not loaded. Please load data first.")
            return
        
        # Get user input for plot parameters
        print("\nRMSD Plot Settings:")
        molecule_type = input("Molecule type (PROTAC/Molecular Glue) [PROTAC]: ").strip() or "PROTAC"
        
        # Select cutoffs based on molecule type
        cutoffs = self.cutoff_protac if molecule_type == "PROTAC" else self.cutoff_molecular_glue
        
        # Ask for threshold display
        add_threshold = input("Add threshold line? (y/n) [y]: ").strip().lower() != 'n'
        threshold_value = 2.0
        if add_threshold:
            threshold_input = input("Threshold value [2.0]: ").strip()
            if threshold_input:
                threshold_value = float(threshold_input)
        
        # Max structure per plot for RMSD plots
        max_structures = 17
        
        print("Generating RMSD plots...")
        try:
            figs, axes = self.rmsd_plotter.plot_rmsd_bars(
                self.df_af3_agg,
                molecule_type=molecule_type,
                classification_cutoff=cutoffs,
                add_threshold=add_threshold,
                threshold_value=threshold_value,
                show_y_labels_on_all=True,
                max_structures_per_plot=max_structures,
                save=False
            )
            
            # Save the plots with appropriate naming
            self.save_plots(figs, f"rmsd_comparison_{molecule_type}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_ptm_bars(self):
        """Generate pTM and ipTM horizontal bar plots."""
        if self.df_af3_agg is None:
            print("Error: AF3 aggregated data not loaded. Please load data first.")
            return
        
        # Get user input for plot parameters
        print("\npTM Plot Settings:")
        molecule_type = input("Molecule type (PROTAC/Molecular Glue) [PROTAC]: ").strip() or "PROTAC"
        
        # Select cutoffs based on molecule type
        cutoffs = self.cutoff_ptm_protac if molecule_type == "PROTAC" else self.cutoff_ptm_molecular_glue
        
        # Ask for threshold display
        add_threshold = input("Add threshold line? (y/n) [y]: ").strip().lower() != 'n'
        threshold_value = 0.5
        if add_threshold:
            threshold_input = input("Threshold value [0.5]: ").strip()
            if threshold_input:
                threshold_value = float(threshold_input)
        
        # Max structure per plot
        max_structures = 15
        
        print("Generating pTM and ipTM plots...")
        try:
            figs, axes = self.ptm_plotter.plot_ptm_bars(
                self.df_af3_agg,
                molecule_type=molecule_type,
                classification_cutoff=cutoffs,
                add_threshold=add_threshold,
                threshold_value=threshold_value,
                show_y_labels_on_all=True,
                max_structures_per_plot=max_structures,
                save=False
            )
            
            # Save the plots with appropriate naming
            self.save_plots(figs, f"ptm_iptm_comparison_{molecule_type}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_comparison(self, comparison_type="boltz1"):
        """Compare AF3 results with other methods."""
        if self.df_af3_agg is None:
            print("Error: AF3 aggregated data not loaded. Please load data first.")
            return
            
        if comparison_type == "boltz1" and self.df_boltz1 is None:
            print("Error: Boltz1 data not loaded. Please load data first.")
            return
        
        # Check for combined results data
        if not hasattr(self, 'df_combined') or self.df_combined is None:
            print("\nError: Combined results data not found.")
            print("Please ensure the file exists at: ../../data/af3_results/combined_results.csv")
            print("This file should contain both AlphaFold3 and Boltz1 results with MODEL_TYPE and SEED columns.")
            return
        
        print(f"\n{comparison_type.upper()} comparison plots:")
        
        # Default settings
        add_threshold = True
        threshold_value = 4.0  # Default for RMSD
        metric_type = 'RMSD'
        molecule_type = 'PROTAC'  # Default to PROTAC
        
        # Get user input for plot parameters
        print("\nComparison Plot Settings:")
        
        # Select molecule type (PROTAC or Molecular Glue)
        molecule_type_input = input("Molecule type (PROTAC/Molecular Glue) [PROTAC]: ").strip() or "PROTAC"
        if molecule_type_input in ['PROTAC', 'Molecular Glue']:
            molecule_type = molecule_type_input
        
        # Select metric type
        metric_type_input = input("Metric type (RMSD/DOCKQ) [RMSD]: ").strip().upper() or "RMSD"
        if metric_type_input in ['RMSD', 'DOCKQ']:
            metric_type = metric_type_input
        
        # Set appropriate threshold value based on metric type
        if metric_type == 'RMSD':
            threshold_value = 4.0
        elif metric_type == 'DOCKQ':
            threshold_value = 0.23
        
        # Ask for threshold display
        add_threshold_input = input(f"Add threshold line at {threshold_value}? (y/n) [y]: ").strip().lower()
        add_threshold = add_threshold_input != 'n'
        
        print(f"Generating AlphaFold3 vs Boltz1 comparison plots for {molecule_type} {metric_type}...")
        try:
            # Call plot_af3_vs_boltz1 with the correct parameters
            figs, axes = self.comparison_plotter.plot_af3_vs_boltz1(
                df=self.df_combined,
                metric_type=metric_type,
                add_threshold=add_threshold,
                threshold_value=threshold_value,
                show_y_labels_on_all=True,
                width=12,
                height=14,
                max_structures=25,
                save=False,
                molecule_type=molecule_type
            )
            
            # Save the plots with appropriate naming
            self.save_plots(figs, f"af3_vs_boltz1_{molecule_type.lower()}_{metric_type.lower()}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def display_menu(self):
        """Display the main menu."""
        print("\n" + "="*50)
        print("PLOTTING TOOLKIT MENU".center(50))
        print("="*50)
        print("\nAvailable Plot Types:")
        print("1. Horizontal Bars (Mean & Std Dev)")
        print("2. RMSD, iRMSD, LRMSD Plots")
        print("3. pTM and ipTM Plots")
        print("4. AF3 vs Boltz1 Comparison")
        print("5. Generate All Plot Types")
        print("\no. Open Output Folder")
        print("q. Quit")
        
        print("\nEnter plot numbers to generate (comma separated for multiple)")
        choice = input("Your choice: ").strip().lower()
        return choice
    
    def run(self):
        """Run the application."""
        print("Welcome to the Plotting Toolkit")
        
        # Load data on first run
        self.load_data()
        
        while True:
            choice = self.display_menu()
            
            if choice == 'q':
                print("Exiting. Thank you for using the Plotting Toolkit!")
                break
                
            if choice == 'o':
                self.open_output_folder()
                continue
                
            if choice == '5':
                # Generate all plot types
                self.plot_horizontal_bars()
                self.plot_rmsd_horizontal_bars()
                self.plot_ptm_bars()
                self.plot_comparison("boltz1")
                continue
            
            # Process comma-separated choices
            plot_choices = choice.split(',')
            
            for plot_choice in plot_choices:
                plot_choice = plot_choice.strip()
                
                if plot_choice == '1':
                    self.plot_horizontal_bars()
                elif plot_choice == '2':
                    self.plot_rmsd_horizontal_bars()
                elif plot_choice == '3':
                    self.plot_ptm_bars()
                elif plot_choice == '4':
                    self.plot_comparison("boltz1")
                elif not plot_choice:
                    continue
                else:
                    print(f"Invalid choice: {plot_choice}")
                    
                # Pause after each plot type when doing multiple
                if len(plot_choices) > 1 and plot_choice != plot_choices[-1].strip():
                    input("Press Enter to continue to the next plot type...")


if __name__ == "__main__":
    app = PlottingApp()
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()