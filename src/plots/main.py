import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
from config import PlotConfig
from data_loader import DataLoader
from horizontal_bars import HorizontalBarPlotter
from ptm_plotter import PTMPlotter
from comparison_plotter import ComparisonPlotter
from training_cutoff_plotter import TrainingCutoffPlotter
from hal_comparison import HALComparisonPlotter
from poi_e3l_plotter import POI_E3LPlotter
from property_plotter import PropertyPlotter
from rmsd_complex_isolated import RMSDComplexIsolatedPlotter
from utils import categorize_by_cutoffs, create_plot_filename

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
        self.df_boltz1_agg = None # New attribute for Boltz1 aggregated data
        self.df_combined = None # Ensure df_combined is initialized
        
        # Initialize plotters
        self.bar_plotter = HorizontalBarPlotter()
        self.ptm_plotter = PTMPlotter()
        self.comparison_plotter = ComparisonPlotter()
        self.training_cutoff_plotter = TrainingCutoffPlotter()
        self.hal_comparison_plotter = HALComparisonPlotter()
        self.poi_e3l_plotter = POI_E3LPlotter()
        self.property_plotter = PropertyPlotter()
        self.rmsd_complex_isolated_plotter = RMSDComplexIsolatedPlotter()
        
        # Set up cutoffs (will be calculated after data load)
        self.cutoff_protac = None
        self.cutoff_molecular_glue = None
        self.cutoff_ptm_protac = None # This is for the categorization logic, not directly the threshold line
        self.cutoff_ptm_molecular_glue = None # This is for the categorization logic, not directly the threshold line
        
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
            
            # Calculate cutoffs for PROTACs using centralized method
            self.cutoff_protac = DataLoader.calculate_classification_cutoffs_from_af3_aggregated_data(
                self.df_af3_agg, molecule_type="PROTAC"
            )
            if self.cutoff_protac:
                print(f"✓ PROTAC RMSD cutoffs calculated: {[f'{c:.2f}' for c in self.cutoff_protac]}")
            else:
                print("Warning: Could not calculate PROTAC RMSD cutoffs from AlphaFold3 data")
            
            # Calculate pTM cutoffs for PROTACs (using existing manual logic for now since centralized method is for RMSD)
            protac_mask = self.df_af3_agg['TYPE'] == 'PROTAC'
            ccd_ptm_mean_protac = self.df_af3_agg.loc[protac_mask, 'CCD_PTM_mean'].dropna()
            
            if len(ccd_ptm_mean_protac) > 0:
                self.cutoff_ptm_protac = [
                    np.percentile(ccd_ptm_mean_protac, 20),
                    np.percentile(ccd_ptm_mean_protac, 40),
                    np.percentile(ccd_ptm_mean_protac, 60),
                    np.percentile(ccd_ptm_mean_protac, 80)
                ]
            
            # Calculate cutoffs for Molecular Glues using centralized method
            self.cutoff_molecular_glue = DataLoader.calculate_classification_cutoffs_from_af3_aggregated_data(
                self.df_af3_agg, molecule_type="Molecular Glue"
            )
            if self.cutoff_molecular_glue:
                print(f"✓ Molecular Glue RMSD cutoffs calculated: {[f'{c:.2f}' for c in self.cutoff_molecular_glue]}")
            else:
                print("Warning: Could not calculate Molecular Glue RMSD cutoffs from AlphaFold3 data")
            
            # Calculate pTM cutoffs for Molecular Glues (using existing manual logic for now since centralized method is for RMSD)
            molecular_glue_mask = self.df_af3_agg['TYPE'] == 'Molecular Glue'
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
            if self.df_boltz1 is not None:
                self.df_boltz1_agg = DataLoader.aggregate_by_pdb_id(self.df_boltz1) # Aggregate Boltz1 data
                print("✓ Boltz1 data loaded and aggregated successfully")
            else:
                print("Warning: Boltz1 data could not be loaded.")
            
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
        
        valid_figs = [fig for fig in figs if fig is not None]
        if not valid_figs:
            print(f"Warning: No valid figures to save for {base_name}")
            return saved_paths
            
        for i, fig in enumerate(valid_figs):
            # Create a descriptive filename
            if len(valid_figs) == 1:
                # Single figure - no page numbering needed
                filename = f"{base_name}_{timestamp}.png"
            else:
                # Multiple figures - add page numbering
                filename = f"{base_name}_page{i+1}of{len(valid_figs)}_{timestamp}.png"
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
        """Generate horizontal bar plots with mean and standard deviation for AF3 and Boltz1."""
        if self.df_af3_agg is None:
            print("Error: AF3 aggregated data not loaded. Please load data first.")
            # Optionally, check if Boltz1 data is available and proceed if only that is needed
            # For now, we require AF3 for cutoffs.
            return
        
        # Get user input for plot parameters
        print("\nHorizontal Bar Plot Settings:")
        molecule_type = input("Molecule type (PROTAC/Molecular Glue) [PROTAC]: ").strip() or "PROTAC"
        
        # Select cutoffs based on molecule type (derived from AF3 data in load_data)
        # These cutoffs will be used for both AF3 and Boltz1 plots for consistency
        af3_derived_cutoffs = self.cutoff_protac if molecule_type == "PROTAC" else self.cutoff_molecular_glue
        
        final_plot_cutoffs = af3_derived_cutoffs
        if final_plot_cutoffs is None:
            default_cutoffs = [2.0, 4.0, 6.0, 8.0] # Define a sensible default
            print(f"Warning: Cutoffs for {molecule_type} could not be derived from AF3 data during load. Using default cutoffs: {default_cutoffs}.")
            final_plot_cutoffs = default_cutoffs
        
        # Ask for threshold display
        add_threshold = input("Add threshold lines? (y/n) [y]: ").strip().lower() != 'n'
        
        # Plot for AlphaFold3 data
        print("\nGenerating AlphaFold3 horizontal bar plots...")
        try:
            figs_af3, _ = self.bar_plotter.plot_bars(
                self.df_af3_agg,
                molecule_type=molecule_type,
                classification_cutoff=final_plot_cutoffs, # Use the final determined cutoffs
                add_threshold=add_threshold,
                threshold_values=[4, 0.23, 4], # Default thresholds for RMSD, DockQ, PROTAC_RMSD
                show_y_labels_on_all=True,
                max_structures_per_plot=20,
                save=False, # Handled by self.save_plots
                smiles_color=PlotConfig.SMILES_PRIMARY, # AF3 specific color
                ccd_color=PlotConfig.CCD_PRIMARY # AF3 specific color
            )
            filename = create_plot_filename('bars', model_type='af3', molecule_type=molecule_type)
            self.save_plots(figs_af3, filename)
        except Exception as e:
            print(f"Error generating AlphaFold3 horizontal bar plots: {e}")

        # Plot for Boltz1 data if available
        if self.df_boltz1_agg is not None:
            print("\nGenerating Boltz1 horizontal bar plots...")
            try:
                figs_boltz1, _ = self.bar_plotter.plot_bars(
                    self.df_boltz1_agg,
                    molecule_type=molecule_type,
                    classification_cutoff=final_plot_cutoffs, # Use the same final_plot_cutoffs as AF3
                    add_threshold=add_threshold,
                    threshold_values=[4, 0.23, 4], # Default thresholds
                    show_y_labels_on_all=True,
                    max_structures_per_plot=20,
                    save=False, # Handled by self.save_plots
                    smiles_color=PlotConfig.BOLTZ1_SMILES_COLOR, # Boltz1 specific color
                    ccd_color=PlotConfig.BOLTZ1_CCD_COLOR # Boltz1 specific color
                )
                filename = create_plot_filename('bars', model_type='boltz1', molecule_type=molecule_type)
                self.save_plots(figs_boltz1, filename)
            except Exception as e:
                print(f"Error generating Boltz1 horizontal bar plots: {e}")
        else:
            print("\nSkipping Boltz1 horizontal bar plots: Boltz1 aggregated data not loaded.")
    
    def plot_rmsd_horizontal_bars(self):
        """Generate RMSD, iRMSD, PROTAC_RMSD horizontal bar plots."""
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
        """Generate pTM and ipTM horizontal bar plots for AF3 and optionally Boltz1 data."""
        if self.df_af3_agg is None and self.df_boltz1_agg is None:
            print("Error: No aggregated data (AF3 or Boltz1) loaded. Please load data first.")
            return
        
        print("\npTM/ipTM Plot Settings:")
        molecule_type = input("Molecule type (PROTAC/Molecular Glue) [PROTAC]: ").strip() or "PROTAC"
        
        add_threshold = input("Add threshold lines? (y/n) [y]: ").strip().lower() != 'n'
        ptm_threshold_val = 0.5
        iptm_threshold_val = 0.6 

        if add_threshold:
            ptm_threshold_input = input(f"PTM threshold value [{ptm_threshold_val}]: ").strip()
            if ptm_threshold_input:
                try:
                    ptm_threshold_val = float(ptm_threshold_input)
                except ValueError:
                    print(f"Invalid PTM threshold value. Using default: {ptm_threshold_val}")
            
            iptm_threshold_input = input(f"ipTM threshold value [{iptm_threshold_val}]: ").strip()
            if iptm_threshold_input:
                try:
                    iptm_threshold_val = float(iptm_threshold_input)
                except ValueError:
                    print(f"Invalid ipTM threshold value. Using default: {iptm_threshold_val}")

        max_structures_input = input("Max structures per plot page [17]: ").strip()
        try:
            max_structures = int(max_structures_input) if max_structures_input else 17
        except ValueError:
            print("Invalid number for max structures. Using default: 17")
            max_structures = 17
        
        # Plot for AlphaFold3 data
        if self.df_af3_agg is not None:
            print("\nGenerating AlphaFold3 pTM and ipTM plots...")
            try:
                figs_af3, _ = self.ptm_plotter.plot_ptm_bars(
                    self.df_af3_agg,
                    molecule_type=molecule_type,
                    data_source="af3",
                    add_threshold=add_threshold,
                    ptm_threshold_value=ptm_threshold_val,
                    iptm_threshold_value=iptm_threshold_val,
                    show_y_labels_on_all=True,
                    max_structures_per_plot=max_structures,
                    save=False 
                )
                
                if figs_af3:
                    filename = create_plot_filename('ptm', model_type='af3', molecule_type=molecule_type)
                    self.save_plots(figs_af3, filename)
                else:
                    print(f"No AlphaFold3 PTM/ipTM plots were generated for {molecule_type}.")
                
            except Exception as e:
                print(f"Error during AlphaFold3 PTM/ipTM plot generation: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping AlphaFold3 pTM/ipTM plots: AF3 aggregated data not loaded.")

        # Plot for Boltz1 data
        if self.df_boltz1_agg is not None:
            print("\nGenerating Boltz1 pTM and ipTM plots...")
            try:
                figs_boltz1, _ = self.ptm_plotter.plot_ptm_bars(
                    self.df_boltz1_agg,
                    molecule_type=molecule_type,
                    data_source="boltz1",
                    add_threshold=add_threshold,
                    ptm_threshold_value=ptm_threshold_val,
                    iptm_threshold_value=iptm_threshold_val,
                    show_y_labels_on_all=True,
                    max_structures_per_plot=max_structures,
                    save=False
                )
                
                if figs_boltz1:
                    filename = create_plot_filename('ptm', model_type='boltz1', molecule_type=molecule_type)
                    self.save_plots(figs_boltz1, filename)
                else:
                    print(f"No Boltz1 PTM/ipTM plots were generated for {molecule_type}.")
            
            except Exception as e:
                print(f"Error during Boltz1 PTM/ipTM plot generation: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping Boltz1 pTM/ipTM plots: Boltz1 aggregated data not loaded.")
    
    def plot_comparison(self, comparison_type="boltz1"):
        """
        Compare AF3 results with other methods.
        
        This function automatically generates comparison plots between AlphaFold3 and Boltz1 models
        for ALL metrics (RMSD, DOCKQ, PROTAC_RMSD, PTM) without user prompts.
        
        Args:
            comparison_type: Type of comparison to make ("boltz1" currently supported)
        """
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
        
        # Get user input for plot parameters
        print("\nComparison Plot Settings:")
        
        # Select molecule type (PROTAC or Molecular Glue)
        molecule_type_input = input("Molecule type (PROTAC/Molecular Glue) [PROTAC]: ").strip() or "PROTAC"
        if molecule_type_input in ['PROTAC', 'Molecular Glue']:
            molecule_type = molecule_type_input
        else:
            molecule_type = 'PROTAC'  # Default fallback
        
        # Ask for comparison type (general or seed-specific)
        comparison_choice = input("Comparison type (1=General, 2=Seed-specific) [1]: ").strip()
        is_seed_specific = comparison_choice == '2'
        
        # If seed-specific, ask for the seed
        specific_seed = None
        if is_seed_specific:
            seed_input = input("Specific seed to filter by [42]: ").strip()
            specific_seed = int(seed_input) if seed_input and seed_input.isdigit() else 42
        
        # Define all metrics to generate
        all_metrics = [
            {'name': 'RMSD', 'threshold': 4.0, 'add_threshold': True},
            {'name': 'DOCKQ', 'threshold': 0.23, 'add_threshold': True},
            {'name': 'PROTAC_RMSD', 'threshold': 4.0, 'add_threshold': True},
            {'name': 'PTM', 'threshold': 0.8, 'add_threshold': False}  # No threshold for PTM
        ]
        
        try:
            # Generate comparison plots for ALL metrics
            for metric_info in all_metrics:
                metric_type = metric_info['name']
                threshold_value = metric_info['threshold']
                add_threshold = metric_info['add_threshold']
                
                print(f"\nGenerating {'Seed-specific' if is_seed_specific else 'Mean'} {metric_type} comparison between AlphaFold3 and Boltz1 for {molecule_type}...")
                
                # Use the unified plot_mean_comparison method with optional specific_seed parameter
                fig, ax = self.comparison_plotter.plot_mean_comparison(
                    df=self.df_combined,
                    metric_type=metric_type,
                    add_threshold=add_threshold,
                    threshold_value=threshold_value,
                    save=False,
                    molecule_type=molecule_type,
                    specific_seed=specific_seed
                )
                
                if fig is not None:
                    # Use appropriate filename based on comparison type
                    if is_seed_specific:
                        filename = create_plot_filename('comparison', molecule_type=molecule_type, 
                                                      metric_type=metric_type, comparison_type='seed', 
                                                      seed=specific_seed)
                        self.save_plots([fig], filename)
                    else:
                        filename = create_plot_filename('comparison', molecule_type=molecule_type, 
                                                      metric_type=metric_type, comparison_type='mean')
                        self.save_plots([fig], filename)
                else:
                    print(f"Warning: Failed to generate {'seed-specific' if is_seed_specific else 'mean'} {metric_type} comparison plot")
            
            print(f"\n✓ Generated comparison plots for all metrics: {', '.join([m['name'] for m in all_metrics])}")
            
            # Generate the horizontal legend as a separate figure
            try:
                print("\nGenerating horizontal legend for comparison plots...")
                legend_fig = self.comparison_plotter.create_horizontal_legend(
                    width=6, 
                    height=1, 
                    save=False,
                    filename="af3_vs_boltz1_legend"
                )
                if legend_fig is not None:
                    filename = create_plot_filename('comparison', legend='legend')
                    self.save_plots([legend_fig], filename)
                    print("✓ Horizontal legend generated successfully")
                else:
                    print("Warning: Failed to generate horizontal legend")
            except Exception as legend_error:
                print(f"Error generating horizontal legend: {legend_error}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def plot_hal_comparison(self):
        """
        Generate plots comparing HAL (No Ligand) results with AlphaFold3 and Boltz1 models.
        """
        print("\nHAL vs Model Comparison Plot Settings:")
        
        # Check for required data files
        hal_file_path = '../../data/hal_04732948/hal_results.csv'
        combined_file_path = '../../data/af3_results/combined_results.csv'
        
        if not os.path.exists(hal_file_path):
            print(f"Error: HAL results file not found at: {hal_file_path}")
            return
            
        if not os.path.exists(combined_file_path):
            print(f"Error: Combined results file not found at: {combined_file_path}")
            return
        
        # Ask for threshold display
        add_threshold = input("Add threshold line? (y/n) [y]: ").strip().lower() != 'n'
        threshold_value = 0.23  # Default DockQ threshold
        
        if add_threshold:
            threshold_input = input(f"Threshold value [{threshold_value}]: ").strip()
            if threshold_input:
                try:
                    threshold_value = float(threshold_input)
                except ValueError:
                    print(f"Invalid threshold value. Using default: {threshold_value}")
        
        try:
            print("\nGenerating HAL vs Model comparison plots...")
            
            # Generate both AF3 and Boltz1 comparisons with legends
            af3_fig, boltz1_fig, af3_legend_fig, boltz1_legend_fig = self.hal_comparison_plotter.plot_both_comparisons(
                hal_file_path=hal_file_path,
                combined_file_path=combined_file_path,
                add_threshold=add_threshold,
                threshold_value=threshold_value,
                save=False,  # Handled by self.save_plots
                debug=False
            )
            
            # Save the main comparison plots
            if af3_fig is not None:
                filename = create_plot_filename('hal', model_type='af3', metric_type='dockq')
                self.save_plots([af3_fig], filename)
                print("✓ AlphaFold3 vs HAL comparison plot generated successfully")
            else:
                print("Warning: Failed to generate AlphaFold3 vs HAL comparison plot")
            
            if boltz1_fig is not None:
                filename = create_plot_filename('hal', model_type='boltz1', metric_type='dockq')
                self.save_plots([boltz1_fig], filename)
                print("✓ Boltz1 vs HAL comparison plot generated successfully")
            else:
                print("Warning: Failed to generate Boltz1 vs HAL comparison plot")
            
            # Save the legend plots
            if af3_legend_fig is not None:
                filename = create_plot_filename('hal', model_type='af3', legend='legend')
                self.save_plots([af3_legend_fig], filename)
                print("✓ AlphaFold3 legend generated successfully")
            else:
                print("Warning: Failed to generate AlphaFold3 legend")
                
            if boltz1_legend_fig is not None:
                filename = create_plot_filename('hal', model_type='boltz1', legend='legend')
                self.save_plots([boltz1_legend_fig], filename)
                print("✓ Boltz1 legend generated successfully")
            else:
                print("Warning: Failed to generate Boltz1 legend")
            
            print(f"\n✓ HAL vs Model comparison plots generated successfully")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def plot_training_cutoff(self):
        """
        Generate plots comparing model performance on structures from before
        and after the 2021-09-30 training cutoff date.
        """
        if not hasattr(self, 'df_combined') or self.df_combined is None:
            print("\nError: Combined results data not found.")
            print("Please ensure the file exists at: ../../data/af3_results/combined_results.csv")
            print("This file should contain both AlphaFold3 and Boltz1 results with MODEL_TYPE and RELEASE_DATE columns.")
            return
        
        print("\nTraining Cutoff Comparison Plot Settings:")
        
        # Determine which molecule type column is available
        molecule_type_col = 'MOLECULE_TYPE' if 'MOLECULE_TYPE' in self.df_combined.columns else 'TYPE'
        
        # Get available molecule types
        if molecule_type_col in self.df_combined.columns:
            available_types = sorted(self.df_combined[molecule_type_col].unique())
            print(f"Available molecule types: {', '.join(available_types)}")
            default_type = "PROTAC" if "PROTAC" in available_types else available_types[0]
        else:
            available_types = ["PROTAC", "Molecular Glue"]
            default_type = "PROTAC"
            print("Warning: No molecule type column found in data")
        
        # Select molecule type
        molecule_type_input = input(f"Molecule type ({'/'.join(available_types)}) [{default_type}]: ").strip() or default_type
        if molecule_type_input in available_types:
            molecule_type = molecule_type_input
        else:
            print(f"Invalid molecule type. Using default: {default_type}")
            molecule_type = default_type
        
        # Get available model types
        available_models = sorted(self.df_combined['MODEL_TYPE'].unique())
        default_model = "AlphaFold3" if "AlphaFold3" in available_models else available_models[0]
        print(f"Available model types: {', '.join(available_models)}")
        
        # Select model type - NOW ASKS IF USER WANTS TO COMPARE OR PLOT SINGLE
        plot_individual_model = True
        model_to_plot = default_model

        if "AlphaFold3" in available_models and "Boltz1" in available_models:
            user_choice = input(f"Compare AlphaFold3 and Boltz1 (y) or plot a single model (n)? [y]: ").strip().lower()
            if user_choice != 'n':
                plot_individual_model = False
            else:
                model_type_input = input(f"Model type to plot ({'/'.join(available_models)}) [{default_model}]: ").strip() or default_model
                if model_type_input in available_models:
                    model_to_plot = model_type_input
                else:
                    print(f"Invalid model type. Using default: {default_model}")
                    model_to_plot = default_model
        else:
            # If not both models are available for comparison, just plot the default/selected one
            model_type_input = input(f"Model type ({'/'.join(available_models)}) [{default_model}]: ").strip() or default_model
            if model_type_input in available_models:
                model_to_plot = model_type_input
            else:
                print(f"Invalid model type. Using default: {default_model}")
                model_to_plot = default_model
        
        # Generate plots for all metric types automatically
        metric_types = ['RMSD', 'DOCKQ', 'PROTAC_RMSD', 'PTM']
        print(f"Generating training cutoff plots for all metrics: {', '.join(metric_types)}")
        
        # Define metric configurations
        metric_configs = {
            'RMSD': {'threshold_value': 4.0, 'add_threshold': True},
            'DOCKQ': {'threshold_value': 0.23, 'add_threshold': True},
            'PROTAC_RMSD': {'threshold_value': 4.0, 'add_threshold': True},
            'PTM': {'threshold_value': 0.8, 'add_threshold': False}  # No threshold for PTM
        }
        
        try:
            for metric_type in metric_types:
                config = metric_configs[metric_type]
                threshold_value = config['threshold_value']
                add_threshold = config['add_threshold']
                
                print(f"\nGenerating plots for {metric_type}...")
                
                if plot_individual_model:
                    # Plot a single selected model
                    print(f"Generating {model_to_plot} training cutoff comparison for {metric_type} ({molecule_type})...")
                    filename = create_plot_filename(
                        'training_cutoff', model_type=model_to_plot, 
                        molecule_type=molecule_type, metric_type=metric_type
                    )
                    self.save_plots([fig], filename)
                else:
                    # Compare AlphaFold3 and Boltz1 with shared Y-axis
                    models_to_compare = ["AlphaFold3", "Boltz1"]
                    shared_ylim = None
                    
                    for model_name in models_to_compare:
                        if model_name not in available_models:
                            print(f"Warning: Model {model_name} not found in data. Skipping.")
                            continue

                        print(f"Generating {model_name} training cutoff comparison for {metric_type} ({molecule_type})...")
                        
                        fig, ax, current_ylim = self.training_cutoff_plotter.plot_training_cutoff_comparison(
                            df=self.df_combined,
                            metric_type=metric_type,
                            model_type=model_name,
                            add_threshold=add_threshold,
                            threshold_value=threshold_value,
                            save=False,
                            molecule_type=molecule_type,
                            fixed_ylim=shared_ylim 
                        )
                        
                        if fig is not None:
                            if shared_ylim is None: # Capture ylim from the first plot
                                shared_ylim = current_ylim
                            
                            filename = create_plot_filename(
                                'training_cutoff', model_type=model_name, 
                                molecule_type=molecule_type, metric_type=metric_type, 
                                comparison_type='compared'
                            )
                            self.save_plots([fig], filename)
                        else:
                            print(f"Warning: Failed to generate training cutoff comparison plot for {model_name} - {metric_type}")
            
            print(f"\n✓ Generated training cutoff plots for all metrics: {', '.join(metric_types)}")
            
            # Generate horizontal legends for both model types
            try:
                print("\nGenerating horizontal legends for training cutoff plots...")
                
                # Generate AlphaFold3 legend
                af3_legend_fig = self.training_cutoff_plotter.create_horizontal_legend(
                    model_type='AlphaFold3',
                    width=6, 
                    height=1, 
                    save=False,
                    filename="af3_training_cutoff_legend"
                )
                if af3_legend_fig is not None:
                    self.save_plots([af3_legend_fig], create_plot_filename('training_cutoff', model_type='AlphaFold3', legend='legend'))
                    print("✓ AlphaFold3 horizontal legend generated successfully")
                else:
                    print("Warning: Failed to generate AlphaFold3 horizontal legend")
                
                # Generate Boltz1 legend
                boltz1_legend_fig = self.training_cutoff_plotter.create_horizontal_legend(
                    model_type='Boltz1',
                    width=6, 
                    height=1, 
                    save=False,
                    filename="boltz1_training_cutoff_legend"
                )
                if boltz1_legend_fig is not None:
                    self.save_plots([boltz1_legend_fig], create_plot_filename('training_cutoff', model_type='Boltz1', legend='legend'))
                    print("✓ Boltz1 horizontal legend generated successfully")
                else:
                    print("Warning: Failed to generate Boltz1 horizontal legend")
                    
            except Exception as legend_error:
                print(f"Error generating horizontal legends for training cutoff plots: {legend_error}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def plot_poi_e3l_rmsd(self):
        """Generate POI and E3L RMSD and DockQ plots."""
        if not hasattr(self, 'df_combined') or self.df_combined is None:
            print("\nError: Combined results data not found.")
            print("Please ensure the file exists at: ../../data/af3_results/combined_results.csv")
            print("This file should contain both AlphaFold3 and Boltz1 results with MODEL_TYPE, SEED, SIMPLE_POI_NAME, and SIMPLE_E3_NAME columns.")
            return
        
        print("\nPOI and E3L Analysis Plot Settings:")
        
        allowed_types = ["PROTAC", "MOLECULAR GLUE"]
        print(f"Available molecule types: {', '.join(allowed_types)}")
        
        molecule_type_input = input(f"Molecule type (PROTAC/MOLECULAR GLUE) [PROTAC]: ").strip() or "PROTAC"
        
        if molecule_type_input.upper() == "MOLECULAR GLUE":
            molecule_type = "MOLECULAR GLUE"
        elif molecule_type_input.upper() == "PROTAC":
            molecule_type = "PROTAC"
        else:
            print(f"Invalid molecule type. Only {'/'.join(allowed_types)} are supported. Using default: PROTAC")
            molecule_type = "PROTAC"
            
        add_threshold = input(f"Add threshold lines? (y/n) [y]: ").strip().lower() != 'n'

        metrics_to_plot = [
            {'name': 'RMSD', 'threshold': 4.0},
            {'name': 'DockQ', 'threshold': 0.23}
        ]

        for metric_info in metrics_to_plot:
            metric_type = metric_info['name']
            threshold_value = metric_info['threshold']

            print(f"\n--- Generating plots for {metric_type} ---")
            
            print(f"Generating combined and individual plots for POI and E3L {metric_type} ({molecule_type})...")
            
            combined_fig, single_figs = self.poi_e3l_plotter.plot_all_model_grids(
                df=self.df_combined,
                model_types=['AlphaFold3', 'Boltz-1'],
                metric_type=metric_type,
                add_threshold=add_threshold,
                threshold_value=threshold_value,
                save=False,
                legend_position='lower right',
                molecule_type=molecule_type
            )
            
            if combined_fig is not None:
                filename = create_plot_filename(
                    'poi_e3l_grid', metric_type=metric_type, 
                    comparison_type='combined', molecule_type=molecule_type
                )
                self.save_plots([combined_fig], filename)
            
            if single_figs:
                for i, fig in enumerate(single_figs):
                    if fig is not None:
                        model_name = 'AlphaFold3' if i == 0 else 'Boltz-1'
                        filename = create_plot_filename(
                            'poi_e3l_single', model_type=model_name, 
                            metric_type=metric_type, molecule_type=molecule_type
                        )
                        self.save_plots([fig], filename)
            
            print(f"Generating individual vertical plots for POI and E3L {metric_type} ({molecule_type})...")
            
            try:
                vertical_figs = self.poi_e3l_plotter.plot_all_model_grids_vertical(
                    df=self.df_combined,
                    model_types=['AlphaFold3', 'Boltz-1'],
                    metric_type=metric_type,
                    add_threshold=add_threshold,
                    threshold_value=threshold_value,
                    save=False,
                    legend_position='lower right',
                    molecule_type=molecule_type
                )
                
                if vertical_figs:
                    model_names = ['AlphaFold3', 'Boltz-1']
                    for i, fig in enumerate(vertical_figs):
                        if fig is not None:
                            model_name = model_names[i]
                            filename = create_plot_filename(
                                'poi_e3l_single_vertical', model_type=model_name,
                                metric_type=metric_type, molecule_type=molecule_type
                            )
                            self.save_plots([fig], filename)
                            print(f"✓ Vertical plot generated for {model_name} with consistent y-axis scaling")
                        else:
                            print(f"Warning: Failed to generate vertical plot for {model_name}")
                else:
                    print("Warning: No vertical plots were generated")
            except Exception as e:
                print(f"Error generating vertical plots: {e}")

        try:
            print("\nGenerating horizontal and vertical legends for POI and E3L plots...")
            
            poi_legend_fig = self.poi_e3l_plotter.create_horizontal_legend(
                protein_type='POI', molecule_type=molecule_type, add_threshold=add_threshold,
                width=8, height=1, save=False, filename=None
            )
            if poi_legend_fig is not None:
                filename = create_plot_filename('poi_e3l_legend', molecule_type=molecule_type, protein_type='poi')
                self.save_plots([poi_legend_fig], filename)
                print("✓ POI horizontal legend generated successfully")
            
            poi_vertical_legend_fig = self.poi_e3l_plotter.create_vertical_legend(
                protein_type='POI', molecule_type=molecule_type, add_threshold=add_threshold,
                width=2, height=6, save=False, filename=None
            )
            if poi_vertical_legend_fig is not None:
                filename = create_plot_filename('poi_e3l_legend_vertical', molecule_type=molecule_type, protein_type='poi')
                self.save_plots([poi_vertical_legend_fig], filename)
                print("✓ POI vertical legend generated successfully")

            e3l_legend_fig = self.poi_e3l_plotter.create_horizontal_legend(
                protein_type='E3L', molecule_type=molecule_type, add_threshold=add_threshold,
                width=6, height=1, save=False, filename=None
            )
            if e3l_legend_fig is not None:
                filename = create_plot_filename('poi_e3l_legend', molecule_type=molecule_type, protein_type='e3l')
                self.save_plots([e3l_legend_fig], filename)
                print("✓ E3L horizontal legend generated successfully")

            e3l_vertical_legend_fig = self.poi_e3l_plotter.create_vertical_legend(
                protein_type='E3L', molecule_type=molecule_type, add_threshold=add_threshold,
                width=2, height=4, save=False, filename=None
            )
            if e3l_vertical_legend_fig is not None:
                filename = create_plot_filename('poi_e3l_legend_vertical', molecule_type=molecule_type, protein_type='e3l')
                self.save_plots([e3l_vertical_legend_fig], filename)
                print("✓ E3L vertical legend generated successfully")
                
        except Exception as legend_error:
            print(f"Error generating legends for POI/E3L plots: {legend_error}")
    
    def plot_property_vs_protac_rmsd(self):
        """Generate combined plot of molecular properties vs PROTAC RMSD."""
        if not hasattr(self, 'df_combined') or self.df_combined is None:
            print("\nError: Combined results data not found.")
            print("Please ensure the file exists at: ../../data/af3_results/combined_results.csv")
            print("This file should contain both AlphaFold3 and Boltz1 results with MODEL_TYPE, Rotatable_Bond_Count, Heavy_Atom_Count, and Molecular_Weight columns.")
            return
        
        print("\nMolecular Property vs PROTAC RMSD Plot Settings:")
        allowed_types = ["PROTAC", "MOLECULAR GLUE"]
        print(f"Available molecule types: {', '.join(allowed_types)}")
        
        # Get molecule type
        molecule_type_input = input(f"Molecule type (PROTAC/MOLECULAR GLUE) [PROTAC]: ").strip() or "PROTAC"
        
        # Validate molecule type
        if molecule_type_input.upper() == "MOLECULAR GLUE":
            molecule_type = "MOLECULAR GLUE"
        elif molecule_type_input.upper() == "PROTAC":
            molecule_type = "PROTAC"
        else:
            print(f"Invalid molecule type. Only {'/'.join(allowed_types)} are supported. Using default: PROTAC")
            molecule_type = "PROTAC"
            
        # Ask about bin customization
        custom_bins = input("\nWould you like to customize the bin sizes? (y/n) [n]: ").strip().lower() == 'y'
        
        # Set bin sizes
        mw_bin_size = 50
        hac_bin_size = 10
        rbc_bin_size = 5
        
        if custom_bins:
            # Get customized bin sizes
            print("\nEnter bin sizes (press Enter to use defaults):")
            mw_input = input("Molecular Weight bins (default 50 Da): ").strip()
            if mw_input and mw_input.isdigit() and int(mw_input) > 0:
                mw_bin_size = int(mw_input)
                
            hac_input = input("Heavy Atom Count bins (default 10 atoms): ").strip()
            if hac_input and hac_input.isdigit() and int(hac_input) > 0:
                hac_bin_size = int(hac_input)
                
            rbc_input = input("Rotatable Bond Count bins (default 10 bonds): ").strip()
            if rbc_input and rbc_input.isdigit() and int(rbc_input) > 0:
                rbc_bin_size = int(rbc_input)
        
        # Create bin settings dictionary
        bin_settings = {
            "Molecular_Weight": mw_bin_size,
            "Heavy_Atom_Count": hac_bin_size,
            "Rotatable_Bond_Count": rbc_bin_size
        }

        # Default to automatically calculating bin sizes for ~8 bins
        ranges = self.property_plotter.get_property_ranges(self.df_combined, molecule_type)
        target_bins = 8
        
        # Auto-calculate bin sizes for main properties
        if not custom_bins:
            if 'Molecular_Weight' in ranges:
                # Always use bin size of 50 for molecular weight as preferred
                mw_bin_size = 50
                
            if 'Heavy_Atom_Count' in ranges:
                hac_range = ranges['Heavy_Atom_Count'][1] - ranges['Heavy_Atom_Count'][0]
                hac_bin_size = max(5, round(hac_range / target_bins / 5) * 5)
                
            if 'Rotatable_Bond_Count' in ranges:
                rbc_range = ranges['Rotatable_Bond_Count'][1] - ranges['Rotatable_Bond_Count'][0]
                rbc_bin_size = max(2, round(rbc_range / target_bins / 2) * 2)
                
            # Update bin settings with auto-calculated values
            bin_settings = {
                "Molecular_Weight": mw_bin_size,
                "Heavy_Atom_Count": hac_bin_size,
                "Rotatable_Bond_Count": rbc_bin_size
            }
        
        # Initialize LogP, HBD, and HBA
        logp_bin_size = 1
        hbd_bin_size = 1
        hba_bin_size = 2
        
        customize_others = input("\nWould you like to customize LogP, H-Bond Donors, and H-Bond Acceptors bin sizes? (y/n) [n]: ").strip().lower() == 'y'
        
        if not customize_others:
            if 'LogP' in ranges:
                logp_range = ranges['LogP'][1] - ranges['LogP'][0]
                logp_bin_size = max(1, round(logp_range / target_bins))
            
            if 'HBD_Count' in ranges:
                hbd_range = ranges['HBD_Count'][1] - ranges['HBD_Count'][0]
                hbd_bin_size = max(1, round(hbd_range / target_bins))
        else:
            # Manual customization
            logp_input = input("LogP bins (default 1 unit): ").strip()
            if logp_input and logp_input.replace('.', '', 1).isdigit() and float(logp_input) > 0:
                logp_bin_size = float(logp_input)
                
            hbd_input = input("H-Bond Donors bins (default 1 unit): ").strip()
            if hbd_input and hbd_input.isdigit() and int(hbd_input) > 0:
                hbd_bin_size = int(hbd_input)
                
            hba_input = input("H-Bond Acceptors bins (default 2 units): ").strip()
            if hba_input and hba_input.isdigit() and int(hba_input) > 0:
                hba_bin_size = int(hba_input)
        
        # Add the additional bin settings
        bin_settings["LogP"] = logp_bin_size
        bin_settings["HBD_Count"] = hbd_bin_size
        bin_settings["HBA_Count"] = hba_bin_size
        
        # Generate the combined plot
        try:
            print(f"Generating combined property plot for {molecule_type}...")
            
            fig, axes = self.property_plotter.plot_combined_properties(
                df=self.df_combined,
                molecule_type=molecule_type,
                bin_settings=bin_settings,
                save=False
            )
            
            if fig is not None:
                # Save the plot with appropriate naming
                filename = create_plot_filename(
                    'property_protac_rmsd', molecule_type=molecule_type, 
                    comparison_type='combined'
                )
                self.save_plots([fig], filename)
            else:
                print(f"Warning: Failed to generate combined property plot")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def plot_rmsd_complex_isolated(self):
        """Generate RMSD plots for complex, POI, and E3."""
        if not hasattr(self, 'df_combined') or self.df_combined is None:
            print("\nError: Combined results data not found.")
            print("Please ensure the file exists at: ../../data/af3_results/combined_results.csv")
            print("This file should contain MODEL_TYPE, MOLECULE_TYPE/TYPE, and relevant RMSD columns:")
            print("CCD_RMSD, CCD_POI_RMSD, CCD_E3_RMSD, SMILES_RMSD, SMILES_POI_RMSD, SMILES_E3_RMSD")
            return

        print("\nRMSD Complex/Isolated Plot Settings:")
        
        # Determine which molecule type column is available
        molecule_type_col = 'MOLECULE_TYPE' if 'MOLECULE_TYPE' in self.df_combined.columns else 'TYPE'
        default_molecule_type = "PROTAC"
        if molecule_type_col in self.df_combined.columns:
            available_molecule_types = sorted(self.df_combined[molecule_type_col].unique())
            if not available_molecule_types:
                 available_molecule_types = ["PROTAC", "Molecular Glue"] # fallback
            default_molecule_type = "PROTAC" if "PROTAC" in available_molecule_types else available_molecule_types[0]
            print(f"Available molecule types: {', '.join(available_molecule_types)}")
        else:
            available_molecule_types = ["PROTAC", "Molecular Glue"]
            print(f"Warning: Molecule type column ('{molecule_type_col}') not found. Using default types.")

        molecule_type_input = input(f"Molecule type ({'/'.join(available_molecule_types)}) [{default_molecule_type}]: ").strip() or default_molecule_type
        if molecule_type_input not in available_molecule_types:
            print(f"Invalid molecule type. Using default: {default_molecule_type}")
            molecule_type = default_molecule_type
        else:
            molecule_type = molecule_type_input
            
        add_threshold_input = input(f"Add threshold line at {PlotConfig.DEFAULT_RMSD_THRESHOLD} Å? (y/n) [y]: ").strip().lower()
        add_threshold = add_threshold_input != 'n'
        threshold_value = PlotConfig.DEFAULT_RMSD_THRESHOLD

        # Use pre-calculated cutoffs from load_data() for consistency across all plotting modules
        if molecule_type == "PROTAC":
            rmsd_classification_cutoff = self.cutoff_protac
        elif molecule_type == "Molecular Glue":
            rmsd_classification_cutoff = self.cutoff_molecular_glue
        else:
            rmsd_classification_cutoff = None
        
        # Fallback to default cutoffs if pre-calculated cutoffs not available
        if rmsd_classification_cutoff is None:
            rmsd_classification_cutoff = [2.0, 4.0, 6.0, 8.0]
            print(f"Using default RMSD cutoffs: {rmsd_classification_cutoff}")
        else:
            print(f"Using pre-calculated AlphaFold3 CCD RMSD-derived cutoffs: {[f'{c:.2f}' for c in rmsd_classification_cutoff]}")

        model_types_to_plot = ['AlphaFold3', 'Boltz1'] # Or filter based on available in df_combined

        for model_type in model_types_to_plot:
            if model_type not in self.df_combined['MODEL_TYPE'].unique():
                print(f"Skipping {model_type}: not found in df_combined['MODEL_TYPE'].")
                continue

            print(f"\nGenerating plots for {model_type}, {molecule_type}...")
            for input_type in ['CCD', 'SMILES']:
                print(f"  Generating aggregated plot for {input_type}...")
                agg_fig, _ = self.rmsd_complex_isolated_plotter.plot_aggregated_rmsd_components(
                    df=self.df_combined,
                    model_type=model_type,
                    molecule_type=molecule_type,
                    input_type=input_type,
                    add_threshold=add_threshold,
                    threshold_value=threshold_value,
                    classification_cutoff=rmsd_classification_cutoff,
                    save=False
                )
                if agg_fig:
                    filename_agg = create_plot_filename(
                        'rmsd_agg', model_type=model_type, 
                        input_type=input_type, molecule_type=molecule_type
                    )
                    self.save_plots([agg_fig], filename_agg)
                else:
                    print(f"    Failed to generate aggregated plot for {model_type} {input_type}.")

                print(f"  Generating per-PDB plots for {input_type}...")
                perpdb_figs, _ = self.rmsd_complex_isolated_plotter.plot_per_pdb_rmsd_components(
                    df=self.df_combined,
                    model_type=model_type,
                    molecule_type=molecule_type,
                    input_type=input_type,
                    add_threshold=add_threshold,
                    threshold_value=threshold_value,
                    classification_cutoff=rmsd_classification_cutoff,
                    save=False
                )
                if perpdb_figs:
                    filename_perpdb_base = create_plot_filename(
                        'rmsd_perpdb', model_type=model_type,
                        input_type=input_type, molecule_type=molecule_type
                    )
                    # save_plots handles adding page numbers if multiple figures
                    self.save_plots(perpdb_figs, filename_perpdb_base)
                else:
                    print(f"    Failed to generate per-PDB plots for {model_type} {input_type}.")

        # Generate vertical legends for all model/input combinations
        try:
            print("\nGenerating vertical legends for RMSD Complex/Isolated plots...")
            
            for model_type in model_types_to_plot:
                if model_type not in self.df_combined['MODEL_TYPE'].unique():
                    continue
                    
                for input_type in ['CCD', 'SMILES']:
                    print(f"  Generating legend for {model_type} {input_type}...")
                    legend_fig = self.rmsd_complex_isolated_plotter.create_vertical_legend(
                        model_type=model_type,
                        input_type=input_type,
                        add_threshold=add_threshold,
                        width=2, 
                        height=4, 
                        save=False,
                        filename=None
                    )
                    if legend_fig is not None:
                        filename = create_plot_filename(
                            'rmsd_legend_vertical', model_type=model_type,
                            input_type=input_type, molecule_type=molecule_type
                        )
                        self.save_plots([legend_fig], filename)
                        print(f"    ✓ Legend generated for {model_type} {input_type}")
                    else:
                        print(f"    Warning: Failed to generate legend for {model_type} {input_type}")
                        
        except Exception as legend_error:
            print(f"Error generating vertical legends for RMSD Complex/Isolated plots: {legend_error}")

        print("\nRMSD Complex/Isolated plots generation complete.")

    def display_menu(self):
        """Display the main menu."""
        print("\n" + "="*50)
        print("PLOTTING TOOLKIT MENU".center(50))
        print("="*50)
        print("\nAvailable Plot Types:")
        print("1. AF3 vs Boltz1 Comparison")
        print("2. Training Cutoff Comparison")
        print("3. HAL vs Model Comparison")
        print("4. POI and E3L Analysis")
        print("5. RMSD Complex/Isolated")
        print("6. Horizontal Bars (Mean & Std Dev)")
        print("7. pTM and ipTM Plots")
        print("8. Property vs PROTAC RMSD Analysis")
        print("9. Generate All Plot Types")
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
                
            if choice == '9':
                # Generate all plot types
                self.plot_comparison("boltz1")
                self.plot_training_cutoff()
                self.plot_hal_comparison()
                self.plot_poi_e3l_rmsd()
                self.plot_rmsd_complex_isolated()
                self.plot_horizontal_bars()
                self.plot_ptm_bars()
                self.plot_property_vs_protac_rmsd()
                continue
            
            # Process comma-separated choices
            plot_choices = choice.split(',')
            
            for plot_choice in plot_choices:
                plot_choice = plot_choice.strip()
                
                if plot_choice == '1':
                    self.plot_comparison("boltz1")
                elif plot_choice == '2':
                    self.plot_training_cutoff()
                elif plot_choice == '3':
                    self.plot_hal_comparison()
                elif plot_choice == '4':
                    self.plot_poi_e3l_rmsd()
                elif plot_choice == '5':
                    self.plot_rmsd_complex_isolated()
                elif plot_choice == '6':
                    self.plot_horizontal_bars()
                elif plot_choice == '7':
                    self.plot_ptm_bars()
                elif plot_choice == '8':
                    self.plot_property_vs_protac_rmsd()
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