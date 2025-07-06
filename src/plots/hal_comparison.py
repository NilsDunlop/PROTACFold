import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from base_plotter import BasePlotter
from config import PlotConfig
from utils import save_figure

class HALComparisonPlotter(BasePlotter):
    """
    Class for creating comparison plots between HAL (No Ligand) results 
    and AlphaFold3/Boltz-1 with CCD and SMILES ligands.
    """
    
    # Constants now imported from PlotConfig
    # Colors for different conditions
    AF3_CCD_COLOR = PlotConfig.CCD_PRIMARY
    AF3_SMILES_COLOR = PlotConfig.SMILES_PRIMARY
    
    BOLTZ1_CCD_COLOR = PlotConfig.BOLTZ1_CCD_COLOR
    BOLTZ1_SMILES_COLOR = PlotConfig.BOLTZ1_SMILES_COLOR
    
    # HAL (No Ligand) color - using a neutral gray
    HAL_COLOR = PlotConfig.HAL_COLOR

    def __init__(self, debug=False):
        """Initialize the HAL comparison plotter."""
        super().__init__()
        self.debug = debug
    
    def _debug_print(self, message):
        """Print debug information if debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] HALComparisonPlotter: {message}")

    def load_and_merge_data(self, hal_file_path, combined_file_path):
        """
        Load HAL results and combined results, then merge them by PDB_ID.
        
        Args:
            hal_file_path (str): Path to hal_results.csv
            combined_file_path (str): Path to combined_results.csv
            
        Returns:
            pd.DataFrame: Merged dataframe sorted by RELEASE_DATE
        """
        self._debug_print(f"Loading HAL data from: {hal_file_path}")
        self._debug_print(f"Loading combined data from: {combined_file_path}")
        
        try:
            hal_df = pd.read_csv(hal_file_path)
            self._debug_print(f"HAL data loaded, shape: {hal_df.shape}")
        except Exception as e:
            print(f"Error loading HAL data: {e}")
            return None
        
        try:
            combined_df = pd.read_csv(combined_file_path)
            self._debug_print(f"Combined data loaded, shape: {combined_df.shape}")
        except Exception as e:
            print(f"Error loading combined data: {e}")
            return None
        
        # Filter combined data to SEED=42 only
        combined_df_filtered = combined_df[combined_df['SEED'] == 42].copy()
        self._debug_print(f"After SEED=42 filter, combined data shape: {combined_df_filtered.shape}")
        
        required_hal_cols = ['PDB_ID', 'RELEASE_DATE', 'AF3_DIMERS_DOCKQ_SCORE']
        required_combined_cols = ['PDB_ID', 'MODEL_TYPE', 'CCD_DOCKQ_SCORE', 'SMILES_DOCKQ_SCORE']
        
        missing_hal_cols = [col for col in required_hal_cols if col not in hal_df.columns]
        missing_combined_cols = [col for col in required_combined_cols if col not in combined_df_filtered.columns]
        
        if missing_hal_cols:
            print(f"Error: Missing columns in HAL data: {missing_hal_cols}")
            return None
        if missing_combined_cols:
            print(f"Error: Missing columns in combined data: {missing_combined_cols}")
            return None
        
        hal_df['RELEASE_DATE'] = pd.to_datetime(hal_df['RELEASE_DATE'])
        
        hal_pdb_ids = set(hal_df['PDB_ID'].unique())
        combined_pdb_ids = set(combined_df_filtered['PDB_ID'].unique())
        
        common_pdb_ids = hal_pdb_ids.intersection(combined_pdb_ids)
        self._debug_print(f"Common PDB_IDs found: {len(common_pdb_ids)}")
        
        if not common_pdb_ids:
            print("Error: No common PDB_IDs found between HAL and combined datasets")
            return None
        
        hal_df_filtered = hal_df[hal_df['PDB_ID'].isin(common_pdb_ids)].copy()
        combined_df_final = combined_df_filtered[combined_df_filtered['PDB_ID'].isin(common_pdb_ids)].copy()
        
        # Merge HAL data with combined data
        af3_data = combined_df_final[combined_df_final['MODEL_TYPE'] == 'AlphaFold3'][['PDB_ID', 'CCD_DOCKQ_SCORE', 'SMILES_DOCKQ_SCORE']].copy()
        af3_data = af3_data.rename(columns={
            'CCD_DOCKQ_SCORE': 'AF3_CCD_DOCKQ_SCORE',
            'SMILES_DOCKQ_SCORE': 'AF3_SMILES_DOCKQ_SCORE'
        })
        
        boltz1_data = combined_df_final[combined_df_final['MODEL_TYPE'] == 'Boltz1'][['PDB_ID', 'CCD_DOCKQ_SCORE', 'SMILES_DOCKQ_SCORE']].copy()
        boltz1_data = boltz1_data.rename(columns={
            'CCD_DOCKQ_SCORE': 'BOLTZ1_CCD_DOCKQ_SCORE',
            'SMILES_DOCKQ_SCORE': 'BOLTZ1_SMILES_DOCKQ_SCORE'
        })
        
        merged_df = hal_df_filtered[['PDB_ID', 'RELEASE_DATE', 'AF3_DIMERS_DOCKQ_SCORE']].copy()
        merged_df = merged_df.merge(af3_data, on='PDB_ID', how='inner')
        merged_df = merged_df.merge(boltz1_data, on='PDB_ID', how='inner')
        
        # Rename HAL column for clarity
        merged_df = merged_df.rename(columns={'AF3_DIMERS_DOCKQ_SCORE': 'HAL_DOCKQ_SCORE'})
        
        merged_df = merged_df.sort_values('RELEASE_DATE').reset_index(drop=True)
        
        self._debug_print(f"Final merged data shape: {merged_df.shape}")
        self._debug_print(f"Columns: {merged_df.columns.tolist()}")
        
        return merged_df

    def plot_hal_comparison(
        self, 
        merged_df, 
        model_type='AlphaFold3',
        add_threshold=True,
        threshold_value=None,
        width=None,
        height=None,
        save=True,
        debug=False
    ):
        """
        Create a bar plot comparing HAL (No Ligand) results with model results.
        
        Args:
            merged_df (pd.DataFrame): Merged dataframe with HAL and model data
            model_type (str): Model type to compare ('AlphaFold3' or 'Boltz1')
            add_threshold (bool): Whether to add a threshold line
            threshold_value (float): Value for the threshold line
            width (int): Figure width (overrides default if provided)
            height (int): Figure height (overrides default if provided)
            save (bool): Whether to save the plot
            debug (bool): Enable additional debugging output
        
        Returns:
            tuple: (fig, ax) The figure and axis objects
        """
        self.debug = debug or self.debug
        
        plot_width = width if width is not None else PlotConfig.HAL_PLOT_WIDTH
        plot_height = height if height is not None else PlotConfig.HAL_PLOT_HEIGHT

        self._debug_print(f"Starting plot_hal_comparison with model_type={model_type}")
        self._debug_print(f"Plot dimensions: width={plot_width}, height={plot_height}")
        
        if threshold_value is None:
            threshold_value = PlotConfig.DEFAULT_DOCKQ_THRESHOLD
            self._debug_print(f"Using default threshold value: {threshold_value}")
        
        if merged_df.empty:
            print("Error: Merged dataframe is empty")
            return None, None
        
        if model_type == 'AlphaFold3':
            ccd_col = 'AF3_CCD_DOCKQ_SCORE'
            smiles_col = 'AF3_SMILES_DOCKQ_SCORE'
            ccd_color = self.AF3_CCD_COLOR
            smiles_color = self.AF3_SMILES_COLOR
            ccd_label = 'AF3 CCD'
            smiles_label = 'AF3 SMILES'
        elif model_type == 'Boltz1':
            ccd_col = 'BOLTZ1_CCD_DOCKQ_SCORE'
            smiles_col = 'BOLTZ1_SMILES_DOCKQ_SCORE'
            ccd_color = self.BOLTZ1_CCD_COLOR
            smiles_color = self.BOLTZ1_SMILES_COLOR
            ccd_label = 'Boltz1 CCD'
            smiles_label = 'Boltz1 SMILES'
        else:
            print(f"Error: Unsupported model type '{model_type}'")
            return None, None
        
        hal_col = 'HAL_DOCKQ_SCORE'
        hal_color = self.HAL_COLOR
        hal_label = 'No Ligand'
        
        required_cols = [ccd_col, smiles_col, hal_col, 'PDB_ID']
        missing_cols = [col for col in required_cols if col not in merged_df.columns]
        if missing_cols:
            print(f"Error: Missing columns in dataframe: {missing_cols}")
            return None, None
        
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        
        n_structures = len(merged_df)
        
        # Create PDB ID labels with asterisks for post-2021-09-30 structures
        training_cutoff_date = pd.to_datetime('2021-09-30')
        pdb_ids = []
        for _, row in merged_df.iterrows():
            pdb_id = row['PDB_ID']
            release_date = pd.to_datetime(row['RELEASE_DATE'])
            if release_date > training_cutoff_date:
                pdb_ids.append(f"{pdb_id}*")
            else:
                pdb_ids.append(pdb_id)
        
        ccd_values = merged_df[ccd_col].values
        smiles_values = merged_df[smiles_col].values
        hal_values = merged_df[hal_col].values
        
        self._debug_print(f"Number of structures: {n_structures}")
        self._debug_print(f"CCD values range: {np.min(ccd_values):.3f} - {np.max(ccd_values):.3f}")
        self._debug_print(f"SMILES values range: {np.min(smiles_values):.3f} - {np.max(smiles_values):.3f}")
        self._debug_print(f"HAL values range: {np.min(hal_values):.3f} - {np.max(hal_values):.3f}")
        
        bar_width = PlotConfig.HAL_BAR_WIDTH
        spacing_factor = PlotConfig.HAL_BAR_SPACING_FACTOR
        
        # Create positions for each group of 3 bars with tighter spacing between PDB groups
        x_positions = np.arange(n_structures) * 0.8  # Reduce spacing between PDB groups
        ccd_positions = x_positions - bar_width * spacing_factor
        smiles_positions = x_positions
        hal_positions = x_positions + bar_width * spacing_factor
        
        bars_ccd = ax.bar(
            ccd_positions, ccd_values, bar_width,
            color=ccd_color, edgecolor=PlotConfig.BAR_EDGE_COLOR,
            linewidth=PlotConfig.BAR_EDGE_WIDTH, label=ccd_label
        )
        
        bars_smiles = ax.bar(
            smiles_positions, smiles_values, bar_width,
            color=smiles_color, edgecolor=PlotConfig.BAR_EDGE_COLOR,
            linewidth=PlotConfig.BAR_EDGE_WIDTH, label=smiles_label
        )
        
        bars_hal = ax.bar(
            hal_positions, hal_values, bar_width,
            color=hal_color, edgecolor=PlotConfig.BAR_EDGE_COLOR,
            linewidth=PlotConfig.BAR_EDGE_WIDTH, label=hal_label
        )
        
        if add_threshold:
            ax.axhline(
                y=threshold_value,
                color=PlotConfig.THRESHOLD_LINE_COLOR,
                linestyle=PlotConfig.THRESHOLD_LINE_STYLE,
                alpha=PlotConfig.THRESHOLD_LINE_ALPHA,
                linewidth=PlotConfig.THRESHOLD_LINE_WIDTH
            )
            self._debug_print(f"Added threshold line at y={threshold_value}")
        
        ax.set_xlabel('PDB ID', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('DockQ Score', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(pdb_ids, rotation=90, ha='center', fontsize=PlotConfig.TICK_LABEL_SIZE)
        
        # Set y-axis ticks with single digit precision
        y_ticks = np.arange(0, 1.1, 0.2)  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks])
        ax.tick_params(axis='y', which='major', labelsize=PlotConfig.TICK_LABEL_SIZE)
        
        ax.grid(axis='y', linestyle=PlotConfig.GRID_LINESTYLE, alpha=PlotConfig.GRID_ALPHA)
        
        # Set y-axis limits to fixed range 0-1 for consistent comparison
        ax.set_ylim(0, 1)
        
        # Set x-axis limits to reduce empty space at beginning and end
        x_margin = bar_width * spacing_factor * 2.5  # Small margin around the bars
        ax.set_xlim(x_positions[0] - x_margin, x_positions[-1] + x_margin)
        
        # No legend in the main plot as requested
        
        plt.tight_layout()
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('black')
        
        if save:
            filename = f"hal_comparison_{model_type.lower()}_dockq"
            if self.debug:
                filename += "_debug"
            save_figure(fig, filename)
            self._debug_print(f"Saved figure as {filename}")
        
        self._debug_print("Successfully completed plot_hal_comparison")
        return fig, ax

    def create_horizontal_legend(self, model_type='AlphaFold3', width=6, height=1, save=True, filename="hal_comparison_legend"):
        """
        Create a standalone horizontal legend figure for HAL comparison plots.
        
        Args:
            model_type (str): Model type ('AlphaFold3' or 'Boltz1') to determine colors
            width (float): Width of the legend figure
            height (float): Height of the legend figure 
            save (bool): Whether to save the figure
            filename (str): Filename for saving
            
        Returns:
            fig: The created legend figure
        """
        fig, ax = plt.subplots(figsize=(width, height))
        
        if model_type == 'AlphaFold3':
            colors = [self.AF3_CCD_COLOR, self.AF3_SMILES_COLOR, self.HAL_COLOR]
            labels = ['AF3 CCD', 'AF3 SMILES', 'No Ligand']
        elif model_type == 'Boltz1':
            colors = [self.BOLTZ1_CCD_COLOR, self.BOLTZ1_SMILES_COLOR, self.HAL_COLOR]
            labels = ['B1 CCD', 'B1 SMILES', 'No Ligand']
        else:
            # Fallback to AF3 colors
            colors = [self.AF3_CCD_COLOR, self.AF3_SMILES_COLOR, self.HAL_COLOR]
            labels = ['AF3 CCD', 'AF3 SMILES', 'No Ligand']
        
        legend_handles = []
        for label, color in zip(labels, colors):
            patch = plt.Rectangle(
                (0, 0), 1, 1,
                facecolor=color,
                edgecolor=PlotConfig.BAR_EDGE_COLOR,
                linewidth=PlotConfig.BAR_EDGE_WIDTH,
                label=label
            )
            legend_handles.append(patch)
        
        threshold_line = plt.Line2D(
            [0, 1], [0, 0],
            color=PlotConfig.THRESHOLD_LINE_COLOR,
            linestyle=PlotConfig.THRESHOLD_LINE_STYLE,
            linewidth=PlotConfig.THRESHOLD_LINE_WIDTH,
            label='Threshold'
        )
        legend_handles.append(threshold_line)
        
        legend = ax.legend(
            handles=legend_handles,
            loc='center',
            ncol=4,  # 4 columns for horizontal layout (3 bars + threshold)
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
        
        if save:
            save_figure(fig, filename)
            
        return fig

    def plot_both_comparisons(
        self,
        hal_file_path,
        combined_file_path,
        add_threshold=True,
        threshold_value=None,
        save=True,
        debug=False
    ):
        """
        Generate both AF3 vs HAL and Boltz1 vs HAL comparison plots.
        
        Args:
            hal_file_path (str): Path to hal_results.csv
            combined_file_path (str): Path to combined_results.csv
            add_threshold (bool): Whether to add threshold lines
            threshold_value (float): Threshold value (uses default if None)
            save (bool): Whether to save plots
            debug (bool): Enable debug output
            
        Returns:
            tuple: (af3_fig, boltz1_fig, af3_legend_fig, boltz1_legend_fig)
        """
        self.debug = debug or self.debug
        
        merged_df = self.load_and_merge_data(hal_file_path, combined_file_path)
        if merged_df is None:
            print("Error: Could not load and merge data")
            return None, None, None, None
        
        print("Generating AlphaFold3 vs HAL comparison...")
        af3_fig, _ = self.plot_hal_comparison(
            merged_df=merged_df,
            model_type='AlphaFold3',
            add_threshold=add_threshold,
            threshold_value=threshold_value,
            save=save,
            debug=debug
        )
        
        print("Generating Boltz1 vs HAL comparison...")
        boltz1_fig, _ = self.plot_hal_comparison(
            merged_df=merged_df,
            model_type='Boltz1',
            add_threshold=add_threshold,
            threshold_value=threshold_value,
            save=save,
            debug=debug
        )
        
        print("Generating legends...")
        af3_legend_fig = self.create_horizontal_legend(
            model_type='AlphaFold3',
            save=save,
            filename="hal_comparison_legend_alphafold3"
        )
        
        boltz1_legend_fig = self.create_horizontal_legend(
            model_type='Boltz1',
            save=save,
            filename="hal_comparison_legend_boltz1"
        )
        
        return af3_fig, boltz1_fig, af3_legend_fig, boltz1_legend_fig 