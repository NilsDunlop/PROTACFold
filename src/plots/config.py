import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

class PlotConfig:
    """Central configuration for all plotting functionality."""
    
    # Date cutoffs
    AF3_CUTOFF = datetime.strptime('2021-09-30', '%Y-%m-%d')
    TRAINING_CUTOFF_DATE = pd.to_datetime('2021-09-30')
    
    # Output path
    SAVE_PATH = '../data/plots'
    
    # Colors
    # CCD colors
    CCD_PRIMARY = '#FA6347'    # red
    CCD_SECONDARY = '#A157DB'  # purple
    CCD_TERTIARY = '#23C9FF'   # light blue
    
    # SMILES colors
    SMILES_PRIMARY = '#1E8FFF'  # blue
    SMILES_SECONDARY = '#FED766' # yellow
    SMILES_TERTIARY = '#57DB80'  # green
    
    # Other colors
    GRAY = '#A9A9A9'
    DARK_BLUE = '#2D3047'
    TEAL = '#1B998B'
    YELLOW = '#EAC435'
    
    # Model comparison colors
    AF3_CCD_COLOR = '#FA6347'      # red (same as CCD_PRIMARY)
    AF3_CCD_COLOR_POI = '#23C9FF'
    AF3_CCD_COLOR_E3 = '#084C61' #074F57 #0A2463 #254441

    AF3_SMILES_COLOR = '#1E8FFF'    # blue (same as SMILES_PRIMARY)
    AF3_SMILES_COLOR_POI = '#FED766'
    AF3_SMILES_COLOR_E3 = '#307351'
    
    BOLTZ1_CCD_COLOR = '#A157DB'    # purple
    BOLTZ1_CCD_COLOR_POI = '#D9DBF1'
    BOLTZ1_CCD_COLOR_E3 = '#66462C'
    
    BOLTZ1_SMILES_COLOR = '#57DB80' # green
    BOLTZ1_SMILES_COLOR_POI = '#336699'
    BOLTZ1_SMILES_COLOR_E3 = '#86BBD8'
    
    # HAL color
    HAL_COLOR = '#808080'
    
    # Font sizes - Based on comparison_plotter.py which has the proper sizes
    TITLE_SIZE = 15        # From comparison_plotter.py
    AXIS_LABEL_SIZE = 14   # From comparison_plotter.py
    TICK_LABEL_SIZE = 13   # From comparison_plotter.py
    LEGEND_TITLE_SIZE = 15 # Increased from 14 to 15  
    LEGEND_TEXT_SIZE = 11  # From comparison_plotter.py
    ANNOTATION_SIZE = 10   # Kept the same
    SUBPLOT_LABEL_SIZE = 15 # Increased from 14 to 15
    VALUE_LABEL_SIZE = 12  # For value labels on bars
    
    # Plot dimensions
    # Standard plot dimensions
    STANDARD_WIDTH = 3
    STANDARD_HEIGHT = 3
    
    # Horizontal bar plot dimensions
    HORIZONTAL_BAR_WIDTH = 8
    HORIZONTAL_BAR_HEIGHT_FACTOR = 0.85  # For calculating height based on structures
    HORIZONTAL_BAR_MIN_HEIGHT = 3
    HORIZONTAL_BAR_HEIGHT_PADDING = 1.2
    
    # Binary plot dimensions
    BINARY_PLOT_WIDTH = 3
    
    # PTM plot dimensions
    PTM_WIDTH = 8.0
    PTM_INITIAL_MAX_HEIGHT = 14.0
    PTM_PAGE_MAX_HEIGHT = 8.0
    PTM_PAGE_HEIGHT_PER_STRUCTURE = 0.6
    
    # HAL comparison dimensions
    HAL_PLOT_WIDTH = 10
    HAL_PLOT_HEIGHT = 3
    
    # POI/E3L grid dimensions
    POI_E3L_GRID_WIDTH = 8.0
    POI_E3L_GRID_HEIGHT_FACTOR = 0.4
    POI_E3L_GRID_HEIGHT_PADDING = 3.0
    POI_E3L_GRID_DEFAULT_HEIGHT = 12.0
    POI_E3L_SINGLE_MODEL_WIDTH = 4.0  # Half of grid width
    POI_E3L_VERTICAL_WIDTH = 6.0
    POI_E3L_VERTICAL_HEIGHT = 4.0
    
    # Property plot dimensions
    PROPERTY_COMBINED_WIDTH = 15
    PROPERTY_COMBINED_HEIGHT = 7
    
    # RMSD complex/isolated dimensions
    RMSD_AGGREGATED_WIDTH = 3
    RMSD_AGGREGATED_HEIGHT = 3
    RMSD_PER_PDB_WIDTH = 5
    
    # Bar properties
    # Horizontal bar properties
    BAR_HEIGHT = 0.2
    BAR_SPACING = 0.03
    
    # PTM bar properties
    PTM_BAR_HEIGHT = 0.30
    PTM_BAR_SPACING = 0.02
    
    # Vertical bar properties (for comparison plots)
    BAR_WIDTH = 0.08
    BAR_EDGE_LINE_WIDTH = 0.5
    BAR_SPACING_FACTOR = 1.8
    
    # Training cutoff bar properties
    TRAINING_BAR_WIDTH = 0.08
    TRAINING_BAR_SPACING_FACTOR = 1.8
    
    # HAL comparison bar properties
    HAL_BAR_WIDTH = 0.20
    HAL_BAR_SPACING_FACTOR = 1.0
    
    # POI/E3L bar properties
    POI_E3L_BAR_WIDTH = 0.6
    POI_E3L_BAR_ALPHA = 1
    
    # Property plot bar properties
    PROPERTY_BAR_ALPHA = 1
    PROPERTY_NORMALIZED_WIDTH_FACTOR = 0.05
    
    # RMSD complex/isolated bar properties
    RMSD_AGGREGATED_BAR_WIDTH = 0.6
    RMSD_BAR_HEIGHT_HORIZONTAL = 0.30
    RMSD_BAR_SPACING_HORIZONTAL = 0
    
    # Bar edge properties
    BAR_EDGE_COLOR = 'black'
    BAR_EDGE_WIDTH = 0.5
    
    # Error bar properties
    ERROR_BAR_CAPSIZE = 3
    ERROR_BAR_THICKNESS = 0.8
    ERROR_BAR_ALPHA = 0.7
    ERROR_BAR_COLOR = 'black'
    
    # Error region properties (for fill_between)
    ERROR_REGION_ALPHA = 0.2
    
    # Grid properties
    GRID_ALPHA = 0.2
    GRID_LINESTYLE = '--'
    
    # Threshold line properties
    THRESHOLD_LINE_ALPHA = 1
    THRESHOLD_LINE_WIDTH = 1.0
    THRESHOLD_LINE_COLOR = 'gray'
    THRESHOLD_LINE_STYLE = '--'
    
    # Default threshold values
    DEFAULT_RMSD_THRESHOLD = 4.0
    DEFAULT_DOCKQ_THRESHOLD = 0.23
    DEFAULT_LRMSD_THRESHOLD = 4.0
    DEFAULT_PTM_THRESHOLD = 0.8
    DEFAULT_IPTM_THRESHOLD = 0.6
    
    # Hatch properties (for training cutoff plots)
    PRE_TRAINING_HATCH = ''
    POST_TRAINING_HATCH = '//'
    POST_TRAINING_HATCH_LINE_WIDTH = 0.5
    LEGEND_POST_TRAINING_HATCH = '///'
    
    # Layout and spacing
    # Horizontal bar layout
    HORIZONTAL_BAR_Y_SPACING = 0.85
    
    # POI/E3L layout
    POI_E3L_X_AXIS_PADDING_FACTOR = 0.05
    POI_E3L_HSPACE = 0.05
    POI_E3L_YLABEL_PAD = 20
    POI_E3L_YLABEL_X_COORD = -0.45
    POI_E3L_YLABEL_Y_COORD = 0.5
    
    # Property plot layout
    PROPERTY_SUBPLOT_SPACING = 0.25
    PROPERTY_SUBPLOT_WSPACE = 0.15
    PROPERTY_SUBPLOT_HSPACE = 0.3
    PROPERTY_SUBPLOT_ROW_OFFSET = 0.05
    
    # RMSD complex/isolated layout
    RMSD_PDB_INTER_GROUP_SPACING = 0.2
    
    # Maximum structures per page/plot
    MAX_STRUCTURES_PER_PAGE = 20
    MAX_STRUCTURES_PER_HORIZONTAL_PLOT = 20
    PTM_MAX_STRUCTURES_PER_PAGE = 17
    
    # Y-axis limits for different plot types
    Y_AXIS_LIMITS = {
        "MOLECULAR GLUE": (0, 55),
        "PROTAC": (0, 50),
        "DEFAULT": (0, 50)
    }
    
    # Default bin sizes for property plots
    DEFAULT_BIN_SIZES = {
        "Molecular_Weight": 50,
        "Heavy_Atom_Count": 10,
        "Rotatable_Bond_Count": 5,
        "LogP": 1,
        "HBD_Count": 1,
        "HBA_Count": 2
    }
    
    # Line properties
    LINE_WIDTH = 1.5
    MARKER_SIZE = 40
    EDGE_WIDTH = 0.5
    
    # Margins and positioning
    MARGIN_Y = 0.6
    MARGIN_X = 0.6
    
    @classmethod
    def apply_style(cls):
        """Apply the configured style to matplotlib."""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'Arial', 'DejaVu Sans']
        plt.rcParams['font.size'] = cls.TICK_LABEL_SIZE
        plt.rcParams['text.color'] = 'black'
        
        mpl.rcParams['axes.titlesize'] = cls.TITLE_SIZE
        mpl.rcParams['axes.labelsize'] = cls.AXIS_LABEL_SIZE
        mpl.rcParams['xtick.labelsize'] = cls.TICK_LABEL_SIZE
        mpl.rcParams['ytick.labelsize'] = cls.TICK_LABEL_SIZE
        mpl.rcParams['legend.fontsize'] = cls.LEGEND_TEXT_SIZE
        mpl.rcParams['legend.title_fontsize'] = cls.LEGEND_TITLE_SIZE