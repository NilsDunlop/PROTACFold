import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

class PlotConfig:
    """Central configuration for all plotting functionality."""
    
    # Date cutoffs
    AF3_CUTOFF = datetime.strptime('2021-09-30', '%Y-%m-%d')
    
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
    
    # Font sizes - INCREASED for better readability in compact Nature-style plots
    TITLE_SIZE = 16        # Kept the same - still appropriate
    AXIS_LABEL_SIZE = 13   # Increased from 12 to 13
    TICK_LABEL_SIZE = 12   # Kept the same - matches horizontal_bars.py
    LEGEND_TITLE_SIZE = 15 # Increased from 14 to 15  
    LEGEND_TEXT_SIZE = 12  # Reduced from 14 to 12 for better proportions
    ANNOTATION_SIZE = 10   # Kept the same
    SUBPLOT_LABEL_SIZE = 15 # Increased from 14 to 15
    
    # Line properties
    GRID_ALPHA = 0.3
    LINE_WIDTH = 1.5
    MARKER_SIZE = 40
    EDGE_WIDTH = 0.5
    
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