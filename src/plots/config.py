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
    SMILES_SECONDARY = '#FC8D62' # orange
    SMILES_TERTIARY = '#57DB80'  # green
    
    # Other colors
    GRAY = '#A9A9A9'
    DARK_BLUE = '#2D3047'
    TEAL = '#1B998B'
    YELLOW = '#EAC435'
    
    # Font sizes
    TITLE_SIZE = 14
    AXIS_LABEL_SIZE = 12
    TICK_LABEL_SIZE = 10
    LEGEND_TITLE_SIZE = 12
    LEGEND_TEXT_SIZE = 10
    ANNOTATION_SIZE = 8
    SUBPLOT_LABEL_SIZE = 12
    
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