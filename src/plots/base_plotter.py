import matplotlib.pyplot as plt
from config import PlotConfig
from utils import save_figure

class BasePlotter:
    """Base class for all plotting functionality."""
    
    def __init__(self, figure_size=(10, 8)):
        """Initialize the base plotter."""
        self.figure_size = figure_size
        # Apply global style settings
        PlotConfig.apply_style()
    
    def create_figure(self, width=None, height=None):
        """Create a new figure."""
        width = width or self.figure_size[0]
        height = height or self.figure_size[1]
        fig, ax = plt.subplots(figsize=(width, height))
        return fig, ax
    
    def save_plot(self, fig, name, save_path=None, dpi=300):
        """Save the figure to a file."""
        save_path = save_path or PlotConfig.SAVE_PATH
        return save_figure(fig, name, save_path, dpi)
    
    def add_legend(self, ax, handles=None, labels=None, **kwargs):
        """Add a legend to the axis."""
        legend_defaults = {
            'loc': 'best',
            'framealpha': 0,  # Transparent background
            'edgecolor': 'none'  # No border
        }
        
        legend_defaults.update(kwargs)
        
        if handles and labels:
            ax.legend(handles, labels, **legend_defaults)
        else:
            ax.legend(**legend_defaults)