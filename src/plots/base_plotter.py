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
    
    def calculate_plot_dimensions(self, num_structures, base_width=12, 
                                height_per_structure=0.5, min_height=8):
        """
        Calculate appropriate figure dimensions based on the number of structures.
        
        Args:
            num_structures: Number of structures to display
            base_width: Base width for the figure
            height_per_structure: Height to allocate per structure
            min_height: Minimum height for the figure
            
        Returns:
            width, height: Appropriate dimensions for the figure
        """
        height = max(min_height, num_structures * height_per_structure + 2)
        return base_width, height