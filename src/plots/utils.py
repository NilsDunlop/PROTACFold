import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def categorize_by_cutoffs(df, value_column, cutoffs, category_column='Category'):
    """Categorize data based on value cutoffs."""
    if cutoffs is None or len(cutoffs) < 4:
        return df
        
    conditions = [
        df[value_column] < cutoffs[0],
        (df[value_column] >= cutoffs[0]) & (df[value_column] < cutoffs[1]),
        (df[value_column] >= cutoffs[1]) & (df[value_column] < cutoffs[2]),
        (df[value_column] >= cutoffs[2]) & (df[value_column] < cutoffs[3]),
        df[value_column] >= cutoffs[3]
    ]
    
    category_labels = [
        f"< {cutoffs[0]:.2f}",
        f"{cutoffs[0]:.2f} - {cutoffs[1]:.2f}",
        f"{cutoffs[1]:.2f} - {cutoffs[2]:.2f}",
        f"{cutoffs[2]:.2f} - {cutoffs[3]:.2f}",
        f"> {cutoffs[3]:.2f}"
    ]
    
    df_result = df.copy()
    df_result[category_column] = np.select(conditions, category_labels, default="Unknown")
    
    return df_result

def save_figure(fig, filename_base, save_path='../data/plots', dpi=300):
    """Save a figure with proper naming and directory creation."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}/{filename_base}_{timestamp}.png"
        
        # Save the figure
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {filename}")
        return True
    except Exception as e:
        print(f"Error saving figure: {e}")
        # Fallback to current directory
        fallback_filename = f"{filename_base}_{timestamp}.png"
        try:
            fig.savefig(fallback_filename, dpi=dpi)
            print(f"Figure saved to current directory: {fallback_filename}")
        except:
            print("Failed to save figure.")
        return False