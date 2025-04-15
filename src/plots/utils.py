import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import math

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

def distribute_structures_evenly(df, max_per_page):
    """
    Distribute structures evenly across multiple pages for consistent visualization.
    
    Args:
        df: DataFrame containing all structures
        max_per_page: Maximum number of structures to show in a single page
        
    Returns:
        pages: List of DataFrames, one for each page
        structures_per_page: Number of structures per page
    """
    total_structures = len(df)
    
    # If we can fit everything on one page, just return the original DataFrame
    if total_structures <= max_per_page:
        return [df], total_structures
        
    # Calculate number of pages needed
    num_pages = math.ceil(total_structures / max_per_page)
    
    # Try to distribute structures evenly across all pages
    structures_per_page = math.ceil(total_structures / num_pages)
    
    # Make sure we don't exceed the maximum
    structures_per_page = min(structures_per_page, max_per_page)
    
    # Recalculate number of pages with the even distribution
    num_pages = math.ceil(total_structures / structures_per_page)
    
    # Split the dataframe into pages with equal number of structures
    pages = []
    for i in range(num_pages):
        start_idx = i * structures_per_page
        end_idx = min(start_idx + structures_per_page, total_structures)
        
        # Get structures for this page
        page_df = df.iloc[start_idx:end_idx].copy()
        pages.append(page_df)
        
    return pages, structures_per_page

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