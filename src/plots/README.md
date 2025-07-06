# Plotting Scripts Documentation

This directory contains a collection of Python scripts for generating various plots to analyze and visualize protein structure prediction results. The main entry point is `main.py`, which provides a command-line interface to generate all available plots.

---

## Scripts Overview

### `main.py`
*   **Purpose**: The main executable script that provides a user-facing command-line interface (CLI).
*   **Class**: `PlottingApp`
*   **Functionality**:
    *   Initializes all the plotter objects and loads the necessary data upon starting.
    *   Presents a menu of available plot types to the user.
    *   Based on user input, it calls the appropriate methods from the various plotter classes to generate the requested plots.
    *   Handles saving the plots to the `data/plots` directory with standardized filenames.

### `config.py`
*   **Purpose**: To centralize all visual and functional configurations for the plots, ensuring a consistent and professional look and feel.
*   **Class**: `PlotConfig`
*   **Functionality**: This class contains static variables that define the entire visual theme and behavior of the plots. This includes:
    *   **Colors**: Defines specific hex codes for different models (`AlphaFold3`, `Boltz1`), data types (CCD, SMILES), and plot elements.
    *   **Font Sizes**: Sets standard sizes for titles, labels, and ticks.
    *   **Dimensions**: Specifies default widths and heights for different types of plots.
    *   **Bar Properties**: Controls the appearance of bars in bar plots (e.g., width, spacing, edge color).
    *   **Thresholds**: Sets default values for performance thresholds (e.g., RMSD < 4.0 Å).
    *   `apply_style`: A class method to apply these settings to `matplotlib`'s global `rcParams`, ensuring all plots adhere to the same style.

### `data_loader.py`
*   **Purpose**: To handle all data loading and preprocessing tasks from CSV files.
*   **Class**: `DataLoader`
*   **Key Methods**:
    *   `load_data`: Loads a CSV file into a pandas DataFrame.
    *   `aggregate_by_pdb_id`: Groups the data by PDB ID and calculates the mean and standard deviation for all numeric metrics across different prediction seeds.
    *   `calculate_classification_cutoffs_from_af3_aggregated_data`: Determines RMSD percentile cutoffs from the AlphaFold3 data, which are then used to consistently categorize structures in other plots.
    *   `identify_binary_structures`: Adds a flag to the DataFrame to identify structures that are binary (protein-protein) complexes based on missing DockQ metrics.
    *   `filter_comparison_data`: A utility to filter the dataset based on common criteria like molecule type, model, or seed.

### `base_plotter.py`
*   **Purpose**: To provide a foundational class with common plotting utilities that are inherited by all other specialized plotter classes.
*   **Class**: `BasePlotter`
*   **Key Methods**:
    *   `create_figure`: A helper to create a new `matplotlib` figure and axis.
    *   `save_plot`: A wrapper to save the generated plot to a file with a consistent naming scheme.
    *   `add_legend`: Adds a legend to a plot with standardized styling.
    *   `calculate_plot_dimensions`: Calculates an appropriate figure height based on the number of data points to ensure readability.

---

## Specialized Plotters

### `comparison_plotter.py`
*   **Purpose**: To create plots that directly compare the performance of `AlphaFold3` and `Boltz1`.
*   **Class**: `ComparisonPlotter`
*   **Key Methods**:
    *   `plot_af3_vs_boltz1`: Generates detailed, paginated horizontal bar plots comparing the two models on a per-structure basis for metrics like RMSD, DockQ, and LRMSD.
    *   `plot_mean_comparison`: Creates a summary bar plot showing the mean performance of each model across all structures, separated by ligand type (CCD vs. SMILES).
    *   `plot_seed_comparison`: A wrapper to compare model performance for a single, specific prediction seed.

### `training_cutoff_plotter.py`
*   **Purpose**: To evaluate how well the models generalize to structures released after their training data cutoff date (September 30, 2021).
*   **Class**: `TrainingCutoffPlotter`
*   **Key Methods**:
    *   `plot_training_cutoff_comparison`: Creates a bar plot that groups data into four categories: Pre-cutoff CCD, Pre-cutoff SMILES, Post-cutoff CCD, and Post-cutoff SMILES. The "Post-cutoff" bars are visually distinguished with hatching.

### `hal_comparison.py`
*   **Purpose**: To compare standard model predictions (which include a ligand) against "No Ligand" (HAL) predictions. This helps quantify the ligand's impact on structure prediction.
*   **Class**: `HALComparisonPlotter`
*   **Key Methods**:
    *   `load_and_merge_data`: Loads and merges the HAL results with the main combined results.
    *   `plot_hal_comparison`: Generates bar plots comparing the DockQ scores of predictions made with a ligand versus those made without.

### `poi_e3l_plotter.py`
*   **Purpose**: To create specialized plots that analyze the model's performance on the two main protein components of the complex: the **P**rotein **o**f **I**nterest (POI) and the E3 Ligase.
*   **Class**: `POI_E3LPlotter`
*   **Functionality**: It groups proteins by family (e.g., Kinases, Nuclear Regulators) and creates grid-based plots to compare POI and E3L metrics for different models.

### `rmsd_complex_isolated.py`
*   **Purpose**: To provide a detailed breakdown of the RMSD error into its constituent parts.
*   **Class**: `RMSDComplexIsolatedPlotter`
*   **Key Methods**:
    *   `plot_aggregated_rmsd_components`: Creates a summary bar plot showing the mean RMSD for the entire complex, the isolated POI, and the isolated E3 ligase.
    *   `plot_per_pdb_rmsd_components`: Generates detailed horizontal bar plots for each structure, showing both the overall complex RMSD and a stacked bar of the POI and E3 RMSDs.

### `horizontal_bars.py`
*   **Purpose**: To create detailed, categorized horizontal bar plots for a single model's performance across multiple metrics (RMSD, DockQ, LRMSD).
*   **Class**: `HorizontalBarPlotter`
*   **Functionality**: It separates structures into "binary" and "ternary" complexes and then further categorizes them by RMSD performance. It automatically handles pagination for categories with many structures.

### `ptm_plotter.py`
*   **Purpose**: To visualize the pTM and ipTM confidence scores from the models.
*   **Class**: `PTMPlotter`
*   **Key Methods**:
    *   `plot_ptm_bars`: Creates a two-panel plot with horizontal bars. The left panel shows pTM scores and the right panel shows ipTM scores, sorted by confidence.

### `property_plotter.py`
*   **Purpose**: To investigate how model performance (LRMSD) correlates with various molecular properties of the ligand.
*   **Class**: `PropertyPlotter`
*   **Key Methods**:
    *   `plot_combined_properties`: Generates a grid of plots showing LRMSD as a function of different binned properties (e.g., Molecular Weight, Rotatable Bond Count, LogP).

---

## Utility Scripts

### `utils.py`
*   **Purpose**: A module containing miscellaneous helper functions used across multiple plotting scripts to avoid code duplication.
*   **Key Functions**:
    *   `categorize_by_cutoffs`: Assigns a category label to data based on predefined cutoff values.
    *   `distribute_structures_evenly`: A helper for pagination that splits a large dataset into evenly sized pages.
    *   `save_figure`: Saves a `matplotlib` figure to a file with a standardized name and timestamp.
    *   `create_plot_filename`: Generates a standardized, concise filename for a plot based on its parameters.

### `dev.py`
*   **Purpose**: A utility script for developers to enable live reloading.
*   **Functionality**: It uses the `watchdog` library to monitor the project directory for any changes to `.py` files. When a change is detected, it automatically restarts the `main.py` script, which is useful for rapid development and testing.