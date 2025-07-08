import pandas as pd
import numpy as np

class DataLoader:
    """Handles loading and preprocessing of data for plotting."""
    
    @staticmethod
    def load_data(file_path):
        """Load data from a CSV file and sort by release date and PDB ID."""
        df = pd.read_csv(file_path, header=0)
        return df.sort_values(['RELEASE_DATE', 'PDB_ID'], ascending=[True, True])
    
    @staticmethod
    def aggregate_by_pdb_id(df):
        """Aggregate data by PDB_ID with mean and standard deviation."""
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Create aggregation dictionary
        agg_dict = {}
        for col in numeric_cols:
            agg_dict[col] = ['mean', 'std']
        
        # Add categorical columns
        categorical_cols = ['TYPE', 'POI_NAME', 'E3_NAME', 'LIGAND_CCD', 'LIGAND_LINK', 'LIGAND_SMILES', 'RELEASE_DATE']
        for col in categorical_cols:
            if col in df.columns:
                agg_dict[col] = 'first'
        
        # Group by PDB_ID and calculate aggregated metrics
        df_agg = df.groupby('PDB_ID').agg(agg_dict)
        
        # Flatten multi-index columns
        df_agg.columns = [col[0] if col[1] == 'first' else '_'.join(col).strip() for col in df_agg.columns.values]
        
        # Reset index for PDB_ID column
        df_agg = df_agg.reset_index()
        
        # Handle standard deviation for single samples
        pdb_counts = df.groupby('PDB_ID').size().to_dict()
        for col in numeric_cols:
            std_col = f"{col}_std"
            if std_col in df_agg.columns:
                mask = df_agg['PDB_ID'].map(pdb_counts) == 1
                df_agg.loc[mask, std_col] = np.nan
        
        # Convert date strings to datetime
        if 'RELEASE_DATE' in df_agg.columns:
            df_agg['RELEASE_DATE'] = pd.to_datetime(df_agg['RELEASE_DATE'])
            
        # Clean numeric values
        for col in df_agg.columns:
            if col.endswith('_mean') or col.endswith('_std'):
                df_agg[col] = pd.to_numeric(df_agg[col], errors='coerce')
                
        return df_agg
    
    @staticmethod
    def calculate_classification_cutoffs_from_af3_aggregated_data(df_agg, molecule_type=None, percentiles=[20, 40, 60, 80]):
        """
        Calculate classification cutoffs from AlphaFold3 CCD RMSD aggregated data.
        
        This is the centralized method for calculating cutoffs that should be used across all plotting modules
        to ensure consistency. Always uses AlphaFold3 CCD_RMSD_mean values from aggregated data.
        
        Args:
            df_agg: Aggregated DataFrame containing AlphaFold3 data
            molecule_type: Optional molecule type filter (e.g., "PROTAC", "Molecular Glue")
            percentiles: List of percentiles to use for cutoffs [20, 40, 60, 80]
            
        Returns:
            List of cutoff values or None if insufficient data
        """
        if df_agg is None or df_agg.empty:
            return None
            
        # Look for AlphaFold3 data - check various possible identifiers
        af3_data = pd.DataFrame()
        if 'MODEL_TYPE' in df_agg.columns:
            unique_model_types = df_agg['MODEL_TYPE'].unique()
            af3_identifiers = ["AlphaFold3", "alphafold3", "ALPHAFOLD3", "AF3"]
            
            # Find which AlphaFold3 identifier is present in the data
            af3_model_type_found = None
            for af3_id in af3_identifiers:
                if af3_id in unique_model_types:
                    af3_model_type_found = af3_id
                    break
            
            if af3_model_type_found is not None:
                af3_data = df_agg[df_agg['MODEL_TYPE'] == af3_model_type_found].copy()
        else:
            # If no MODEL_TYPE column, assume all data is AlphaFold3
            af3_data = df_agg.copy()
        
        if af3_data.empty:
            return None
            
        # Filter by molecule type if specified
        if molecule_type:
            molecule_type_col = 'MOLECULE_TYPE' if 'MOLECULE_TYPE' in af3_data.columns else 'TYPE'
            if molecule_type_col in af3_data.columns:
                af3_data = af3_data[af3_data[molecule_type_col] == molecule_type].copy()
        
        # Extract CCD RMSD mean values
        if 'CCD_RMSD_mean' not in af3_data.columns:
            return None
            
        ccd_rmsd_mean_values = af3_data['CCD_RMSD_mean'].dropna()
        
        # Need at least 5 data points for 4 distinct percentile cuts
        if len(ccd_rmsd_mean_values) < 5:
            return None
            
        # Calculate percentile cutoffs
        cutoffs = [np.percentile(ccd_rmsd_mean_values, p) for p in percentiles]
        return cutoffs
    

        
    @staticmethod
    def identify_binary_structures(df):
        """
        Identify binary structures based on missing DockQ metrics.
        Binary structures are identified by having NaN values for both 
        SMILES_DOCKQ_SCORE and CCD_DOCKQ_SCORE, as well as 
        SMILES_DOCKQ_LRMSD and CCD_DOCKQ_LRMSD.
        
        Args:
            df: DataFrame with the structure data
            
        Returns:
            DataFrame with an additional column 'is_binary' (True/False)
        """
        df_with_binary = df.copy()
        
        # Check if all four DockQ metrics are NaN
        binary_mask = (
            df['SMILES_DOCKQ_SCORE_mean'].isna() & 
            df['CCD_DOCKQ_SCORE_mean'].isna() & 
            df['SMILES_DOCKQ_LRMSD_mean'].isna() & 
            df['CCD_DOCKQ_LRMSD_mean'].isna()
        )
        
        df_with_binary['is_binary'] = binary_mask
        return df_with_binary
        
    @staticmethod
    def calculate_comparison_metrics(df, model_types, metric_columns):
        """
        Calculate mean, standard error, and count statistics for each model and ligand type.
        
        Args:
            df: DataFrame containing filtered data
            model_types: List of model types (e.g., ["AlphaFold3", "Boltz1"])
            metric_columns: Tuple of (smiles_col, ccd_col, label)
            
        Returns:
            Dictionary containing means, errors, and counts for each model/metric combination
        """
        smiles_col, ccd_col, _ = metric_columns
        
        # Handle None value for model_types
        if model_types is None:
            model_types = ["AlphaFold3", "Boltz1"]
        
        # Initialize dictionaries to store values
        means = {}
        errors = {}
        counts = {}
        
        # For each model type, calculate metrics
        for model in model_types:
            model_df = df[df['MODEL_TYPE'] == model]
            
            # Calculate CCD metrics
            ccd_values = model_df[ccd_col].dropna()
            if len(ccd_values) > 0:
                means[f"{model}_CCD"] = ccd_values.mean()
                errors[f"{model}_CCD"] = ccd_values.std() / np.sqrt(len(ccd_values))  # Standard error
                counts[f"{model}_CCD"] = len(ccd_values)
            else:
                means[f"{model}_CCD"] = 0
                errors[f"{model}_CCD"] = 0
                counts[f"{model}_CCD"] = 0
                
            # Calculate SMILES metrics
            smiles_values = model_df[smiles_col].dropna()
            if len(smiles_values) > 0:
                means[f"{model}_SMILES"] = smiles_values.mean()
                errors[f"{model}_SMILES"] = smiles_values.std() / np.sqrt(len(smiles_values))
                counts[f"{model}_SMILES"] = len(smiles_values)
            else:
                means[f"{model}_SMILES"] = 0
                errors[f"{model}_SMILES"] = 0
                counts[f"{model}_SMILES"] = 0
        
        # Calculate improvement percentages
        metrics = {
            'means': means,
            'errors': errors,
            'counts': counts,
            'improvements': {}
        }
        
        # Calculate improvement percentages if data exists
        if "AlphaFold3_CCD" in means and "Boltz1_CCD" in means and means["Boltz1_CCD"] > 0:
            metrics['improvements']['CCD'] = (means["Boltz1_CCD"] - means["AlphaFold3_CCD"]) / means["Boltz1_CCD"] * 100
            
        if "AlphaFold3_SMILES" in means and "Boltz1_SMILES" in means and means["Boltz1_SMILES"] > 0:
            metrics['improvements']['SMILES'] = (means["Boltz1_SMILES"] - means["AlphaFold3_SMILES"]) / means["Boltz1_SMILES"] * 100
        
        return metrics

    @staticmethod
    def filter_comparison_data(df, molecule_type, model_types=None, seeds=None, metric_type=None, get_metric_columns_func=None):
        """
        Filter data based on common criteria for model comparison plots.
        
        Args:
            df: DataFrame containing the data to filter
            molecule_type: Type of molecule to filter by ('PROTAC' or 'Molecular Glue')
            model_types: List of model types to include (None for all)
            seeds: List of seeds to include (None for all)
            metric_type: Type of metric to filter by ('RMSD' or 'DOCKQ')
            get_metric_columns_func: Function to get metric columns
            
        Returns:
            Filtered DataFrame or None if empty
        """
        # Check if MOLECULE_TYPE column exists, otherwise use TYPE column
        molecule_type_col = 'MOLECULE_TYPE' if 'MOLECULE_TYPE' in df.columns else 'TYPE'
        
        if model_types is None:
            model_types = ["AlphaFold3", "Boltz1"]
        
        if seeds is None:
            if 'SEED' in df.columns:
                seeds = sorted(df['SEED'].unique())
            else:
                seeds = [1]  # Default if no SEED column
                
        # Filter the dataframe for the specified molecule type
        if molecule_type_col in df.columns:
            df_filtered = df[df[molecule_type_col] == molecule_type].copy()
        else:
            df_filtered = df.copy()
            print("Warning: TYPE column not found, no molecule type filtering applied")
        
        # Filter for specified model types
        df_filtered = df_filtered[df_filtered['MODEL_TYPE'].isin(model_types)]
        
        # Filter for specified seeds
        if 'SEED' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['SEED'].isin(seeds)]
            
        # Filter out N/A values if metric_type is provided
        if metric_type is not None and get_metric_columns_func is not None:
            metric_columns = get_metric_columns_func(metric_type)
            if metric_columns:
                smiles_col, ccd_col, _ = metric_columns
                # Filter out rows where both metric columns are N/A
                has_data_mask = ~(df_filtered[smiles_col].isna() & df_filtered[ccd_col].isna())
                df_filtered = df_filtered[has_data_mask]
                
        # Check if we have any data after filtering
        if df_filtered.empty:
            print(f"ERROR: No data available after filtering")
            return None
                
        return df_filtered