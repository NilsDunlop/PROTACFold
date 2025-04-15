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
    def calculate_percentiles(df, column, percentiles=[20, 40, 60, 80], filter_column=None, filter_value=None):
        """Calculate percentile cutoffs for a column, optionally filtering by another column."""
        if filter_column and filter_value:
            filtered_df = df[df[filter_column] == filter_value]
        else:
            filtered_df = df
            
        sorted_values = filtered_df[column].dropna().sort_values()
        if len(sorted_values) > 0:
            cutoffs = [np.percentile(sorted_values, p) for p in percentiles]
            return cutoffs
        return None
        
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