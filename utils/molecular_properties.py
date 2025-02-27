#!/usr/bin/env python3

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def calculate_molecular_properties(df, smiles_column='Canonical_SMILES'):
    """
    Calculate molecular properties for compounds in a DataFrame using RDKit.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing SMILES strings
    smiles_column : str, optional
        Name of the column containing SMILES strings, default is 'Canonical_SMILES'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added molecular property columns
    """
    # Make a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Initialize lists to store calculated properties
    mol_weight = []
    heavy_atom_count = []
    ring_count = []
    rotatable_bond_count = []
    logp = []
    hba_count = []
    hbd_count = []
    tpsa = []
    aromatic_rings = []
    aliphatic_rings = []
    
    # Process each SMILES string
    for smiles in df[smiles_column]:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            # If SMILES conversion fails, add NaN values
            mol_weight.append(np.nan)
            heavy_atom_count.append(np.nan)
            ring_count.append(np.nan)
            rotatable_bond_count.append(np.nan)
            logp.append(np.nan)
            hba_count.append(np.nan)
            hbd_count.append(np.nan)
            tpsa.append(np.nan)
            aromatic_rings.append(np.nan)
            aliphatic_rings.append(np.nan)
            continue
        
        # Calculate molecular properties
        mol_weight.append(Descriptors.MolWt(mol))
        heavy_atom_count.append(mol.GetNumHeavyAtoms())
        ring_count.append(rdMolDescriptors.CalcNumRings(mol))
        rotatable_bond_count.append(Descriptors.NumRotatableBonds(mol))
        logp.append(Descriptors.MolLogP(mol))
        hba_count.append(Lipinski.NumHAcceptors(mol))
        hbd_count.append(Lipinski.NumHDonors(mol))
        tpsa.append(Descriptors.TPSA(mol))
        
        # Count aromatic and aliphatic rings
        aromatic_ring_count = 0
        aliphatic_ring_count = 0
        
        # Use the correct method to get rings
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            # Check if all atoms in the ring are aromatic
            is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
            if is_aromatic:
                aromatic_ring_count += 1
            else:
                aliphatic_ring_count += 1
        
        aromatic_rings.append(aromatic_ring_count)
        aliphatic_rings.append(aliphatic_ring_count)
    
    # Add calculated properties to the result DataFrame
    result_df['Molecular_Weight'] = mol_weight
    result_df['Heavy_Atom_Count'] = heavy_atom_count
    result_df['Ring_Count'] = ring_count
    result_df['Rotatable_Bond_Count'] = rotatable_bond_count
    result_df['LogP'] = logp
    result_df['HBA_Count'] = hba_count
    result_df['HBD_Count'] = hbd_count
    result_df['TPSA'] = tpsa
    result_df['Aromatic_Rings'] = aromatic_rings
    result_df['Aliphatic_Rings'] = aliphatic_rings
    
    return result_df


def main():
    """
    Example usage of the calculate_molecular_properties function.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate molecular properties for compounds in a CSV file')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--smiles_column', type=str, default='Canonical_SMILES', 
                        help='Name of the column containing SMILES strings (default: Canonical_SMILES)')
    
    args = parser.parse_args()
    
    # Read input CSV file
    try:
        df = pd.read_csv(args.input)
        print(f"Successfully read input file: {args.input}")
        print(f"Number of compounds: {len(df)}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Check if SMILES column exists
    if args.smiles_column not in df.columns:
        print(f"Error: Column '{args.smiles_column}' not found in the input file.")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Calculate molecular properties
    print("Calculating molecular properties...")
    result_df = calculate_molecular_properties(df, smiles_column=args.smiles_column)
    
    # Save results to output CSV file
    try:
        result_df.to_csv(args.output, index=False)
        print(f"Successfully saved results to: {args.output}")
    except Exception as e:
        print(f"Error saving output file: {e}")
    
    # Print summary of calculated properties
    print("\nSummary of calculated properties:")
    for prop in ['Molecular_Weight', 'Heavy_Atom_Count', 'Ring_Count', 'Rotatable_Bond_Count', 
                'LogP', 'HBA_Count', 'HBD_Count', 'TPSA', 'Aromatic_Rings', 'Aliphatic_Rings']:
        if prop in result_df.columns:
            print(f"{prop}: Mean = {result_df[prop].mean():.2f}, Min = {result_df[prop].min():.2f}, Max = {result_df[prop].max():.2f}")


if __name__ == "__main__":
    main()