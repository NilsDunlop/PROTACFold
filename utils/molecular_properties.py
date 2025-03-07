#!/usr/bin/env python3
"""
Molecular Properties Calculator

This script calculates various molecular properties for compounds using RDKit.

Features:
- Processes a CSV file containing SMILES strings
- Calculates key molecular properties such as molecular weight, LogP, and counts of 
  various structural features
- Outputs the results to a new CSV file with the calculated properties

Usage:
    python molecular_properties.py --input <input_csv> --output <output_csv> [--smiles_column <column_name>]

Parameters:
    --input         Path to input CSV file containing SMILES strings
    --output        Path to output CSV file to save results
    --smiles_column Name of the column containing SMILES strings (default: 'Canonical_SMILES')

Output properties:
    - Molecular_Weight: Molecular weight of the compound
    - Heavy_Atom_Count: Number of non-hydrogen atoms
    - Ring_Count: Total number of rings
    - Rotatable_Bond_Count: Number of rotatable bonds
    - LogP: Calculated octanol-water partition coefficient
    - HBA_Count: Number of hydrogen bond acceptors
    - HBD_Count: Number of hydrogen bond donors
    - TPSA: Topological polar surface area
    - Aromatic_Rings: Number of aromatic rings
    - Aliphatic_Rings: Number of aliphatic rings
"""

import argparse
import pandas as pd
import numpy as np
from typing import List
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def calculate_molecular_properties(df: pd.DataFrame, smiles_column: str = 'Canonical_SMILES') -> pd.DataFrame:
    """
    Calculate molecular properties for compounds in a DataFrame using RDKit.
    
    Args:
        df: DataFrame containing SMILES strings
        smiles_column: Name of the column containing SMILES strings
    
    Returns:
        DataFrame with added molecular property columns
    
    Note:
        Properties calculated include molecular weight, heavy atom count,
        ring counts, rotatable bonds, LogP, hydrogen bond donors/acceptors,
        topological polar surface area, and aromatic/aliphatic ring counts.
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


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Calculate molecular properties for compounds in a CSV file')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--smiles_column', type=str, default='Canonical_SMILES', 
                        help='Name of the column containing SMILES strings (default: Canonical_SMILES)')
    
    return parser.parse_args()


def print_summary(df: pd.DataFrame, property_columns: List[str]) -> None:
    """
    Print a summary of the calculated molecular properties.
    
    Args:
        df: DataFrame containing the calculated properties
        property_columns: List of property column names to include in the summary
    """
    print("\nSummary of calculated properties:")
    for prop in property_columns:
        if prop in df.columns:
            print(f"{prop}: Mean = {df[prop].mean():.2f}, Min = {df[prop].min():.2f}, Max = {df[prop].max():.2f}")


def main() -> None:
    """
    Main function to run the molecular properties calculation workflow.
    
    This function:
    1. Parses command line arguments
    2. Reads the input CSV file
    3. Calculates molecular properties
    4. Saves the results to the output CSV file
    5. Prints a summary of the calculated properties
    """
    # Parse command line arguments
    args = parse_arguments()
    
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
        return
    
    # Print summary of calculated properties
    property_columns = [
        'Molecular_Weight', 'Heavy_Atom_Count', 'Ring_Count', 'Rotatable_Bond_Count',
        'LogP', 'HBA_Count', 'HBD_Count', 'TPSA', 'Aromatic_Rings', 'Aliphatic_Rings'
    ]
    print_summary(result_df, property_columns)


if __name__ == "__main__":
    main()