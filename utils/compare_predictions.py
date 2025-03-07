#!/usr/bin/env python3
"""
Protein Structure Comparison Tool

This script compares predicted protein structures with experimental structures
using Root Mean Square Deviation (RMSD) of alpha carbon atoms.

Features:
- Supports both PDB and mmCIF file formats
- Calculates RMSD between matching residues
- Can process individual PDB IDs or compare specific files
- Handles batch processing of multiple structures

Usage:
    python compare_predictions.py [-id PDB_ID] [-e EXP_FILE] [-p PRED_FILE]

Arguments:
    -id, --pdb_id    PDB ID to compare (e.g., 7jto)
    -e, --exp_file   Specific experimental file name (e.g., 7jto_pdb.cif)
    -p, --pred_file  Specific predicted file name (e.g., 7jto_af3.cif)

If no arguments are provided, the script will process all matching files
in the data directories.

Output:
    - RMSD values between predicted and experimental structures
"""

import os
import argparse
from glob import glob
from typing import Dict, Union, Optional
import numpy as np
from Bio import PDB
from Bio.PDB import Polypeptide


def get_parser(file_path: str) -> Union[PDB.PDBParser, PDB.MMCIFParser]:
    """
    Determine the appropriate parser based on file extension.
    
    Args:
        file_path: Path to the structure file
        
    Returns:
        PDB or mmCIF parser appropriate for the file
    """
    if file_path.lower().endswith(('.cif', '.mmcif')):
        return PDB.MMCIFParser(QUIET=True)
    return PDB.PDBParser(QUIET=True)


def compare_structures(pred_path: str, exp_path: str) -> Dict[str, float]:
    """
    Compare predicted and experimental protein structures using RMSD.
    
    Args:
        pred_path: Path to predicted structure file (PDB/mmCIF format)
        exp_path: Path to experimental structure file (PDB/mmCIF format)
    
    Returns:
        Dictionary containing comparison metrics:
        - rmsd: Root Mean Square Deviation of CA atoms
        
    Raises:
        ValueError: If structure files cannot be parsed or no valid atoms found
        RuntimeError: If RMSD calculation fails
    """
    # Parse structures with error handling
    try:
        pred_parser = get_parser(pred_path)
        exp_parser = get_parser(exp_path)
        
        pred_structure = pred_parser.get_structure("predicted", pred_path)
        exp_structure = exp_parser.get_structure("experimental", exp_path)
    except Exception as e:
        raise ValueError(f"Failed to parse structure files: {e}")
    
    def calc_rmsd() -> float:
        """Calculate RMSD between CA atoms of matching residues."""
        pred_coords = []
        exp_coords = []
        
        # Get CA atoms (alpha carbons) from both structures
        for pred_model, exp_model in zip(pred_structure, exp_structure):
            for pred_chain, exp_chain in zip(pred_model, exp_model):
                for pred_res, exp_res in zip(pred_chain, exp_chain):
                    # Skip non-standard residues and handle missing atoms
                    if ('CA' in pred_res and 'CA' in exp_res and 
                        pred_res.get_resname() in Polypeptide.standard_aa_names and
                        exp_res.get_resname() in Polypeptide.standard_aa_names):
                        pred_coords.append(pred_res['CA'].get_coord())
                        exp_coords.append(exp_res['CA'].get_coord())
        
        if not pred_coords or not exp_coords:
            raise ValueError("No valid CA atoms found for comparison")
            
        # Convert to numpy arrays
        pred_coords = np.array(pred_coords)
        exp_coords = np.array(exp_coords)
        
        # Calculate RMSD
        diff = pred_coords - exp_coords
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        return rmsd

    # Calculate RMSD with error handling
    try:
        results = {
            'rmsd': calc_rmsd()
        }
    except Exception as e:
        raise RuntimeError(f"Error calculating RMSD: {e}")

    return results


def process_files(exp_file: str, pred_file: str) -> Optional[Dict[str, float]]:
    """
    Process a pair of experimental and predicted structure files.
    
    Args:
        exp_file: Path to experimental structure file
        pred_file: Path to predicted structure file
        
    Returns:
        Dictionary of results if successful, None otherwise
    """
    if not os.path.exists(exp_file):
        print(f"Error: Experimental file not found: {exp_file}")
        return None
    if not os.path.exists(pred_file):
        print(f"Error: Predicted file not found: {pred_file}")
        return None

    try:
        results = compare_structures(pred_file, exp_file)
        return results
    except Exception as e:
        print(f"Error processing files: {e}")
        return None


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Compare predicted and experimental protein structures')
    parser.add_argument('-id', '--pdb_id', help='PDB ID to compare (e.g., 7jto)', default=None)
    parser.add_argument('-e', '--exp_file', help='Specific experimental file name (e.g., 7jto_pdb.cif)', default=None)
    parser.add_argument('-p', '--pred_file', help='Specific predicted file name (e.g., 7jto_af3.cif)', default=None)
    
    return parser.parse_args()


def main() -> None:
    """
    Main function to run the structure comparison workflow.
    
    This function:
    1. Parses command line arguments
    2. Determines which files to process
    3. Compares structures and reports results
    """
    args = parse_arguments()

    # Define paths relative to the utils directory
    exp_dir = os.path.join(os.path.dirname(__file__), "..", "data", "experimental")
    pred_dir = os.path.join(os.path.dirname(__file__), "..", "data", "predicted")

    # Ensure directories exist
    for directory in [exp_dir, pred_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

    if args.exp_file and args.pred_file:
        # Use specific filenames provided
        exp_file = os.path.join(exp_dir, args.exp_file)
        pred_file = os.path.join(pred_dir, args.pred_file)
        
        results = process_files(exp_file, pred_file)
        if results:
            print(f"Results for {args.exp_file} vs {args.pred_file}:")
            print(f"RMSD: {results['rmsd']:.2f} Å")
            
    elif args.pdb_id:
        # Look for specific PDB ID
        pdb_id = args.pdb_id.lower()  # Ensure lowercase for consistency
        exp_file = os.path.join(exp_dir, f"{pdb_id}_pdb.cif")
        pred_file = os.path.join(pred_dir, f"{pdb_id}_af3.cif")
        
        results = process_files(exp_file, pred_file)
        if results:
            print(f"Results for {pdb_id}:")
            print(f"RMSD: {results['rmsd']:.2f} Å")
            
    else:
        # Process all matching files in the directories
        exp_files = glob(os.path.join(exp_dir, "*_pdb.cif"))
        
        if not exp_files:
            print(f"No experimental files found in {exp_dir}")
            return
            
        successful = 0
        total = len(exp_files)
        all_rmsd = []
        
        for exp_file in exp_files:
            # Extract PDB ID from experimental filename
            pdb_id = os.path.basename(exp_file).split('_')[0]
            pred_file = os.path.join(pred_dir, f"{pdb_id}_af3.cif")
            
            if not os.path.exists(pred_file):
                print(f"Skipping {pdb_id}: No matching prediction file found")
                continue

            print(f"\nProcessing {pdb_id}...")
            results = process_files(exp_file, pred_file)
            
            if results:
                print(f"RMSD: {results['rmsd']:.2f} Å")
                successful += 1
                all_rmsd.append(results['rmsd'])
        
        # Print summary
        if successful > 0:
            avg_rmsd = np.mean(all_rmsd)
            print(f"\nSummary: Successfully processed {successful} out of {total} structures")
            print(f"Average RMSD: {avg_rmsd:.2f} Å")
        else:
            print("\nNo structures were successfully processed")


if __name__ == "__main__":
    main()