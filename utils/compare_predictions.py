from Bio import PDB
from Bio.PDB import *
import numpy as np
from typing import Dict, Union, Tuple
import argparse
import os
from glob import glob

def compare_structures(pred_path: str, exp_path: str) -> Dict[str, float]:
    """
    Compare predicted and experimental protein structures using RMSD.
    
    Args:
        pred_path: Path to predicted structure file (PDB/mmCIF format)
        exp_path: Path to experimental structure file (PDB/mmCIF format)
    
    Returns:
        Dictionary containing comparison metrics:
        - rmsd: Root Mean Square Deviation of CA atoms
    """
    # Parse structures with error handling
    try:
        # Determine parser based on file extension
        def get_parser(file_path: str) -> Union[PDB.PDBParser, PDB.MMCIFParser]:
            if file_path.lower().endswith(('.cif', '.mmcif')):
                return PDB.MMCIFParser(QUIET=True)
            return PDB.PDBParser(QUIET=True)
        
        pred_parser = get_parser(pred_path)
        exp_parser = get_parser(exp_path)
        
        pred_structure = pred_parser.get_structure("predicted", pred_path)
        exp_structure = exp_parser.get_structure("experimental", exp_path)
    except Exception as e:
        raise ValueError(f"Failed to parse structure files: {e}")
    
    def calc_rmsd() -> float:
        pred_coords = []
        exp_coords = []
        
        # Get CA atoms (alpha carbons) from both structures
        for pred_model, exp_model in zip(pred_structure, exp_structure):
            for pred_chain, exp_chain in zip(pred_model, exp_model):
                for pred_res, exp_res in zip(pred_chain, exp_chain):
                    # Skip non-standard residues and handle missing atoms
                    if ('CA' in pred_res and 'CA' in exp_res and 
                        pred_res.get_resname() in PDB.Polypeptide.standard_aa_names and
                        exp_res.get_resname() in PDB.Polypeptide.standard_aa_names):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare predicted and experimental protein structures')
    parser.add_argument('-id', '--pdb_id', help='PDB ID to compare (e.g., 7jto)', default=None)
    parser.add_argument('-e', '--exp_file', help='Specific experimental file name (e.g., 7jto_pdb.cif)', default=None)
    parser.add_argument('-p', '--pred_file', help='Specific predicted file name (e.g., 7jto_af3.cif)', default=None)
    args = parser.parse_args()

    # Define paths relative to the utils directory
    exp_dir = os.path.join(os.path.dirname(__file__), "..", "data", "experimental")
    pred_dir = os.path.join(os.path.dirname(__file__), "..", "data", "predicted")

    if args.exp_file and args.pred_file:
        # Use specific filenames provided
        exp_file = os.path.join(exp_dir, args.exp_file)
        pred_file = os.path.join(pred_dir, args.pred_file)
        
        if not os.path.exists(exp_file):
            print(f"Error: Experimental file not found: {exp_file}")
            exit(1)
        if not os.path.exists(pred_file):
            print(f"Error: Predicted file not found: {pred_file}")
            exit(1)

        try:
            results = compare_structures(pred_file, exp_file)
            print(f"Results for {args.exp_file} vs {args.pred_file}:")
            print(f"RMSD: {results['rmsd']:.2f} Å")
        except Exception as e:
            print(f"Error processing files: {e}")
            exit(1)
            
    elif args.pdb_id:
        # Look for specific PDB ID
        exp_file = os.path.join(exp_dir, f"{args.pdb_id}_pdb.cif")
        pred_file = os.path.join(pred_dir, f"{args.pdb_id}_af3.cif")
        
        if not os.path.exists(exp_file):
            print(f"Error: Experimental file not found: {exp_file}")
            exit(1)
        if not os.path.exists(pred_file):
            print(f"Error: Predicted file not found: {pred_file}")
            exit(1)

        try:
            results = compare_structures(pred_file, exp_file)
            print(f"Results for {args.pdb_id}:")
            print(f"RMSD: {results['rmsd']:.2f} Å")
        except Exception as e:
            print(f"Error processing {args.pdb_id}: {e}")
            exit(1)
    else:
        # Process all matching files in the directories
        exp_files = glob(os.path.join(exp_dir, "*_pdb.cif"))
        for exp_file in exp_files:
            # Extract PDB ID from experimental filename
            pdb_id = os.path.basename(exp_file).split('_')[0]
            pred_file = os.path.join(pred_dir, f"{pdb_id}_af3.cif")
            
            if not os.path.exists(pred_file):
                print(f"Skipping {pdb_id}: No matching prediction file found")
                continue

            try:
                results = compare_structures(pred_file, exp_file)
                print(f"\nResults for {pdb_id}:")
                print(f"RMSD: {results['rmsd']:.2f} Å")
            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")
                continue