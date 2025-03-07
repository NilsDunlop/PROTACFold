#!/usr/bin/env python3
"""
Protein Structure Superposition Tool

This script superposes AlphaFold model structures onto experimental structures 
and calculates the Root Mean Square Deviation (RMSD) between them.

Features:
- Loads experimental and predicted protein structures from mmCIF files
- Aligns structures based on matching Cα atoms
- Calculates RMSD between aligned structures
- Saves aligned model structures as PDB files

Usage:
    python superpose.py -pdb PDB_CODE [PDB_CODE ...]

Arguments:
    -pdb: One or more PDB codes to process (e.g., '7jto 2pv7')

Output:
    - Aligned model structures saved to the specified output directory
    - RMSD values printed to the console

Requirements:
    - ProDy: For structure parsing and manipulation
    - MDTraj: For RMSD calculation
    - NumPy: For coordinate handling
"""

import os
import argparse
from typing import Optional
import numpy as np
import prody as pr
import mdtraj as md


def superpose_cif(
    pdb_code: str, 
    exp_dir: str = "../data/experimental/", 
    pred_dir: str = "../data/predicted/", 
    output_dir: str = "../data/aligned/"
) -> Optional[float]:
    """
    Superpose an AlphaFold ModelCIF structure onto an experimental mmCIF structure
    and calculate RMSD using MDTraj.

    Args:
        pdb_code: The base name of the PDB file (e.g., "7jto")
        exp_dir: Directory containing experimental mmCIF files
        pred_dir: Directory containing AlphaFold ModelCIF files
        output_dir: Directory to save the aligned model

    Returns:
        RMSD value if successful, None otherwise
        
    Note:
        This function uses only Cα atoms for superposition to ensure
        consistent and biologically meaningful alignment.
    """
    # Construct full file paths
    target_cif = os.path.join(exp_dir, f"{pdb_code}_pdb.cif")
    model_cif = os.path.join(pred_dir, f"{pdb_code}_af3.cif")
    output_file = os.path.join(output_dir, f"{pdb_code}_aligned_model.pdb")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {pdb_code}...")
    print(f"Loading experimental structure from: {target_cif}")
    print(f"Loading predicted structure from: {model_cif}")

    # Load the structures
    try:
        target = pr.parseMMCIF(target_cif)
        model = pr.parseMMCIF(model_cif)
    except Exception as e:
        print(f"❌ Error loading CIF files for {pdb_code}: {e}. Skipping.")
        return None

    if target is None or model is None:
        print(f"❌ Error: Could not parse CIF files for {pdb_code}. Skipping.")
        return None

    # Select only Cα atoms
    target_ca = target.select("name CA")
    model_ca = model.select("name CA")

    if target_ca is None or model_ca is None or len(target_ca) == 0 or len(model_ca) == 0:
        print(f"❌ Error: No Cα atoms found in {pdb_code}. Skipping.")
        return None

    # Build residue number mappings for matching
    target_residues = {atom.getResnum(): atom for atom in target_ca}
    model_residues = {atom.getResnum(): atom for atom in model_ca}

    # Find common residue numbers
    common_resnums = sorted(set(target_residues.keys()) & set(model_residues.keys()))

    if len(common_resnums) == 0:
        print(f"❌ Error: No matching residues found for {pdb_code}. Skipping.")
        return None

    # Extract matched Cα atom coordinates
    target_coords = np.array([target_residues[resnum].getCoords() for resnum in common_resnums])
    model_coords = np.array([model_residues[resnum].getCoords() for resnum in common_resnums])

    print(f"✅ Using {len(target_coords)} matching Cα atoms for superposition.")

    # Calculate RMSD using MDTraj
    target_traj = md.Trajectory(target_coords[np.newaxis, :, :], topology=None)
    model_traj = md.Trajectory(model_coords[np.newaxis, :, :], topology=None)

    # Superpose model onto target using MDTraj
    rmsd = md.rmsd(model_traj, target_traj)[0]

    try:
        # Save aligned model using ProDy
        pr.writePDB(output_file, model)
        print(f"✅ Superposition RMSD: {rmsd:.3f} Å")
        print(f"✅ Aligned model saved as {output_file}")
    except Exception as e:
        print(f"❌ Error saving aligned model: {e}")
        return rmsd
    
    return rmsd


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Superpose AlphaFold3 ModelCIF structures onto experimental PDB structures and compute RMSD"
    )
    parser.add_argument(
        "-pdb", 
        nargs="+", 
        required=True, 
        help="One or more PDB codes (e.g. '7jto 2pv7')"
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Main function to run the superposition workflow.
    
    This function:
    1. Parses command line arguments
    2. Processes each PDB code
    3. Reports results
    """
    args = parse_arguments()
    
    # Collect results
    results = {}
    
    # Process each PDB code separately
    for pdb_code in args.pdb:
        pdb_code = pdb_code.lower()  # Ensure lowercase for consistency
        rmsd = superpose_cif(pdb_code)
        results[pdb_code] = rmsd
    
    # Print summary
    successful = sum(1 for rmsd in results.values() if rmsd is not None)
    print(f"\nSummary: Successfully processed {successful} out of {len(results)} structures.")
    
    if successful > 0:
        avg_rmsd = np.mean([rmsd for rmsd in results.values() if rmsd is not None])
        print(f"Average RMSD: {avg_rmsd:.3f} Å")


if __name__ == "__main__":
    main()