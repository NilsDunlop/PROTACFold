import prody as pr
import mdtraj as md
import os
import numpy as np
import argparse

def superpose_cif(pdb_code, exp_dir="../data/experimental/", pred_dir="../data/predicted/", output_dir="../data/aligned/"):
    """
    Superposes an AlphaFold ModelCIF structure onto an experimental mmCIF structure
    and calculates RMSD using MDTraj.

    - pdb_code: The base name of the PDB file (e.g., "7jto").
    - exp_dir: Directory containing experimental mmCIF files.
    - pred_dir: Directory containing AlphaFold ModelCIF files.
    - output_dir: Directory to save the aligned model.
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
    target = pr.parseMMCIF(target_cif)
    model = pr.parseMMCIF(model_cif)

    if target is None or model is None:
        print(f"❌ Error: Could not parse CIF files for {pdb_code}. Skipping.")
        return

    # Select only Cα atoms
    target_ca = target.select("name CA")
    model_ca = model.select("name CA")

    if target_ca is None or model_ca is None:
        print(f"❌ Error: No Cα atoms found in {pdb_code}. Skipping.")
        return

    # Build residue number mappings for matching
    target_residues = {atom.getResnum(): atom for atom in target_ca}
    model_residues = {atom.getResnum(): atom for atom in model_ca}

    # Find common residue numbers
    common_resnums = sorted(set(target_residues.keys()) & set(model_residues.keys()))

    if len(common_resnums) == 0:
        print(f"❌ Error: No matching residues found for {pdb_code}. Skipping.")
        return

    # Extract matched Cα atom coordinates
    target_coords = np.array([target_residues[resnum].getCoords() for resnum in common_resnums])
    model_coords = np.array([model_residues[resnum].getCoords() for resnum in common_resnums])

    print(f"✅ Using {len(target_coords)} matching Cα atoms for superposition.")

    # Calculate RMSD using MDTraj
    target_traj = md.Trajectory(target_coords[np.newaxis, :, :], topology=None)
    model_traj = md.Trajectory(model_coords[np.newaxis, :, :], topology=None)

    # Superpose model onto target using MDTraj
    rmsd = md.rmsd(model_traj, target_traj)[0]

    # Save aligned model using ProDy
    pr.writePDB(output_file, model)

    print(f"✅ Superposition RMSD: {rmsd:.3f} Å")
    print(f"✅ Aligned model saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Superpose AlphaFold3 ModelCIF structures onto experimental PDB structures and compute RMSD")
    parser.add_argument("-pdb", nargs="+", required=True, help="One or more PDB codes (e.g. '7jto 2pv7')")
    
    args = parser.parse_args()

    # Process each PDB code separately
    for pdb_code in args.pdb:
        superpose_cif(pdb_code)