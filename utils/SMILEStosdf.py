#!/usr/bin/env python3

import os
import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import tqdm  
import argparse
import logging 
import multiprocessing
from functools import partial
import time

def sanitize_filename(filename: str) -> str:
    """
    Remove or replace characters that might not be safe in filenames.
    E.g., spaces, slashes, etc.
    """
    # Replace any non-word character (anything not in a-zA-Z0-9_) with an underscore.
    return re.sub(r"[^\w\-]+", "_", filename)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert SMILES to 3D SDF files')
    parser.add_argument('--csv', type=str, help='Input CSV file path (optional)')
    parser.add_argument('--outdir', type=str, help='Output directory (optional)')
    parser.add_argument('--seed', type=int, default=0xF00D, help='Random seed for conformer generation')
    parser.add_argument('--optimize', action='store_true', help='Perform UFF optimization')
    return parser.parse_args()

def generate_conformers_with_fallback(mol, n_conf=1, seed=0xF00D, max_attempts=3):
    """Generate conformers with multiple fallback methods if initial attempt fails."""
    mol = Chem.AddHs(mol) 
    start_time = time.time()
    method_used = None

    # First try: ETKDGv3 (current method)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params)
    if cids:
        method_used = "ETKDGv3"
        for cid in cids:
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
        end_time = time.time()
        logging.debug(f"Generated conformer using {method_used} in {end_time - start_time:.2f} seconds")
        return mol, method_used
        
    # First fallback: Try with different seeds
    for attempt in range(max_attempts):
        params.randomSeed = seed + attempt + 1
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params)
        if cids:
            method_used = f"ETKDGv3_attempt_{attempt + 2}"
            for cid in cids:
                AllChem.UFFOptimizeMolecule(mol, confId=cid)
            end_time = time.time()
            logging.debug(f"Generated conformer using {method_used} in {end_time - start_time:.2f} seconds")
            return mol, method_used
            
    # Second fallback: Try ETKDG
    params = AllChem.ETKDG()
    params.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params)
    if cids:
        method_used = "ETKDG"
        for cid in cids:
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
        end_time = time.time()
        logging.debug(f"Generated conformer using {method_used} in {end_time - start_time:.2f} seconds")
        return mol, method_used
        
    # Third fallback: Try basic distance geometry
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf)
    if cids:
        method_used = "BasicDG"
        for cid in cids:
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
        end_time = time.time()
        logging.debug(f"Generated conformer using {method_used} in {end_time - start_time:.2f} seconds")
        return mol, method_used
    
    end_time = time.time()
    logging.warning(f"Failed to generate conformer after {end_time - start_time:.2f} seconds")
    return None, None

def setup_logging():
    # Create logs directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up logging with file in logs directory
    log_file = os.path.join(logs_dir, "smiles_to_sdf.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

def process_molecule(row, output_dir, seed):
    """Process a single molecule (to be run in parallel)"""
    smiles = row["Smiles"]
    compound_id = str(row["Compound ID"])

    # Convert to RDKit Mol
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.error(f"Failed conversion - Invalid SMILES | Compound ID: {compound_id} | SMILES: {smiles}")
        return {"status": "failed", "reason": "invalid_smiles", "compound_id": compound_id, "smiles": smiles}

    # Generate conformers with fallback methods
    mol_result, method_used = generate_conformers_with_fallback(mol, n_conf=1, seed=seed)
    if mol_result is None:
        logging.error(f"Failed conversion - Conformer generation failed | Compound ID: {compound_id} | SMILES: {smiles}")
        return {"status": "failed", "reason": "conformer_generation", "compound_id": compound_id, "smiles": smiles}

    # Set internal properties
    mol_result.SetProp("_Name", compound_id)
    mol_result.SetProp("_ConformerGenerationMethod", method_used)

    # Build the output filename
    safe_id = sanitize_filename(compound_id)
    output_filename = os.path.join(output_dir, f"{safe_id}.sdf")

    # Write SDF
    with Chem.SDWriter(output_filename) as writer:
        writer.write(mol_result)

    return {"status": "success", "method": method_used, "compound_id": compound_id, "filename": output_filename}

def main():
    setup_logging()
    logging.info("Starting SMILES to SDF conversion")
    args = parse_args()
    
    # Get the project root directory (1 level up from this script, not 2)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Changed: removed one dirname call
    
    # Use command line args if provided, otherwise use defaults
    csv_path = args.csv if args.csv else os.path.join(project_root, "data", "PROTAC-Degradation-DB-Updated.csv")
    output_dir = args.outdir if args.outdir else os.path.join(project_root, "data", "sdf")
    
    # Log the paths to help with debugging
    logging.info(f"Looking for CSV file at: {csv_path}")
    logging.info(f"Output directory will be: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Ensure the columns exist
    if "Smiles" not in df.columns:
        raise ValueError("CSV does not contain a 'Smiles' column.")
    if "Compound ID" not in df.columns:
        raise ValueError("CSV does not contain a 'Compound ID' column (adjust script if needed).")

    # We only care about rows that actually have a SMILES
    df = df.dropna(subset=["Smiles"])
    logging.info(f"Found {df.shape[0]} total SMILES entries")

    # Find and log duplicates before dropping them
    duplicates = df[df.duplicated(subset=["Smiles"], keep="first")]
    if not duplicates.empty:
        logging.info("\nRemoving duplicate SMILES entries (keeping first occurrence):")
        logging.info(f"Number of duplicates removed: {len(duplicates)}")
        logging.info("\nDuplicate entries removed:")
        for _, row in duplicates.iterrows():
            logging.info(f"Compound ID: {row['Compound ID']}")
            logging.info(f"SMILES: {row['Smiles']}")
            logging.info("---")
        
        # Save duplicates to CSV
        duplicates_csv = os.path.join(os.path.dirname(output_dir), "duplicate_entries.csv")
        duplicates.to_csv(duplicates_csv, index=False)
        logging.info(f"\nDuplicate entries have been saved to: {duplicates_csv}")

    # Drop duplicates by Smiles, keeping the *first* occurrence
    df_unique = df.drop_duplicates(subset=["Smiles"], keep="first")
    logging.info(f"\nProcessing {df_unique.shape[0]} unique SMILES entries")

    # Set up parallel processing
    num_cores = multiprocessing.cpu_count() - 1  # Leave one core free
    logging.info(f"Using {num_cores} CPU cores for parallel processing")

    # Create partial function with fixed arguments
    process_mol_partial = partial(process_molecule, 
                                output_dir=output_dir, 
                                seed=args.seed)

    # Process molecules in parallel with progress bar
    start_time = time.time()
    with multiprocessing.Pool(num_cores) as pool:
        results = list(tqdm.tqdm(
            pool.imap(process_mol_partial, [row for _, row in df_unique.iterrows()]),
            total=df_unique.shape[0],
            desc="Converting SMILES to SDF"
        ))

    # Analyze results
    successful = [r for r in results if r and r["status"] == "success"]
    failed = [r for r in results if r and r["status"] == "failed"]
    
    # Group failures by reason
    invalid_smiles = [r for r in failed if r["reason"] == "invalid_smiles"]
    failed_conformers = [r for r in failed if r["reason"] == "conformer_generation"]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Log summary statistics
    logging.info(f"Successfully converted {len(successful)} molecules")
    logging.info(f"Failed to convert {len(failed)} molecules")
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    logging.info(f"Average time per molecule: {total_time/df_unique.shape[0]:.2f} seconds")
    
    # Log detailed failure information
    if invalid_smiles:
        logging.info("\nMolecules with invalid SMILES:")
        for fail in invalid_smiles:
            logging.info(f"Compound ID: {fail['compound_id']}")
            logging.info(f"SMILES: {fail['smiles']}")
            logging.info("---")
    
    if failed_conformers:
        logging.info("\nMolecules where conformer generation failed:")
        for fail in failed_conformers:
            logging.info(f"Compound ID: {fail['compound_id']}")
            logging.info(f"SMILES: {fail['smiles']}")
            logging.info("---")

    # Optionally save failed molecules to a CSV for further analysis
    if failed:
        failed_df = pd.DataFrame(failed)
        failed_csv = os.path.join(os.path.dirname(output_dir), "failed_conversions.csv")
        failed_df.to_csv(failed_csv, index=False)
        logging.info(f"\nFailed conversions have been saved to: {failed_csv}")

if __name__ == "__main__":
    main()
