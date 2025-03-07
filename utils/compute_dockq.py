#!/usr/bin/env python3
"""
DockQ Analyzer

This script analyzes protein structure models by computing DockQ scores between
AlphaFold models and reference PDB structures.

DockQ is a quality measure for protein-protein docking models that combines RMSD-based
measures with interface contact similarity scores. This script automates the process of
computing DockQ scores for multiple structure models.

Usage:
    python compute_dockq.py

Requirements:
    - DockQ command line tool must be installed and accessible in the PATH
    - Reference PDB structures must be available

Configuration:
    - Set the appropriate directory paths in the main function for:
      - AlphaFold model files
      - Reference PDB structures
      - Output directory for results
"""

import os
import subprocess
from typing import Dict


def run_dockq(
    model_path: str, 
    reference_path: str, 
    output_json: str
) -> bool:
    """
    Run DockQ analysis for a single model.
    
    Args:
        model_path: Path to the AlphaFold model PDB file
        reference_path: Path to the reference PDB file
        output_json: Path where the output JSON will be saved
        
    Returns:
        True if analysis was successful, False otherwise
    """
    cmd = f"DockQ {model_path} {reference_path} --json {output_json}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running DockQ: {e}")
        return False


def extract_pdb_id(filename: str) -> str:
    """
    Extract PDB ID from the filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        Extracted PDB ID
    """
    if filename.startswith('fold_'):
        pdb_id = filename.split('_')[1]
    else:
        pdb_id = filename.split('_')[0]
    return pdb_id


def process_models(
    af_models_dir: str, 
    reference_dir: str, 
    output_dir: str
) -> Dict[str, bool]:
    """
    Process multiple AlphaFold models.
    
    Args:
        af_models_dir: Directory containing AlphaFold model files
        reference_dir: Directory containing reference PDB files
        output_dir: Directory where output JSON files will be saved
        
    Returns:
        Dictionary mapping PDB IDs to success status
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all AF model files
    af_files = [f for f in os.listdir(af_models_dir) if f.endswith('.pdb')]
    
    results = {}
    
    for af_file in af_files:
        # Extract PDB ID from filename
        pdb_id = extract_pdb_id(af_file)
        
        # Paths for files
        af_model_path = os.path.join(af_models_dir, af_file)
        ref_pdb_path = os.path.join(reference_dir, f"{pdb_id}.pdb")
        output_json = os.path.join(output_dir, f"{pdb_id}_dockq.json")

        # Debug prints
        print(f"\nProcessing file: {af_file}")
        print(f"Extracted PDB ID: {pdb_id}")
        print(f"Looking for reference PDB at: {ref_pdb_path}")
        print(f"AF model path: {af_model_path}")
        print(f"Output will be saved to: {output_json}")

        # Check if reference PDB exists
        if not os.path.exists(ref_pdb_path):
            print(f"❌ Reference PDB not found for {pdb_id}, skipping...")
            results[pdb_id] = False
            continue
        else:
            print(f"✓ Found reference PDB file")

        # Run DockQ analysis
        print(f"Processing {pdb_id}...")
        success = run_dockq(af_model_path, ref_pdb_path, output_json)
        results[pdb_id] = success
        
        if success:
            print(f"Completed DockQ analysis for {pdb_id}")
        else:
            print(f"Failed DockQ analysis for {pdb_id}")
    
    return results


def main() -> None:
    """
    Main function to run the DockQ analysis workflow.
    
    This function defines the directories and calls the processing function.
    """
    # Set directory paths
    af_models_dir = r"C:\Users\nilsd\Desktop\Python\Alphafold\RelatedPaper\all_models_v3\AF3\with_context"
    reference_dir = r"C:\Users\nilsd\Desktop\Python\Alphafold\RelatedPaper\all_models\PDB"
    output_dir = r"C:\Users\nilsd\Desktop\Python\Alphafold\RelatedPaper\dockq_results"
    
    # Validate paths
    if not os.path.isdir(af_models_dir):
        print(f"Error: AlphaFold models directory '{af_models_dir}' does not exist.")
        return
    
    if not os.path.isdir(reference_dir):
        print(f"Error: Reference PDB directory '{reference_dir}' does not exist.")
        return

    # Process models
    results = process_models(af_models_dir, reference_dir, output_dir)
    
    # Print summary
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    print(f"\nSummary: Successfully processed {success_count} out of {total_count} models.")


if __name__ == "__main__":
    main()
