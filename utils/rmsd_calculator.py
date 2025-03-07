#!/usr/bin/env python3
"""
RMSD Calculator for PyMOL

This script calculates the Root Mean Square Deviation (RMSD) between protein 
structures in PyMOL, specifically comparing models generated with different 
approaches (Original vs SMILES vs CCD).

The script automatically:
1. Detects loaded models based on naming conventions
2. Finds the relevant chains in each model based on sequence matching
3. Creates selections for each chain
4. Aligns the models and calculates RMSD values

Usage:
    1. Modify the POI_SEQUENCE and E3_SEQUENCE variables below to match your proteins
    2. Load this script in PyMOL: run utils/rmsd_calculator.py
    3. Run the command in PyMOL: calculate_rmsd

Requirements:
    - PyMOL with Python interface
    - Loaded structural models (original, SMILES, and CCD)
"""

from typing import Tuple, List, Optional
from pymol import cmd

# ------------------------------------------------------------------------------
# Provide your amino acid chain sequences here.
#
# IMPORTANT: If these variables are left blank the script will exit.
# ------------------------------------------------------------------------------
POI_SEQUENCE = """MHHHHHHSSGRENLYFQGTQSKPTPVKPNYALKFTLAGHTKAVSSVKFSPNGEWLASSSADKLIKIWGAYDGKFEKTISGHKLGISDVAWSSDSNLLVSASDDKTLKIWDVSSGKCLKTLKGHSNYVFCCNFNPQSNLIVSGSFDESVRIWDVKTGKCLKTLPAHSDPVSAVHFNRDGSLIVSSSYDGLCRIWDTASGQCLKTLIDDDNPPVSFVKFSPNGKYILAATLDNTLKLWDYSKGKCLKTYTGHKNEKYCIFANFSVTGGKWIVSGSEDNLVYIWNLQTKEIVQKLQGHTDVVISTACHPTENIIASAALENDKTIKLWKSDC"""

E3_SEQUENCE = """MGSSHHHHHHSSGRENLYFQGSSRASAFRPISVFREANEDESGFTCCAFSARERFLMLGTCTGQLKLYNVFSGQEEASYNCHNSAITHLEPSRDGSLLLTSATWSQPLSALWGMKSVFDMKHSFTEDHYVEFSKHSQDRVIGTKGDIAHIYDIQTGNKLLTLFNPDLANNYKRNCATFNPTDDLVLNDGVLWDVRSAQAIHKFDKFNMNISGVFHPNGLEVIINTEIWDLRTFHLLHTVPALDQCRVVFNHTGTVMYGAMLQADDEDDLMEERMKSPFGSSFRTFNATDYKPIATIDVKRNIFDLCTDTKDCYLAVIENQGSMDALNMDTVCRLYEVG"""

# ------------------------------------------------------------------------------
# Helper function: Look for a chain that contains the given sequence.
#
# Both the provided sequence and the FASTA string from PyMOL have whitespace
# (newlines/spaces) removed to avoid formatting issues.
# ------------------------------------------------------------------------------
def find_chain_by_sequence(sequence: str, model_name: str) -> str:
    """
    Searches for a chain within the specified model that contains the given
    amino acid sequence.
    
    Args:
        sequence: Amino acid sequence to search for
        model_name: Name of the PyMOL model/object
        
    Returns:
        The chain ID if found
        
    Raises:
        ValueError: If the sequence is not found in any chain
    """
    seq_clean = "".join(sequence.split())  # Remove all whitespace
    for chain in cmd.get_chains(model_name):
        fasta = cmd.get_fastastr(f"{model_name} and chain {chain}")
        fasta_clean = "".join(fasta.split())
        if seq_clean in fasta_clean:
            return chain
    raise ValueError(f"Sequence not found in model '{model_name}'.")

# ------------------------------------------------------------------------------
# Main routine: Automatically detect the models, find the target chains (POI and
# E3) in each, create selections and compute the RMSDs using align.
#
# The names of the models are determined from the filename:
#   - Original model: name does NOT contain 'smiles_model' or 'ccd_model'
#   - Smiles model: name contains 'smiles_model'
#   - CCD model: name contains 'ccd_model'
# ------------------------------------------------------------------------------
def calculate_rmsd() -> None:
    """
    Main function to calculate RMSD between protein structures.
    
    This function:
    1. Detects the loaded models
    2. Validates input sequences
    3. Processes the POI and E3 chains
    4. Calculates and reports RMSD values
    """
    try:
        # Check sequences
        if not POI_SEQUENCE.strip():
            print("Error: POI sequence is empty. Please provide a sequence in the script.")
            return
        if not E3_SEQUENCE.strip():
            print("Error: E3 sequence is empty. Please provide a sequence in the script.")
            return
            
        # Detect models
        original_model, smiles_model, ccd_model = detect_models()
        print(f"Identified models: Original='{original_model}', Smiles='{smiles_model}', CCD='{ccd_model}'")
        
        # Process POI
        process_protein_chain("POI", POI_SEQUENCE, original_model, smiles_model, ccd_model)
        
        # Process E3
        process_protein_chain("E3", E3_SEQUENCE, original_model, smiles_model, ccd_model)
        
    except ValueError as e:
        print(f"Error: {e}")
        return

def detect_models() -> Tuple[str, str, str]:
    """
    Auto-detect loaded models based on naming conventions.
    
    Returns:
        Tuple containing (original_model, smiles_model, ccd_model)
        
    Raises:
        ValueError: If the expected models cannot be found
    """
    # Get all loaded objects (models) in PyMOL
    models = cmd.get_object_list("all")
    if not models:
        raise ValueError("No models loaded. Please load your model files first.")

    # Auto-detect model names based on naming conventions
    orig_models = [m for m in models if "smiles_model" not in m and "ccd_model" not in m]
    smiles_models = [m for m in models if "smiles_model" in m]
    ccd_models = [m for m in models if "ccd_model" in m]

    if len(orig_models) != 1:
        raise ValueError(f"Expected exactly one original model, found: {', '.join(orig_models)}")
    if len(smiles_models) != 1:
        raise ValueError(f"Expected exactly one smiles model, found: {', '.join(smiles_models)}")
    if len(ccd_models) != 1:
        raise ValueError(f"Expected exactly one ccd model, found: {', '.join(ccd_models)}")

    return orig_models[0], smiles_models[0], ccd_models[0]

def process_protein_chain(
    protein_type: str, 
    sequence: str, 
    original_model: str, 
    smiles_model: str, 
    ccd_model: str
) -> Tuple[float, float]:
    """
    Process a protein chain by creating selections and calculating RMSD values.
    
    Args:
        protein_type: Type of protein (e.g., "POI" or "E3")
        sequence: Amino acid sequence of the protein
        original_model: Name of the original model
        smiles_model: Name of the SMILES model
        ccd_model: Name of the CCD model
        
    Returns:
        Tuple of (rmsd_smiles, rmsd_ccd) values
        
    Raises:
        ValueError: If a required chain cannot be found
    """
    print(f"\n--- Processing {protein_type} chain ---")
    
    # Find chains by sequence
    try:
        chain_orig = find_chain_by_sequence(sequence, original_model)
        chain_smiles = find_chain_by_sequence(sequence, smiles_model)
        chain_ccd = find_chain_by_sequence(sequence, ccd_model)
    except ValueError as e:
        raise ValueError(f"{protein_type} error: {e}")

    # Create selections
    sel_orig = f"chain{chain_orig}_{protein_type}"
    sel_smiles = f"chain{chain_smiles}_{protein_type}_smiles"
    sel_ccd = f"chain{chain_ccd}_{protein_type}_ccd"

    cmd.select(sel_orig, f"{original_model} and chain {chain_orig}")
    cmd.select(sel_smiles, f"{smiles_model} and chain {chain_smiles}")
    cmd.select(sel_ccd, f"{ccd_model} and chain {chain_ccd}")

    # Align structures and calculate RMSD
    rmsd_smiles = cmd.align(sel_smiles, sel_orig)[0]
    rmsd_ccd = cmd.align(sel_ccd, sel_orig)[0]

    # Print results
    print(f"\nRMSD for {protein_type} (smiles vs. original): {rmsd_smiles:.3f} Å")
    print(f"RMSD for {protein_type} (ccd vs. original): {rmsd_ccd:.3f} Å")
    
    return rmsd_smiles, rmsd_ccd

# ------------------------------------------------------------------------------
# Register as a PyMOL command.
# After running this script (e.g. run pymol_scripts/calculate_rmsd.py) in PyMOL,
# simply type "calculate_rmsd" in the PyMOL command prompt.
# ------------------------------------------------------------------------------
cmd.extend("calculate_rmsd", calculate_rmsd)