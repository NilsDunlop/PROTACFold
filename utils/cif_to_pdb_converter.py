#!/usr/bin/env python3
"""
CIF to PDB Converter

This script converts all CIF (Crystallographic Information File) files in a specified directory
to PDB (Protein Data Bank) format using PyMOL.

Usage:
    1. Set the 'input_dir' variable to the directory containing your CIF files
    2. Run the script: python cif_to_pdb_converter.py

Requirements:
    - PyMOL must be installed and accessible from Python
    - The pymol module must be importable

Input:
    - A directory containing .cif files

Output:
    - PDB files with the same base name as the input CIF files, saved in the same directory
    - For each file.cif, a corresponding file.pdb will be created

Note:
    This script uses PyMOL's command-line interface to perform the conversion.
    It will load each CIF file, save it as PDB, and then clear the PyMOL session
    before processing the next file.
"""

import os
from pymol import cmd


def convert_cif_to_pdb(input_dir: str) -> None:
    """
    Convert all CIF files in the specified directory to PDB format.
    
    Args:
        input_dir: Directory containing CIF files to convert
        
    Returns:
        None
    """
    # Change to the input directory
    os.chdir(input_dir)

    # Loop through all CIF files in the directory
    for file in os.listdir(input_dir):
        if file.endswith(".cif"):
            # Define the output PDB filename
            output_file = file.replace(".cif", ".pdb")
            
            print(f"Converting {file} to {output_file}...")
            
            # Load the CIF file into PyMOL
            cmd.load(file, "structure")
            
            # Save it as a PDB file
            cmd.save(output_file, "structure")
            
            # Clear the structure from PyMOL to avoid conflicts
            cmd.delete("structure")

    print("All conversions are complete!")


def main() -> None:
    """Main function to execute the script."""
    # Set the directory containing your CIF files
    input_dir = ""
    
    if not input_dir:
        print("Error: Please set the 'input_dir' variable to the path containing your CIF files.")
        return
    
    if not os.path.isdir(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist or is not accessible.")
        return
        
    convert_cif_to_pdb(input_dir)


if __name__ == "__main__":
    main()
