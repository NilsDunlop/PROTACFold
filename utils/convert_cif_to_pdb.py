import os
from pymol import cmd

# Set the directory containing your CIF files
input_dir = "/Users/faerazo/Downloads/LIGANDS"

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
