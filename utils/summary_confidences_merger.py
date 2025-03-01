"""
Summary Confidences Merger

This script walks through a directory structure containing AlphaFold3 confidence files
and merges them into a single Excel spreadsheet for easier analysis.

Usage:
    1. Set the 'base_dir' variable to the root directory containing your AlphaFold3 results
    2. Set the 'output_path' variable to your desired output Excel file location
    3. Run the script: python summary_confidences_merger.py

Expected directory structure:
    base_dir/
    ├── PDB_ID_1/
    │   ├── ccd/
    │   │   └── *_ccd_summary_confidences.json
    │   └── smiles/
    │       └── *_smiles_summary_confidences.json
    ├── PDB_ID_2/
    │   ├── ccd/
    │   │   └── *_ccd_summary_confidences.json
    │   └── smiles/
    │       └── *_smiles_summary_confidences.json
    └── ...

Output:
    An Excel file containing a table with the following columns:
    - Name: Combined identifier in format PDB_ID_TYPE (e.g., 8DSO_CCD)
    - fraction_disordered: Fraction of disordered residues
    - has_clash: Whether the structure has clashes
    - iptm: Interface predicted TM-score
    - ptm: Predicted TM-score
    - ranking_score: AlphaFold3 ranking score
"""

# Python
import os
import json
import pandas as pd

# Set the base directory containing AlphaFold3 results
base_dir = "" 

rows = []

# Walk through all subdirectories
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith("_ccd_summary_confidences.json") or file.endswith("_smiles_summary_confidences.json"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # Extract required keys (they might be missing in some files)
            keys = ["fraction_disordered", "has_clash", "iptm", "ptm", "ranking_score"]
            extracted = {k: data.get(k, None) for k in keys}

            # Expected structure: /AF3_Feb/<Folder>/<Subfolder>/summary_confidences.json
            parts = os.path.normpath(file_path).split(os.sep)
            folder_name = parts[-3].upper() if len(parts) >= 3 else ""

            # Determine Type: use the subfolder name which is one level up from the file
            subfolder_name = parts[-2].lower() if len(parts) >= 2 else ""
            if "ccd" in subfolder_name:
                typ = "CCD"
            elif "smiles" in subfolder_name:
                typ = "SMILES"
            else:
                typ = "Unknown"

            # Create a combined Name column in the format: MAINFOLDER_TYP (e.g. 8DSO_CCD, 8DSO_SMILES)
            name = f"{folder_name}_{typ}"
            row = {"Name": name}
            row.update(extracted)
            rows.append(row)


df = pd.DataFrame(rows)
df = df.sort_values(by=["Name"], key=lambda col: col.str.upper())

# Write the DataFrame to an Excel file
output_path = "/summary_confidences.xlsx"
df.to_excel(output_path, index=False)
print(f"Excel file written to {output_path}")