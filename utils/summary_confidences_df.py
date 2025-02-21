# Python
import os
import json
import pandas as pd

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