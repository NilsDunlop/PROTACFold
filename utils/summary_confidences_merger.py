#!/usr/bin/env python3
"""
Summary Confidences Merger

This script walks through a directory structure containing AlphaFold3 confidence files
and merges them into a single Excel spreadsheet for easier analysis.

Usage:
    python summary_confidences_merger.py --base_dir /path/to/results --output /path/to/output.xlsx

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

import os
import json
import argparse
from typing import Dict, List, Any
import pandas as pd


def process_confidence_files(base_dir: str) -> List[Dict[str, Any]]:
    """
    Process AlphaFold3 confidence files and extract relevant data.
    
    Args:
        base_dir: Root directory containing AlphaFold3 results
        
    Returns:
        List of dictionaries containing extracted data
    """
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
    
    return rows


def create_excel_report(rows: List[Dict[str, Any]], output_path: str) -> None:
    """
    Create an Excel report from the processed data.
    
    Args:
        rows: List of dictionaries containing the extracted data
        output_path: Path where the Excel file will be saved
        
    Returns:
        None
    """
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Name"], key=lambda col: col.str.upper())

    # Write the DataFrame to an Excel file
    df.to_excel(output_path, index=False)
    print(f"Excel file written to {output_path}")


def main() -> None:
    """Main function to execute the script."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Merge AlphaFold3 confidence files into a single Excel spreadsheet."
    )
    parser.add_argument(
        "--base_dir", 
        type=str, 
        required=True,
        help="Base directory containing AlphaFold3 results"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=False,
        help="Output path for the Excel file (default: <base_dir>/summary_confidences.xlsx)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    base_dir = args.base_dir
    
    # If output path is not specified, create a default path in the base directory
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(base_dir, "summary_confidences.xlsx")
    
    if not os.path.isdir(base_dir):
        print(f"Error: The directory '{base_dir}' does not exist or is not accessible.")
        return
    
    # Process confidence files
    rows = process_confidence_files(base_dir)
    
    if not rows:
        print("No confidence files found. Please check your directory structure.")
        return
    
    # Create and save the Excel report
    create_excel_report(rows, output_path)


if __name__ == "__main__":
    main()