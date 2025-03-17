import os
import sys
import json
import argparse
import subprocess
import pandas as pd
import re
import pymol
from pymol import cmd

# Initialize PyMol in headless mode quiet mode
pymol.finish_launching(['pymol', '-qc'])

def extract_dockq_values(output):
    """Extract DockQ, iRMSD, and LRMSD values from DockQ output."""
    dockq_score = None
    irmsd = None
    lrmsd = None
    
    dockq_match = re.search(r'DockQ:\s+(\d+\.\d+)', output)
    irmsd_match = re.search(r'iRMSD:\s+(\d+\.\d+)', output)
    lrmsd_match = re.search(r'LRMSD:\s+(\d+\.\d+)', output)
    
    if dockq_match:
        dockq_score = float(dockq_match.group(1))
    if irmsd_match:
        irmsd = float(irmsd_match.group(1))
    if lrmsd_match:
        lrmsd = float(lrmsd_match.group(1))
        
    return dockq_score, irmsd, lrmsd

def extract_confidence_values(json_file):
    """Extract specific confidence values from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    fraction_disordered = data.get('fraction_disordered', None)
    has_clash = data.get('has_clash', None)
    iptm = data.get('iptm', None)
    ptm = data.get('ptm', None)
    ranking_score = data.get('ranking_score', None)
    
    return fraction_disordered, has_clash, iptm, ptm, ranking_score

def compute_rmsd_with_pymol(model_path, ref_path, pdb_id, model_type):
    """Compute RMSD using PyMol."""
    # Load structures
    model_name = f"{pdb_id}_{model_type}_model"
    ref_name = pdb_id
    cmd.load(model_path, model_name)
    cmd.load(ref_path, ref_name)
    
    # Compute RMSD
    alignment_name = f"aln_{model_name}_to_{ref_name}"
    rmsd = cmd.align(f"polymer and name CA and ({model_name})", 
                    f"polymer and name CA and ({ref_name})", 
                    quiet=0, 
                    object=alignment_name, 
                    reset=1)[0]
    
    # Clean up
    cmd.delete("all")
    
    return rmsd

def run_dockq(model_path, ref_path):
    """Run DockQ and return the output."""
    try:
        result = subprocess.run(
            ["DockQ", model_path, ref_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running DockQ: {e}")
        return None

def process_pdb_folder(folder_path, pdb_id, results):
    """Process a single PDB ID folder."""
    ref_path = os.path.join(folder_path, f"{pdb_id}.cif")
    
    # Check if reference file exists
    if not os.path.exists(ref_path):
        print(f"Reference file {ref_path} not found. Skipping {pdb_id}.")
        return
    
    # Process SMILES model
    smiles_folder = os.path.join(folder_path, f"{pdb_id.lower()}_ternary_smiles")
    smiles_model_path = os.path.join(smiles_folder, f"{pdb_id.lower()}_ternary_smiles_model.cif")
    smiles_json_path = os.path.join(smiles_folder, f"{pdb_id.lower()}_ternary_smiles_summary_confidences.json")
    
    # Process CCD model
    ccd_folder = os.path.join(folder_path, f"{pdb_id.lower()}_ternary_ccd")
    ccd_model_path = os.path.join(ccd_folder, f"{pdb_id.lower()}_ternary_ccd_model.cif")
    ccd_json_path = os.path.join(ccd_folder, f"{pdb_id.lower()}_ternary_ccd_summary_confidences.json")
    
    result_row = {"PDB_ID": pdb_id}
    
    # SMILES processing
    if os.path.exists(smiles_model_path):
        try:
            # Compute RMSD
            smiles_rmsd = compute_rmsd_with_pymol(smiles_model_path, ref_path, pdb_id, "ternary_smiles")
            result_row["SMILES RMSD"] = smiles_rmsd
            
            # Run DockQ
            smiles_dockq_output = run_dockq(smiles_model_path, ref_path)
            if smiles_dockq_output:
                dockq_score, irmsd, lrmsd = extract_dockq_values(smiles_dockq_output)
                result_row["SMILES DOCKQ SCORE"] = dockq_score
                result_row["SMILES DOCKQ iRMSD"] = irmsd
                result_row["SMILES DOCKQ LRMSD"] = lrmsd
        except Exception as e:
            print(f"Error processing SMILES model for {pdb_id}: {e}")
        
        # Extract confidence values
        if os.path.exists(smiles_json_path):
            try:
                fraction_disordered, has_clash, iptm, ptm, ranking_score = extract_confidence_values(smiles_json_path)
                result_row["SMILES FRACTION DISORDERED"] = fraction_disordered
                result_row["SMILES HAS_CLASH"] = has_clash
                result_row["SMILES IPTM"] = iptm
                result_row["SMILES PTM"] = ptm
                result_row["SMILES RANKING_SCORE"] = ranking_score
            except Exception as e:
                print(f"Error extracting SMILES confidence values for {pdb_id}: {e}")
    
    # CCD processing
    if os.path.exists(ccd_model_path):
        try:
            # Compute RMSD
            ccd_rmsd = compute_rmsd_with_pymol(ccd_model_path, ref_path, pdb_id, "ternary_ccd")
            result_row["CCD RMSD"] = ccd_rmsd
            
            # Run DockQ
            ccd_dockq_output = run_dockq(ccd_model_path, ref_path)
            if ccd_dockq_output:
                dockq_score, irmsd, lrmsd = extract_dockq_values(ccd_dockq_output)
                result_row["CCD DOCKQ SCORE"] = dockq_score
                result_row["CCD DOCKQ iRMSD"] = irmsd
                result_row["CCD DOCKQ LRMSD"] = lrmsd
        except Exception as e:
            print(f"Error processing CCD model for {pdb_id}: {e}")
        
        # Extract confidence values
        if os.path.exists(ccd_json_path):
            try:
                fraction_disordered, has_clash, iptm, ptm, ranking_score = extract_confidence_values(ccd_json_path)
                result_row["CCD FRACTION DISORDERED"] = fraction_disordered
                result_row["CCD HAS_CLASH"] = has_clash
                result_row["CCD IPTM"] = iptm
                result_row["CCD PTM"] = ptm
                result_row["CCD RANKING_SCORE"] = ranking_score
            except Exception as e:
                print(f"Error extracting CCD confidence values for {pdb_id}: {e}")
    
    results.append(result_row)

def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaFold predictions against experimental structures.")
    parser.add_argument("folder", help="Path to the folder containing PDB ID folders")
    parser.add_argument("--output", default="../data/af3_results/evaluation_results.csv", help="Output CSV file name")
    args = parser.parse_args()
    
    # Check if the folder exists
    if not os.path.exists(args.folder):
        print(f"Error: Folder {args.folder} does not exist.")
        sys.exit(1)
    
    results = []
    
    # Process each PDB ID folder
    for item in os.listdir(args.folder):
        folder_path = os.path.join(args.folder, item)
        if os.path.isdir(folder_path):
            print(f"Processing {item}...")
            process_pdb_folder(folder_path, item, results)
    
    # Create dataframe and save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()