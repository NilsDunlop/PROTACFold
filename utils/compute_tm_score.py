import os
import subprocess
import re
import sys
import glob

def extract_tmalign_results(output):
    """Extract relevant information from TMalign output."""
    results = {}
    
    # Extract alignment info
    align_match = re.search(r'Aligned length=\s+(\d+),\s+RMSD=\s+([0-9.]+),\s+Seq_ID=([^\n]+)', output)
    if align_match:
        results['Aligned_length'] = align_match.group(1)
        results['RMSD'] = align_match.group(2)
        results['Seq_ID'] = align_match.group(3)
    
    # Extract TM-scores
    tm_score1_match = re.search(r'TM-score=\s+([0-9.]+)\s+\(if normalized by length of Chain_1\)', output)
    if tm_score1_match:
        results['TM_score_Chain_1'] = tm_score1_match.group(1)
    
    tm_score2_match = re.search(r'TM-score=\s+([0-9.]+)\s+\(if normalized by length of Chain_2\)', output)
    if tm_score2_match:
        results['TM_score_Chain_2'] = tm_score2_match.group(1)
    
    return results

def run_tmalign(pred_file, exp_file):
    """Run TMalign and return the output."""
    try:
        result = subprocess.run(
            ["TMalign", pred_file, exp_file], 
            check=True, 
            text=True, 
            capture_output=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running TMalign: {e}")
        return None
    except FileNotFoundError:
        print("TMalign executable not found. Make sure it's in your PATH.")
        return None

def process_pdbid_folder(pdbid_folder, results_dir):
    """Process a single PDBID folder."""
    # Extract PDBID from folder name
    pdbid = os.path.basename(pdbid_folder)
    inner_folder = os.path.join(pdbid_folder, pdbid)
    
    if not os.path.isdir(inner_folder):
        print(f"Warning: Expected inner folder {inner_folder} not found. Skipping.")
        return
    
    # Find experimental structure file (case insensitive)
    exp_files = []
    for pattern in [f"{pdbid.lower()}.pdb", f"{pdbid.upper()}.pdb"]:
        exp_file_path = os.path.join(inner_folder, pattern)
        if os.path.isfile(exp_file_path):
            exp_files.append(exp_file_path)
    
    if not exp_files:
        print(f"Warning: Experimental structure file not found for {pdbid}. Skipping.")
        return
    
    exp_file = exp_files[0]
    
    # Find prediction files
    ccd_file = os.path.join(inner_folder, f"{pdbid.lower()}_ccd_model.pdb")
    smiles_file = os.path.join(inner_folder, f"{pdbid.lower()}_smiles_model.pdb")
    
    if not os.path.isfile(ccd_file):
        print(f"Warning: CCD prediction file not found for {pdbid}.")
        ccd_file = None
    
    if not os.path.isfile(smiles_file):
        print(f"Warning: SMILES prediction file not found for {pdbid}.")
        smiles_file = None
    
    if not ccd_file and not smiles_file:
        print(f"Error: No prediction files found for {pdbid}. Skipping.")
        return
    
    # Prepare output file
    output_file = os.path.join(results_dir, f"{pdbid.lower()}_TMalign_results.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"TMalign Results for {pdbid}\n")
        f.write("="*50 + "\n\n")
        
        # Process SMILES prediction
        if smiles_file:
            f.write("SMILES Prediction vs Experimental Structure:\n")
            f.write("-"*50 + "\n")
            
            tmalign_output = run_tmalign(smiles_file, exp_file)
            if tmalign_output:
                results = extract_tmalign_results(tmalign_output)
                
                # Write results in a specific order with full alignment information
                f.write(f"Aligned length= {results.get('Aligned_length', 'N/A')}, RMSD= {results.get('RMSD', 'N/A')}, Seq_ID={results.get('Seq_ID', 'N/A')}\n")
                f.write(f"TM-score= {results.get('TM_score_Chain_1', 'N/A')} (if normalized by length of Chain_1)\n")
                f.write(f"TM-score= {results.get('TM_score_Chain_2', 'N/A')} (if normalized by length of Chain_2)\n")
                
            else:
                f.write("Failed to run TMalign for SMILES prediction.\n")

        # Process CCD prediction
        if ccd_file:
            f.write("\nCCD Prediction vs Experimental Structure:\n")
            f.write("-"*50 + "\n")
            
            tmalign_output = run_tmalign(ccd_file, exp_file)
            if tmalign_output:
                results = extract_tmalign_results(tmalign_output)
                
                # Write results in a specific order with full alignment information
                f.write(f"Aligned length= {results.get('Aligned_length', 'N/A')}, RMSD= {results.get('RMSD', 'N/A')}, Seq_ID={results.get('Seq_ID', 'N/A')}\n")
                f.write(f"TM-score= {results.get('TM_score_Chain_1', 'N/A')} (if normalized by length of Chain_1)\n")
                f.write(f"TM-score= {results.get('TM_score_Chain_2', 'N/A')} (if normalized by length of Chain_2)\n")
                
            else:
                f.write("Failed to run TMalign for CCD prediction.\n")
            
            f.write("\n")
    
    print(f"Processed {pdbid}. Results saved to {output_file}")

def main(folder_path):
    """Main function to process all PDBID folders."""
    if not os.path.isdir(folder_path):
        print(f"Error: The specified folder path {folder_path} does not exist.")
        sys.exit(1)
    
    # Create results directory
    results_dir = os.path.join(folder_path, "TMalign_Results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all immediate subdirectories in the root folder
    pdbid_folders = [f for f in glob.glob(os.path.join(folder_path, "*")) 
                     if os.path.isdir(f) and os.path.basename(f) != "TMalign_Results"]
    
    if not pdbid_folders:
        print(f"Warning: No PDBID folders found in {folder_path}")
        sys.exit(0)
    
    print(f"Found {len(pdbid_folders)} PDBID folders. Starting processing...")
    
    # Process each PDBID folder
    for folder in pdbid_folders:
        process_pdbid_folder(folder, results_dir)
    
    print(f"Processing complete. Results saved to {results_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: Please provide the folder path as a command-line argument.")
        print("Usage: python compute_tm_score.py PATH_TO_FOLDER")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    main(folder_path)