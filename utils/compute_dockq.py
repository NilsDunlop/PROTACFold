import os
import subprocess
from pathlib import Path

def run_dockq_analysis():
    af_models_dir = r"C:\Users\nilsd\Desktop\Python\Alphafold\RelatedPaper\all_models_v3\AF3\with_context"
    reference_dir = r"C:\Users\nilsd\Desktop\Python\Alphafold\RelatedPaper\all_models\PDB"
    output_dir = r"C:\Users\nilsd\Desktop\Python\Alphafold\RelatedPaper\dockq_results"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all AF model files
    af_files = [f for f in os.listdir(af_models_dir) if f.endswith('.pdb')]

    for af_file in af_files:
        # Extract PDB ID from filename
        if af_file.startswith('fold_'):
            pdb_id = af_file.split('_')[1]
        else:
            pdb_id = af_file.split('_')[0]
        
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
            continue
        else:
            print(f"✓ Found reference PDB file")

        # Construct and run DockQ command
        cmd = f"DockQ {af_model_path} {ref_pdb_path} --json {output_json}"
        try:
            print(f"Processing {pdb_id}...")
            subprocess.run(cmd, shell=True, check=True)
            print(f"Completed DockQ analysis for {pdb_id}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {pdb_id}: {e}")

if __name__ == "__main__":
    run_dockq_analysis()
