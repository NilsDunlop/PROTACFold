import os
import sys
import json
import argparse
import subprocess
import pandas as pd
import numpy as np
import re
import pymol
import requests
from datetime import datetime
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.api import extract_ligand_ccd_from_pdb, extract_comp_ids, fetch_ligand_data

# Initialize PyMol in headless mode quiet mode
pymol.finish_launching(['pymol', '-qc'])

def get_pdb_release_date(pdb_id):
    """
    Get the initial release date for a PDB entry.
    
    Args:
        pdb_id: The PDB ID (e.g., "6HAY")
        
    Returns:
        string: The release date in YYYY-MM-DD format or error message
    """
    if not pdb_id:
        return "No ID provided"
    
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    
    try:
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching release date for {pdb_id}: {response.status_code}")
            return None
        
        data = response.json()
        
        # Navigate to the revision history
        revisions = data.get('pdbx_audit_revision_history')
        if not isinstance(revisions, list) or not revisions:
            print(f"No revision history found for {pdb_id}")
            return None
        
        # Find the initial release
        initial_release = next((r for r in revisions if r.get('ordinal') == 1), None)
        if not initial_release:
            initial_release = next((r for r in revisions if r.get('ordinal') == '1'), None)
            
        if not initial_release:
            print(f"Initial release not found for {pdb_id}")
            return None
        
        # Return formatted date
        release_date = initial_release.get('revision_date', '').split('T')[0]
        
        # Validate date format
        try:
            datetime.strptime(release_date, '%Y-%m-%d')
            return release_date
        except ValueError:
            print(f"Invalid date format: {release_date}")
            return None
        
    except Exception as e:
        print(f"Error fetching release date for {pdb_id}: {e}")
        return None

def calculate_molecular_properties_from_smiles(smiles):
    """
    Calculate molecular properties for a compound from a SMILES string using RDKit.
    
    Args:
        smiles: SMILES string of the compound
    
    Returns:
        Dictionary containing calculated molecular properties
    """
    if not smiles:
        return {}
    
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        print(f"WARNING: Could not parse SMILES: {smiles}")
        return {}
    
    # Calculate molecular properties
    properties = {
        'Molecular_Weight': Descriptors.MolWt(mol),
        'Heavy_Atom_Count': mol.GetNumHeavyAtoms(),
        'Ring_Count': rdMolDescriptors.CalcNumRings(mol),
        'Rotatable_Bond_Count': Descriptors.NumRotatableBonds(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBA_Count': Lipinski.NumHAcceptors(mol),
        'HBD_Count': Lipinski.NumHDonors(mol),
        'TPSA': Descriptors.TPSA(mol)
    }
    
    # Count aromatic and aliphatic rings
    aromatic_ring_count = 0
    aliphatic_ring_count = 0
    
    # Use the correct method to get rings
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        # Check if all atoms in the ring are aromatic
        is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
        if is_aromatic:
            aromatic_ring_count += 1
        else:
            aliphatic_ring_count += 1
    
    properties['Aromatic_Rings'] = aromatic_ring_count
    properties['Aliphatic_Rings'] = aliphatic_ring_count
    
    return properties

def fetch_smile_strings(pdb_id):
    """Fetch canonical SMILES strings for the primary drug-like ligand in a PDB structure."""
    try:
        ligand_ccd = extract_ligand_ccd_from_pdb(pdb_id)
        comp_ids = extract_comp_ids(ligand_ccd)
        
        if not comp_ids:
            return {}
        
        # Common non-drug components to filter out
        common_components = {
            'HOH', 'WAT', 'H2O',  # Water
            'GOL', 'EDO', 'PEG',  # Glycerol, ethylene glycol, polyethylene glycol
            'SO4', 'PO4', 'CIT',  # Common ions/buffers
            'CL', 'NA', 'MG', 'CA', 'ZN', 'FE', 'FE2',  # Common ions
            'EPE', 'MES', 'TRIS', 'HEPES',  # Common buffers
            'ACT', 'IMD', 'BME', 'IPA', 'HED', 'MPD'  # Other common additives
        }
        
        # Dictionary to store details about each component
        component_details = {}
        
        # Fetch details for each component
        for comp_id in comp_ids:
            try:
                ligand_data = fetch_ligand_data(pdb_id, comp_id)
                
                # Extract chemical data
                chem_data = ligand_data.get("chemical_data", {})
                data = chem_data.get("data", {})
                chem_comp = data.get("chem_comp", {})
                
                if chem_comp:
                    # Get basic info
                    comp_info = chem_comp.get("chem_comp", {})
                    name = comp_info.get("name", "")
                    formula = comp_info.get("formula", "")
                    comp_type = comp_info.get("type", "")
                    formula_weight = comp_info.get("formula_weight", 0)
                    
                    # Get SMILES
                    descriptors = chem_comp.get("rcsb_chem_comp_descriptor", {})
                    smiles_stereo = descriptors.get("SMILES_stereo", "")
                    
                    # Store the data
                    component_details[comp_id] = {
                        'name': name,
                        'formula': formula,
                        'type': comp_type,
                        'weight': formula_weight,
                        'SMILES_stereo': smiles_stereo,
                        'is_common': comp_id in common_components,
                        'smiles_complexity': len(smiles_stereo) if smiles_stereo else 0
                    }
            except Exception:
                pass
        
        # Filter out common components
        drug_like_components = {k: v for k, v in component_details.items() 
                               if not v['is_common'] and v['smiles_complexity'] > 5}
        
        if not drug_like_components:
            sorted_components = sorted(component_details.items(), 
                                     key=lambda x: x[1]['smiles_complexity'], 
                                     reverse=True)
            if sorted_components:
                primary_comp_id = sorted_components[0][0]
                return {primary_comp_id: {
                    'SMILES_stereo': component_details[primary_comp_id]['SMILES_stereo']
                }}
            return {}
        
        # Sort by stereo SMILES complexity/molecular weight to find primary ligand
        sorted_drug_components = sorted(drug_like_components.items(), 
                                      key=lambda x: (x[1]['smiles_complexity'], x[1]['weight']), 
                                      reverse=True)
        
        # Select primary component
        primary_comp_id = sorted_drug_components[0][0]
        
        return {primary_comp_id: {
            'SMILES_stereo': component_details[primary_comp_id]['SMILES_stereo']
        }}
        
    except Exception as e:
        print(f"Error in fetch_smile_strings for {pdb_id}: {e}")
        return {}

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
    
    # Start with PDB ID
    result_row = {"PDB_ID": pdb_id}
    
    # Get and add release date
    release_date = get_pdb_release_date(pdb_id)
    if release_date:
        result_row["RELEASE_DATE"] = release_date
    
    # Initialize variables to store SMILES and ligand ID
    smiles_stereo = None
    ligand_id = None
    
    # Process SMILES model
    smiles_folder = os.path.join(folder_path, f"{pdb_id.lower()}_ternary_smiles")
    smiles_model_path = os.path.join(smiles_folder, f"{pdb_id.lower()}_ternary_smiles_model.cif")
    smiles_json_path = os.path.join(smiles_folder, f"{pdb_id.lower()}_ternary_smiles_summary_confidences.json")
    
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
    
    # Process CCD model
    ccd_folder = os.path.join(folder_path, f"{pdb_id.lower()}_ternary_ccd")
    ccd_model_path = os.path.join(ccd_folder, f"{pdb_id.lower()}_ternary_ccd_model.cif")
    ccd_json_path = os.path.join(ccd_folder, f"{pdb_id.lower()}_ternary_ccd_summary_confidences.json")
    
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
    
    # Find and store primary ligand details
    try:
        smile_strings = fetch_smile_strings(pdb_id)
        if smile_strings:
            ligand_id = list(smile_strings.keys())[0]
            result_row["LIGAND_ID"] = ligand_id
            
            # Create ligand link URL
            result_row["LIGAND_LINK"] = f"https://www.rcsb.org/ligand/{ligand_id}"
            
            # Save SMILES
            smiles_stereo = smile_strings[ligand_id].get('SMILES_stereo')
            result_row["CANONICAL_SMILES"] = smiles_stereo
        else:
            print(f"No suitable ligands found for {pdb_id}")
    except Exception as e:
        print(f"Error fetching SMILE strings for {pdb_id}: {e}")
    
    # Calculate and add molecular properties
    if smiles_stereo:
        mol_properties = calculate_molecular_properties_from_smiles(smiles_stereo)
        for prop, value in mol_properties.items():
            result_row[prop] = value
    
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