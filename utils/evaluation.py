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
from PIL import Image
import tempfile
import shutil
from datetime import datetime
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.website.api import extract_ligand_ccd_from_pdb, extract_comp_ids, fetch_ligand_data

# Initialize PyMol in headless mode quiet mode
pymol.finish_launching(['pymol', '-cq'])

def get_pdb_release_date(pdb_id):
    """Get the initial release date for a PDB entry."""
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

def capture_side_by_side_views(pdb_id, smiles_model_path, ccd_model_path, ref_path, output_folder, ligand_id=None):
    """Capture separate screenshots of ref+CCD and ref+SMILES models and combines them side by side."""
    if not os.path.exists(smiles_model_path) or not os.path.exists(ccd_model_path) or not os.path.exists(ref_path):
        print(f"One or more required structure files not found for {pdb_id}")
        return None
    
    # Create directories
    os.makedirs(output_folder, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    final_paths = []
    
    # Image settings
    width, height, dpi = 800, 800, 150
    angles = [0, 90, 180]
    
    # Setup and render a single view
    def render_model_view(model_path, model_color, model_type, angle):
        try:
            cmd.delete("all")
            
            # Load structures
            model_name = f"{pdb_id}_{model_type}_model"
            ref_name = pdb_id
            cmd.load(model_path, model_name)
            cmd.load(ref_path, ref_name)
            
            # Setup view
            cmd.set("cartoon_transparency", 0.7)
            cmd.set("ray_opaque_background", 1)
            cmd.bg_color("white")
            cmd.set("ray_shadows", 0)
            cmd.remove("solvent or resn HOH or resn WAT")
            
            # Align and color
            cmd.align(f"polymer and name CA and ({model_name})", 
                     f"polymer and name CA and ({ref_name})", 
                     quiet=1, reset=1)
            cmd.color(model_color, model_name)
            cmd.color("firebrick", ref_name)
            cmd.show("cartoon", "all")
            
            # Find and highlight ligands
            ref_chains = cmd.get_chains(ref_name)
            model_chains = cmd.get_chains(model_name)
            
            # Try ligand by ID first
            ligand_found = False
            if ligand_id:
                cmd.select("ligand_ref", f"resn {ligand_id} and {ref_name}")
                if cmd.count_atoms("ligand_ref") > 0:
                    cmd.show("sticks", "ligand_ref")
                    cmd.color("gold", "ligand_ref")
                    
                    # Try model ligand by unique chain
                    unique_chains = [c for c in model_chains if c not in ref_chains]
                    if unique_chains:
                        cmd.select("ligand_model", f"chain {unique_chains[-1]} and {model_name}")
                        if cmd.count_atoms("ligand_model") > 0:
                            cmd.show("sticks", "ligand_model")
                            cmd.color(model_color, "ligand_model")
                    
                    cmd.zoom("ligand_ref", 5)
                    ligand_found = True
            
            # Fallback to hetero atoms if needed
            if not ligand_found:
                cmd.select("het_ref", f"hetatm and not solvent and {ref_name}")
                if cmd.count_atoms("het_ref") > 0:
                    cmd.show("sticks", "het_ref")
                    cmd.color("gold", "het_ref")
                    
                    cmd.select("het_model", f"hetatm and not solvent and {model_name}")
                    if cmd.count_atoms("het_model") > 0:
                        cmd.show("sticks", "het_model")
                        cmd.color(model_color, "het_model")
                    
                    cmd.zoom("het_ref", 5)
                else:
                    cmd.zoom("all")
            
            # Rotate and render
            cmd.rotate("y", angle)
            
            output_path = os.path.join(temp_dir, f"{model_type}_{angle}.png")
            cmd.ray(width, height)
            cmd.png(output_path, dpi=dpi)
            return output_path
        
        except Exception as e:
            print(f"Error capturing {model_type} view at {angle}°: {e}")
            return None
    
    try:
        # Generate views for each model and angle
        model_views = {}
        for angle in angles:
            # Render CCD view
            ccd_path = render_model_view(ccd_model_path, "blue", "ccd", angle)
            if ccd_path:
                model_views.setdefault(angle, {})["ccd"] = ccd_path
            
            # Render SMILES view
            smiles_path = render_model_view(smiles_model_path, "cyan", "smiles", angle)
            if smiles_path:
                model_views.setdefault(angle, {})["smiles"] = smiles_path
        
        # Combine the images side by side
        for angle in angles:
            if angle in model_views and "ccd" in model_views[angle] and "smiles" in model_views[angle]:
                try:
                    ccd_img = Image.open(model_views[angle]["ccd"])
                    smiles_img = Image.open(model_views[angle]["smiles"])
                    
                    # Create and save the combined image
                    combined = Image.new('RGB', (width * 2, height))
                    combined.paste(ccd_img, (0, 0))
                    combined.paste(smiles_img, (width, 0))
                    
                    combined_path = os.path.join(output_folder, f"{pdb_id}_side_by_side_{angle}deg.png")
                    combined.save(combined_path)
                    final_paths.append(combined_path)
                except Exception as e:
                    print(f"Error combining images for {angle}° view: {e}")
        
        return final_paths if final_paths else None
    
    except Exception as e:
        print(f"Error in side-by-side view generation: {e}")
        return None
    
    finally:
        cmd.delete("all")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {e}")

def extract_component_info(analysis_file):
    """Extract POI and E3 information from analysis file."""
    poi_name = None
    poi_sequence = None
    e3_name = None
    e3_sequence = None
    
    try:
        if not os.path.exists(analysis_file):
            print(f"Analysis file not found: {analysis_file}")
            return poi_name, poi_sequence, e3_name, e3_sequence
            
        with open(analysis_file, 'r') as f:
            content = f.read()
            
        # Extract POI information
        poi_match = re.search(r'Protein of Interest: (.+?)\n(.+?)(?=\n\n|$)', content, re.DOTALL)
        if poi_match:
            poi_name = poi_match.group(1).split('|')[1] if '|' in poi_match.group(1) else poi_match.group(1)
            poi_sequence = ''.join(poi_match.group(2).strip().split())
        
        # Extract E3 information
        e3_match = re.search(r'E3 Ubiquitin Ligase: (.+?)\n(.+?)(?=\n\n|$)', content, re.DOTALL)
        if e3_match:
            e3_name = e3_match.group(1).split('|')[1] if '|' in e3_match.group(1) else e3_match.group(1)
            e3_sequence = ''.join(e3_match.group(2).strip().split())
            
    except Exception as e:
        print(f"Error extracting component info: {e}")
    
    return poi_name, poi_sequence, e3_name, e3_sequence

def sequence_similarity(seq1, seq2):
    """Calculate sequence similarity between two sequences."""
    if not seq1 or not seq2:
        return 0.0
    
    # Find the longest common subsequence
    shorter = seq1 if len(seq1) <= len(seq2) else seq2
    longer = seq2 if len(seq1) <= len(seq2) else seq1
    
    best_similarity = 0.0
    for i in range(len(longer) - len(shorter) + 1):
        matches = sum(a == b for a, b in zip(shorter, longer[i:i+len(shorter)]))
        similarity = matches / len(shorter)
        best_similarity = max(best_similarity, similarity)
    
    return best_similarity

def find_chain_by_sequence(sequence, model_name, min_similarity=0.8):
    """Find a chain that best matches the given sequence."""
    if not sequence:
        return None
        
    # Get all chains in the model
    try:
        chains = cmd.get_chains(model_name)
    except:
        print(f"Error getting chains for {model_name}")
        return None
        
    if not chains:
        print(f"No chains found in {model_name}")
        return None
    
    # Store chain sequences for analysis
    best_match = {"chain": None, "similarity": 0.0, "sequence": None}
    
    for chain in chains:
        try:
            # Get the sequence of this chain
            fasta = cmd.get_fastastr(f"{model_name} and chain {chain}")
            if not fasta:
                continue
                
            # Clean up the FASTA sequence
            chain_seq = ''.join(fasta.split())
            
            # Check for exact match or containment
            if sequence in chain_seq or chain_seq in sequence:
                return chain
            
            # Calculate similarity
            similarity = sequence_similarity(sequence, chain_seq)
            
            if similarity > best_match["similarity"]:
                best_match = {
                    "chain": chain, 
                    "similarity": similarity,
                    "sequence": chain_seq
                }
        except Exception as e:
            print(f"Error checking chain {chain}: {e}")
    
    # If we have a good match, use it
    if best_match["similarity"] >= min_similarity:
        return best_match["chain"]
    
    # If no good match and we have chains, use the first one as fallback
    if chains:
        return chains[0]
        
    return None

def calculate_component_rmsd(model_path, ref_path, pdb_id, model_type, poi_sequence, e3_sequence):
    """Calculate RMSD for POI and E3 components with enhanced sequence matching."""
    results = {
        "POI_RMSD": "N/A",
        "E3_RMSD": "N/A"
    }
    
    if not os.path.exists(model_path) or not os.path.exists(ref_path):
        return results
    
    # Load structures
    model_name = f"{pdb_id}_{model_type}_model"
    ref_name = f"{pdb_id}_ref"
    
    try:
        cmd.delete("all")
        cmd.load(model_path, model_name)
        cmd.load(ref_path, ref_name)
        
        # Process POI
        if poi_sequence:
            poi_chain_model = find_chain_by_sequence(poi_sequence, model_name)
            poi_chain_ref = find_chain_by_sequence(poi_sequence, ref_name)
            
            if poi_chain_model and poi_chain_ref:
                # Create selections
                cmd.select("poi_model", f"{model_name} and chain {poi_chain_model}")
                cmd.select("poi_ref", f"{ref_name} and chain {poi_chain_ref}")
                
                # Calculate RMSD
                try:
                    poi_rmsd = cmd.align("poi_model", "poi_ref")[0]
                    results["POI_RMSD"] = poi_rmsd
                except Exception as e:
                    print(f"Error calculating POI RMSD for {pdb_id} ({model_type}): {e}")
        
        # Process E3
        if e3_sequence:
            e3_chain_model = find_chain_by_sequence(e3_sequence, model_name)
            e3_chain_ref = find_chain_by_sequence(e3_sequence, ref_name)
            
            if e3_chain_model and e3_chain_ref:
                # Create selections
                cmd.select("e3_model", f"{model_name} and chain {e3_chain_model}")
                cmd.select("e3_ref", f"{ref_name} and chain {e3_chain_ref}")
                
                # Calculate RMSD
                try:
                    e3_rmsd = cmd.align("e3_model", "e3_ref")[0]
                    results["E3_RMSD"] = e3_rmsd
                except Exception as e:
                    print(f"Error calculating E3 RMSD for {pdb_id} ({model_type}): {e}")
    
    except Exception as e:
        print(f"Error in component RMSD calculation for {pdb_id} ({model_type}): {e}")
    
    finally:
        # Clean up
        cmd.delete("all")
    
    return results

def calculate_molecular_properties_from_smiles(smiles):
    """Calculate molecular properties for a compound from a SMILES string using RDKit."""
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
    analysis_file = os.path.join(folder_path, f"{pdb_id}_analysis.txt")
    
    # Create images folder for screenshots
    images_folder = os.path.join(folder_path, "images")
    os.makedirs(images_folder, exist_ok=True)
    
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
    else:
        result_row["RELEASE_DATE"] = "N/A"
    
    # Extract POI and E3 information if available
    poi_name, poi_sequence, e3_name, e3_sequence = None, None, None, None
    if os.path.exists(analysis_file):
        poi_name, poi_sequence, e3_name, e3_sequence = extract_component_info(analysis_file)
    
    # Set default values
    result_row["POI_NAME"] = poi_name if poi_name else "N/A"
    result_row["POI_SEQUENCE"] = poi_sequence if poi_sequence else "N/A"
    result_row["E3_NAME"] = e3_name if e3_name else "N/A"
    result_row["E3_SEQUENCE"] = e3_sequence if e3_sequence else "N/A"
    
    # Initialize default values for metrics
    for metric in ["SMILES RMSD", "SMILES_POI_RMSD", "SMILES_E3_RMSD", 
                  "SMILES DOCKQ SCORE", "SMILES DOCKQ iRMSD", "SMILES DOCKQ LRMSD",
                  "CCD RMSD", "CCD_POI_RMSD", "CCD_E3_RMSD", 
                  "CCD DOCKQ SCORE", "CCD DOCKQ iRMSD", "CCD DOCKQ LRMSD",
                  "SMILES FRACTION DISORDERED", "SMILES HAS_CLASH", "SMILES IPTM", 
                  "SMILES PTM", "SMILES RANKING_SCORE", "CCD FRACTION DISORDERED", 
                  "CCD HAS_CLASH", "CCD IPTM", "CCD PTM", "CCD RANKING_SCORE"]:
        result_row[metric] = "N/A"
    
    # Find and store primary ligand details
    ligand_id = None
    try:
        smile_strings = fetch_smile_strings(pdb_id)
        if smile_strings:
            ligand_id = list(smile_strings.keys())[0]
            result_row["LIGAND_CCD"] = ligand_id
            
            # Create ligand link URL
            result_row["LIGAND_LINK"] = f"https://www.rcsb.org/ligand/{ligand_id}"
            
            # Save SMILES
            smiles_stereo = smile_strings[ligand_id].get('SMILES_stereo')
            result_row["LIGAND_SMILES"] = smiles_stereo if smiles_stereo else "N/A"
        else:
            print(f"No suitable ligands found for {pdb_id}")
            result_row["LIGAND_CCD"] = "N/A"
            result_row["LIGAND_LINK"] = "N/A"
            result_row["LIGAND_SMILES"] = "N/A"
    except Exception as e:
        print(f"Error fetching SMILE strings for {pdb_id}: {e}")
        result_row["LIGAND_CCD"] = "N/A"
        result_row["LIGAND_LINK"] = "N/A"
        result_row["LIGAND_SMILES"] = "N/A"
    
    # Initialize molecular property fields
    for prop in ['Molecular_Weight', 'Heavy_Atom_Count', 'Ring_Count', 'Rotatable_Bond_Count',
                'LogP', 'HBA_Count', 'HBD_Count', 'TPSA', 'Aromatic_Rings', 'Aliphatic_Rings']:
        result_row[prop] = "N/A"
    
    # Calculate and add molecular properties
    if "LIGAND_SMILES" in result_row and result_row["LIGAND_SMILES"] != "N/A":
        mol_properties = calculate_molecular_properties_from_smiles(result_row["LIGAND_SMILES"])
        for prop, value in mol_properties.items():
            result_row[prop] = value
    
    # Process SMILES model
    smiles_folder_patterns = [
        os.path.join(folder_path, f"{pdb_id.lower()}_ternary_smiles"),
        os.path.join(folder_path, f"{pdb_id.lower()}_smiles")
    ]
    
    smiles_model_path = None
    smiles_json_path = None
    
    # Find the first existing SMILES folder and corresponding files
    for folder_pattern in smiles_folder_patterns:
        if os.path.exists(folder_pattern):
            model_name = f"{os.path.basename(folder_pattern)}_model.cif"
            json_name = f"{os.path.basename(folder_pattern)}_summary_confidences.json"
            temp_model_path = os.path.join(folder_pattern, model_name)
            temp_json_path = os.path.join(folder_pattern, json_name)
            
            if os.path.exists(temp_model_path):
                smiles_model_path = temp_model_path
                smiles_json_path = temp_json_path if os.path.exists(temp_json_path) else None
                break
    
    if smiles_model_path:
        try:
            # Compute overall RMSD
            smiles_rmsd = compute_rmsd_with_pymol(smiles_model_path, ref_path, pdb_id, "smiles")
            result_row["SMILES RMSD"] = smiles_rmsd
            
            # Compute component-specific RMSD if sequences are available
            if poi_sequence or e3_sequence:
                component_rmsd = calculate_component_rmsd(
                    smiles_model_path, ref_path, pdb_id, "smiles", 
                    poi_sequence, e3_sequence
                )
                result_row["SMILES_POI_RMSD"] = component_rmsd["POI_RMSD"]
                result_row["SMILES_E3_RMSD"] = component_rmsd["E3_RMSD"]
            
            # Run DockQ
            smiles_dockq_output = run_dockq(smiles_model_path, ref_path)
            if smiles_dockq_output:
                dockq_score, irmsd, lrmsd = extract_dockq_values(smiles_dockq_output)
                result_row["SMILES DOCKQ SCORE"] = dockq_score if dockq_score is not None else "N/A"
                result_row["SMILES DOCKQ iRMSD"] = irmsd if irmsd is not None else "N/A"
                result_row["SMILES DOCKQ LRMSD"] = lrmsd if lrmsd is not None else "N/A"
                
        except Exception as e:
            print(f"Error processing SMILES model for {pdb_id}: {e}")
        
        # Extract confidence values
        if smiles_json_path and os.path.exists(smiles_json_path):
            try:
                fraction_disordered, has_clash, iptm, ptm, ranking_score = extract_confidence_values(smiles_json_path)
                result_row["SMILES FRACTION DISORDERED"] = fraction_disordered if fraction_disordered is not None else "N/A"
                result_row["SMILES HAS_CLASH"] = has_clash if has_clash is not None else "N/A"
                result_row["SMILES IPTM"] = iptm if iptm is not None else "N/A"
                result_row["SMILES PTM"] = ptm if ptm is not None else "N/A"
                result_row["SMILES RANKING_SCORE"] = ranking_score if ranking_score is not None else "N/A"
            except Exception as e:
                print(f"Error extracting SMILES confidence values for {pdb_id}: {e}")
    
    # Process CCD model
    ccd_folder_patterns = [
        os.path.join(folder_path, f"{pdb_id.lower()}_ternary_ccd"),
        os.path.join(folder_path, f"{pdb_id.lower()}_ccd")
    ]
    
    ccd_model_path = None
    ccd_json_path = None
    
    # Find the first existing CCD folder and corresponding files
    for folder_pattern in ccd_folder_patterns:
        if os.path.exists(folder_pattern):
            model_name = f"{os.path.basename(folder_pattern)}_model.cif"
            json_name = f"{os.path.basename(folder_pattern)}_summary_confidences.json"
            temp_model_path = os.path.join(folder_pattern, model_name)
            temp_json_path = os.path.join(folder_pattern, json_name)
            
            if os.path.exists(temp_model_path):
                ccd_model_path = temp_model_path
                ccd_json_path = temp_json_path if os.path.exists(temp_json_path) else None
                break
    
    if ccd_model_path:
        try:
            # Compute overall RMSD
            ccd_rmsd = compute_rmsd_with_pymol(ccd_model_path, ref_path, pdb_id, "ccd")
            result_row["CCD RMSD"] = ccd_rmsd
            
            # Compute component-specific RMSD if sequences are available
            if poi_sequence or e3_sequence:
                component_rmsd = calculate_component_rmsd(
                    ccd_model_path, ref_path, pdb_id, "ccd", 
                    poi_sequence, e3_sequence
                )
                result_row["CCD_POI_RMSD"] = component_rmsd["POI_RMSD"]
                result_row["CCD_E3_RMSD"] = component_rmsd["E3_RMSD"]
            
            # Run DockQ
            ccd_dockq_output = run_dockq(ccd_model_path, ref_path)
            if ccd_dockq_output:
                dockq_score, irmsd, lrmsd = extract_dockq_values(ccd_dockq_output)
                result_row["CCD DOCKQ SCORE"] = dockq_score if dockq_score is not None else "N/A"
                result_row["CCD DOCKQ iRMSD"] = irmsd if irmsd is not None else "N/A"
                result_row["CCD DOCKQ LRMSD"] = lrmsd if lrmsd is not None else "N/A"
                
        except Exception as e:
            print(f"Error processing CCD model for {pdb_id}: {e}")
        
        # Extract confidence values
        if ccd_json_path and os.path.exists(ccd_json_path):
            try:
                fraction_disordered, has_clash, iptm, ptm, ranking_score = extract_confidence_values(ccd_json_path)
                result_row["CCD FRACTION DISORDERED"] = fraction_disordered if fraction_disordered is not None else "N/A"
                result_row["CCD HAS_CLASH"] = has_clash if has_clash is not None else "N/A"
                result_row["CCD IPTM"] = iptm if iptm is not None else "N/A"
                result_row["CCD PTM"] = ptm if ptm is not None else "N/A"
                result_row["CCD RANKING_SCORE"] = ranking_score if ranking_score is not None else "N/A"
            except Exception as e:
                print(f"Error extracting CCD confidence values for {pdb_id}: {e}")
    
    # Create side-by-side ligand visualizations
    if smiles_model_path and ccd_model_path:
        try:
            side_by_side_screenshots = capture_side_by_side_views(
                pdb_id,
                smiles_model_path,
                ccd_model_path,
                ref_path,
                images_folder,
                ligand_id
            )
            if not side_by_side_screenshots:
                print(f"Failed to create side-by-side ligand views for {pdb_id}")
        except Exception as e:
            print(f"Error creating side-by-side ligand views for {pdb_id}: {e}")
    
    results.append(result_row)

def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaFold predictions against experimental structures.")
    parser.add_argument("folder", help="Path to the folder containing PDB ID folders")
    parser.add_argument("--output", "-o", default="evaluation_results.csv", help="Output CSV file name or full path (default: saves to ../data/af3_results/)")
    parser.add_argument("--type", "-t", default="PROTAC", help="Type of molecules to evaluate (e.g. 'PROTAC', 'Molecular Glue', )")
    args = parser.parse_args()
    
    # Check if the input folder exists
    if not os.path.exists(args.folder):
        print(f"Error: Folder {args.folder} does not exist.")
        sys.exit(1)
    
    # Determine output path
    if os.path.isabs(args.output) or '/' in args.output or '\\' in args.output:
        output_path = args.output
        output_dir = os.path.dirname(output_path)
    else:
        default_dir = "../data/af3_results/"
        output_dir = default_dir
        output_path = os.path.join(default_dir, args.output)
    
    # Make sure the output directory exists
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
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
        
        # Add TYPE column
        df['TYPE'] = args.type
        
        # Define the column order
        column_order = [
            'PDB_ID', 'TYPE', 'RELEASE_DATE', 'POI_NAME', 'POI_SEQUENCE', 'E3_NAME', 'E3_SEQUENCE',
            'SMILES RMSD', 'SMILES_POI_RMSD', 'SMILES_E3_RMSD', 'SMILES DOCKQ SCORE', 'SMILES DOCKQ iRMSD', 'SMILES DOCKQ LRMSD',
            'CCD RMSD', 'CCD_POI_RMSD', 'CCD_E3_RMSD', 'CCD DOCKQ SCORE', 'CCD DOCKQ iRMSD', 'CCD DOCKQ LRMSD',
            'LIGAND_CCD', 'LIGAND_LINK', 'LIGAND_SMILES',
            'SMILES FRACTION DISORDERED', 'SMILES HAS_CLASH', 'SMILES IPTM', 'SMILES PTM', 'SMILES RANKING_SCORE',
            'CCD FRACTION DISORDERED', 'CCD HAS_CLASH', 'CCD IPTM', 'CCD PTM', 'CCD RANKING_SCORE',
            'Molecular_Weight', 'Heavy_Atom_Count', 'Ring_Count', 'Rotatable_Bond_Count',
            'LogP', 'HBA_Count', 'HBD_Count', 'TPSA', 'Aromatic_Rings', 'Aliphatic_Rings'
        ]
        
        # Ensure all columns exist
        for col in column_order:
            if col not in df.columns:
                df[col] = "N/A"
        
        # Reorder columns
        df = df[column_order]
        
        # Sort by release date
        df['TEMP_DATE'] = pd.to_datetime(df['RELEASE_DATE'], errors='coerce')
        df = df.sort_values(by='TEMP_DATE').drop(columns=['TEMP_DATE'])
        
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()