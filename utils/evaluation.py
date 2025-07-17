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
import logging
from datetime import datetime
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.website.api import extract_ligand_ccd_from_pdb, extract_comp_ids, fetch_ligand_data
from model_adapters import get_model_adapter, Boltz1Adapter

# Initialize PyMol in headless mode quiet mode
pymol.finish_launching(['pymol', '-cq'])

def setup_logging(log_file=None, log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file. If None, only console logging is enabled.
        log_level: Logging level (default: logging.INFO)
    """
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Set up file handler if log file is specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logging to {log_file}")
    
    return logger

def get_pdb_release_date(pdb_id):
    """Get the initial release date for a PDB entry."""
    if not pdb_id:
        return "No ID provided"
    
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    
    try:
        response = requests.get(url)
        
        if response.status_code != 200:
            logging.warning(f"Error fetching release date for {pdb_id}: {response.status_code}")
            return None
        
        data = response.json()
        
        # Navigate to the revision history
        revisions = data.get('pdbx_audit_revision_history')
        if not isinstance(revisions, list) or not revisions:
            logging.warning(f"No revision history found for {pdb_id}")
            return None
        
        # Find the initial release
        initial_release = next((r for r in revisions if r.get('ordinal') == 1), None)
        if not initial_release:
            initial_release = next((r for r in revisions if r.get('ordinal') == '1'), None)
            
        if not initial_release:
            logging.warning(f"Initial release not found for {pdb_id}")
            return None
        
        # Return formatted date
        release_date = initial_release.get('revision_date', '').split('T')[0]
        
        # Validate date format
        try:
            datetime.strptime(release_date, '%Y-%m-%d')
            return release_date
        except ValueError:
            logging.warning(f"Invalid date format: {release_date}")
            return None
        
    except Exception as e:
        logging.error(f"Error fetching release date for {pdb_id}: {e}")
        return None

def capture_side_by_side_views(pdb_id, smiles_model_path, ccd_model_path, ref_path, output_folder, ligand_id=None):
    """Capture separate screenshots of ref+CCD and ref+SMILES models and combines them side by side."""
    if not os.path.exists(smiles_model_path) or not os.path.exists(ccd_model_path) or not os.path.exists(ref_path):
        logging.warning(f"One or more required structure files not found for {pdb_id}")
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
            logging.error(f"Error capturing {model_type} view at {angle}°: {e}")
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
                    logging.error(f"Error combining images for {angle}° view: {e}")
        
        return final_paths if final_paths else None
    
    except Exception as e:
        logging.error(f"Error in side-by-side view generation: {e}")
        return None
    
    finally:
        cmd.delete("all")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logging.warning(f"Failed to clean up temporary directory: {e}")

def extract_component_info(analysis_file):
    """Extract POI and E3 information from analysis file."""
    poi_name = None
    poi_sequence = None
    e3_name = None
    e3_sequence = None
    
    try:
        if not os.path.exists(analysis_file):
            logging.warning(f"Analysis file not found: {analysis_file}")
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
        logging.error(f"Error extracting component info: {e}")
    
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
        logging.error(f"Error getting chains for {model_name}")
        return None
        
    if not chains:
        logging.warning(f"No chains found in {model_name}")
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
            logging.error(f"Error checking chain {chain}: {e}")
    
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
                    logging.error(f"Error calculating POI RMSD for {pdb_id} ({model_type}): {e}")
        
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
                    logging.error(f"Error calculating E3 RMSD for {pdb_id} ({model_type}): {e}")
    
    except Exception as e:
        logging.error(f"Error in component RMSD calculation for {pdb_id} ({model_type}): {e}")
    
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
        logging.warning(f"Could not parse SMILES: {smiles}")
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
            logging.info(f"No component IDs found for {pdb_id}")
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
            except Exception as e:
                logging.debug(f"Error fetching data for component {comp_id} in {pdb_id}: {e}")
                pass
        
        # Filter out common components
        drug_like_components = {k: v for k, v in component_details.items() 
                               if not v['is_common'] and v['smiles_complexity'] > 5}
        
        if not drug_like_components:
            logging.info(f"No drug-like components found for {pdb_id}, using most complex component")
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
        logging.info(f"Selected {primary_comp_id} as primary ligand for {pdb_id}")
        
        return {primary_comp_id: {
            'SMILES_stereo': component_details[primary_comp_id]['SMILES_stereo']
        }}
        
    except Exception as e:
        logging.error(f"Error in fetch_smile_strings for {pdb_id}: {e}")
        return {}

def get_primary_ligand_ccd(pdb_id):
    """Get the primary ligand CCD identifier for a PDB structure (enhanced version)."""
    try:
        ligand_ccd = extract_ligand_ccd_from_pdb(pdb_id)
        comp_ids = extract_comp_ids(ligand_ccd)
        
        if not comp_ids:
            logging.debug(f"No component IDs found for {pdb_id}")
            return "N/A"
        
        # Common non-drug components to filter out (expanded list)
        common_components = {
            'HOH', 'WAT', 'H2O',  # Water
            'GOL', 'EDO', 'PEG',  # Glycerol, ethylene glycol, polyethylene glycol
            'SO4', 'PO4', 'CIT',  # Common ions/buffers
            'CL', 'NA', 'MG', 'CA', 'ZN', 'FE', 'FE2',  # Common ions
            'EPE', 'MES', 'TRIS', 'HEPES',  # Common buffers
            'ACT', 'IMD', 'BME', 'IPA', 'HED', 'MPD',  # Other common additives
            'SCN', 'CN', 'BR', 'I', 'F',  # Small ions that can slip through
            'DMS', 'ACE', 'MTL'  # Small organic molecules
        }
        
        # Get detailed information about each component to make smarter selection
        component_details = {}
        for comp_id in comp_ids:
            if comp_id not in common_components:
                try:
                    ligand_data = fetch_ligand_data(pdb_id, comp_id)
                    
                    # Extract chemical data
                    chem_data = ligand_data.get("chemical_data", {})
                    data = chem_data.get("data", {})
                    chem_comp = data.get("chem_comp", {})
                    
                    if chem_comp:
                        # Get basic info
                        comp_info = chem_comp.get("chem_comp", {})
                        formula = comp_info.get("formula", "")
                        formula_weight = comp_info.get("formula_weight", 0)
                        
                        # Get SMILES for complexity estimation
                        descriptors = chem_comp.get("rcsb_chem_comp_descriptor", {})
                        smiles = descriptors.get("SMILES_stereo", "")
                        
                        # Calculate complexity score based on multiple factors
                        complexity_score = 0
                        if formula_weight:
                            complexity_score += min(formula_weight / 100, 10)  # Weight contribution (max 10)
                        if smiles:
                            complexity_score += min(len(smiles) / 10, 10)  # SMILES length contribution (max 10)
                        
                        component_details[comp_id] = {
                            'formula': formula,
                            'weight': formula_weight,
                            'smiles': smiles,
                            'complexity_score': complexity_score
                        }
                        
                        logging.debug(f"{pdb_id} component {comp_id}: weight={formula_weight}, smiles_len={len(smiles)}, score={complexity_score}")
                        
                except Exception as e:
                    logging.debug(f"Error fetching detailed data for component {comp_id} in {pdb_id}: {e}")
                    # Fallback: assign low complexity score
                    component_details[comp_id] = {
                        'formula': '',
                        'weight': 0,
                        'smiles': '',
                        'complexity_score': 1
                    }
        
        if component_details:
            # Sort by complexity score (highest first) to get the most drug-like ligand
            sorted_components = sorted(component_details.items(), 
                                     key=lambda x: x[1]['complexity_score'], 
                                     reverse=True)
            
            primary_comp_id = sorted_components[0][0]
            primary_details = sorted_components[0][1]
            
            logging.debug(f"Selected {primary_comp_id} as primary ligand for {pdb_id} "
                         f"(weight: {primary_details['weight']}, "
                         f"SMILES length: {len(primary_details['smiles'])}, "
                         f"complexity: {primary_details['complexity_score']:.2f})")
            
            return primary_comp_id
        else:
            # Fallback: filter out known small components and take the first remaining
            drug_like_components = [comp_id for comp_id in comp_ids if comp_id not in common_components]
            
            if drug_like_components:
                primary_ligand = drug_like_components[0]
                logging.debug(f"Fallback: Selected {primary_ligand} as primary ligand for {pdb_id}")
                return primary_ligand
            else:
                # Last resort: return the first component
                if comp_ids:
                    primary_ligand = comp_ids[0]
                    logging.debug(f"Last resort: Using {primary_ligand} for {pdb_id}")
                    return primary_ligand
                else:
                    return "N/A"
        
    except Exception as e:
        logging.error(f"Error getting ligand CCD for {pdb_id}: {e}")
        return "N/A"

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

def compute_ligand_rmsd_with_pymol(model_path, ref_path, pdb_id, model_type, ligand_ccd):
    """Compute ligand RMSD using PyMol by aligning organic molecules with specific ligand fallback."""
    try:
        # Load structures
        model_name = f"{pdb_id}_{model_type}_ligand_model"
        ref_name = f"{pdb_id}_ligand_ref"
        cmd.load(model_path, model_name)
        cmd.load(ref_path, ref_name)
        
        # Select organic molecules (ligands) from both structures
        cmd.select("lig_ref", f"{ref_name} and (organic)")
        cmd.select("lig_model", f"{model_name} and (organic)")
        
        # Check if ligands exist in both structures
        ref_ligand_count = cmd.count_atoms("lig_ref")
        model_ligand_count = cmd.count_atoms("lig_model")
        
        if ref_ligand_count == 0:
            logging.debug(f"No organic ligands found in reference structure for {pdb_id}")
            cmd.delete("all")
            return "N/A"
            
        if model_ligand_count == 0:
            logging.debug(f"No organic ligands found in model structure for {pdb_id} ({model_type})")
            cmd.delete("all")
            return "N/A"
        
        ligand_rmsd = "N/A"
        method_used = "none"
        
        # Try align with organic selection first
        try:
            alignment_result = cmd.align("lig_model", "lig_ref", quiet=1)
            if alignment_result and alignment_result[0] is not None and alignment_result[0] > 0:
                ligand_rmsd = alignment_result[0]
                method_used = "align_organic"
                logging.debug(f"Successfully used align with organic selection for {pdb_id} ({model_type}): RMSD = {ligand_rmsd}")
        except Exception as align_error:
            logging.debug(f"Align with organic selection failed for {pdb_id} ({model_type}): {align_error}")
        
        # If organic align failed, try specific ligand selection with align first
        if ligand_rmsd == "N/A" and ligand_ccd != "N/A":
            try:
                logging.debug(f"Trying specific ligand selection with align for {pdb_id} ({model_type}) with ligand CCD: {ligand_ccd}")
                
                # Select reference ligand by CCD
                cmd.select("lig_ref_specific", f"{ref_name} and resn {ligand_ccd}")
                ref_specific_count = cmd.count_atoms("lig_ref_specific")
                
                if ref_specific_count == 0:
                    logging.debug(f"No ligand found with CCD {ligand_ccd} in reference structure for {pdb_id}")
                else:
                    # Select model ligand based on model type
                    model_ligand_found = False
                    lig_resn_used = None
                    
                    if "smiles" in model_type.lower():
                        # For SMILES models, look for residues starting with LIG_
                        # Get all residue names in the model
                        try:
                            # Get list of residue names in the model
                            model_residues = []
                            cmd.iterate(f"{model_name}", "model_residues.append(resn)", space=locals())
                            lig_residues = [resn for resn in set(model_residues) if resn.startswith("LIG")]
                            
                            if lig_residues:
                                lig_resn_used = lig_residues[0]  # Take the first LIG_ residue found
                                cmd.select("lig_model_specific", f"{model_name} and resn {lig_resn_used}")
                                model_specific_count = cmd.count_atoms("lig_model_specific")
                                if model_specific_count > 0:
                                    model_ligand_found = True
                                    logging.debug(f"Found SMILES ligand {lig_resn_used} in model for {pdb_id}")
                        except Exception as e:
                            logging.debug(f"Error finding LIG_ residue in SMILES model for {pdb_id}: {e}")
                    
                    else:
                        # For CCD models, use the same CCD as reference
                        lig_resn_used = ligand_ccd
                        cmd.select("lig_model_specific", f"{model_name} and resn {ligand_ccd}")
                        model_specific_count = cmd.count_atoms("lig_model_specific")
                        if model_specific_count > 0:
                            model_ligand_found = True
                            logging.debug(f"Found CCD ligand {ligand_ccd} in model for {pdb_id}")
                    
                    # If both specific ligands found, try align first
                    if model_ligand_found:
                        # Try align with specific selections first
                        try:
                            alignment_result = cmd.align("lig_model_specific", "lig_ref_specific", quiet=1)
                            if alignment_result and alignment_result[0] is not None and alignment_result[0] > 0:
                                ligand_rmsd = alignment_result[0]
                                method_used = "align_specific"
                                logging.debug(f"Successfully used specific ligand align for {pdb_id} ({model_type}): RMSD = {ligand_rmsd}")
                        except Exception as align_specific_error:
                            logging.debug(f"Specific ligand align failed for {pdb_id} ({model_type}): {align_specific_error}")
                        
                        # If align with specific selections failed, try pair_fit as last resort
                        if ligand_rmsd == "N/A":
                            try:
                                logging.debug(f"Trying pair_fit as last resort for {pdb_id} ({model_type})")
                                pair_fit_result = cmd.pair_fit("lig_model_specific", "lig_ref_specific", quiet=1)
                                if pair_fit_result is not None and pair_fit_result > 0:
                                    ligand_rmsd = pair_fit_result  # pair_fit returns RMSD directly as float
                                    method_used = "pair_fit_specific"
                                    logging.debug(f"Successfully used specific ligand pair fit for {pdb_id} ({model_type}): RMSD = {ligand_rmsd}")
                            except Exception as pair_fit_error:
                                logging.debug(f"Pair_fit also failed for {pdb_id} ({model_type}): {pair_fit_error}")
                        
                        # Final fallback: try pair_fit without hydrogen atoms
                        if ligand_rmsd == "N/A":
                            try:
                                logging.debug(f"Trying pair_fit without hydrogens as final fallback for {pdb_id} ({model_type})")
                                
                                # Create selections without hydrogen atoms
                                cmd.select("lig_ref_no_h", f"lig_ref_specific and not hydro")
                                cmd.select("lig_model_no_h", f"lig_model_specific and not hydro")
                                
                                # Check if we still have atoms after removing hydrogens
                                ref_no_h_count = cmd.count_atoms("lig_ref_no_h")
                                model_no_h_count = cmd.count_atoms("lig_model_no_h")
                                
                                if ref_no_h_count > 0 and model_no_h_count > 0:
                                    pair_fit_no_h_result = cmd.pair_fit("lig_model_no_h", "lig_ref_no_h", quiet=1)
                                    if pair_fit_no_h_result is not None and pair_fit_no_h_result > 0:
                                        ligand_rmsd = pair_fit_no_h_result
                                        method_used = "pair_fit_no_hydrogens"
                                        logging.debug(f"Successfully used pair_fit without hydrogens for {pdb_id} ({model_type}): RMSD = {ligand_rmsd}")
                                else:
                                    logging.debug(f"No atoms remaining after hydrogen removal for {pdb_id} ({model_type})")
                                    
                            except Exception as no_h_error:
                                logging.debug(f"Pair_fit without hydrogens also failed for {pdb_id} ({model_type}): {no_h_error}")
                    else:
                        logging.debug(f"Could not find matching ligand in model for {pdb_id} ({model_type})")
                    
            except Exception as specific_error:
                logging.debug(f"Specific ligand selection also failed for {pdb_id} ({model_type}): {specific_error}")
        
        # Clean up
        cmd.delete("all")
        
        # Log the final result
        if ligand_rmsd != "N/A":
            logging.debug(f"Final ligand RMSD for {pdb_id} ({model_type}): {ligand_rmsd} (method: {method_used})")
        else:
            logging.debug(f"Both organic and specific ligand alignment failed for {pdb_id} ({model_type})")
        
        return ligand_rmsd
        
    except Exception as e:
        logging.error(f"Error computing ligand RMSD for {pdb_id} ({model_type}): {e}")
        cmd.delete("all")
        return "N/A"

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
        logging.error(f"Error running DockQ: {e}")
        return None

def process_pdb_folder(folder_path, pdb_id, results, model_adapter):
    """Process a single PDB ID folder with multiple seeds."""
    ref_path = os.path.join(folder_path, f"{pdb_id}.cif")
    analysis_file = os.path.join(folder_path, f"{pdb_id}_analysis.txt")
    
    # Create images folder for screenshots
    images_folder = os.path.join(folder_path, "images")
    os.makedirs(images_folder, exist_ok=True)
    
    # Check if reference file exists
    if not os.path.exists(ref_path):
        logging.warning(f"Reference file {ref_path} not found. Skipping {pdb_id}.")
        return
    
    # Extract POI and E3 information if available
    poi_name, poi_sequence, e3_name, e3_sequence = None, None, None, None
    if os.path.exists(analysis_file):
        poi_name, poi_sequence, e3_name, e3_sequence = extract_component_info(analysis_file)
    
    # Get the primary ligand CCD identifier
    ligand_ccd = get_primary_ligand_ccd(pdb_id)
    
    # Find and store primary ligand details
    ligand_id = None
    ligand_info = {}
    try:
        smile_strings = fetch_smile_strings(pdb_id)
        if smile_strings:
            ligand_id = list(smile_strings.keys())[0]
            ligand_info = {
                "LIGAND_CCD": ligand_id,
                "LIGAND_LINK": f"https://www.rcsb.org/ligand/{ligand_id}",
                "LIGAND_SMILES": smile_strings[ligand_id].get('SMILES_stereo', "N/A")
            }
        else:
            logging.info(f"No suitable ligands found for {pdb_id}")
            ligand_info = {
                "LIGAND_CCD": ligand_ccd if ligand_ccd != "N/A" else "N/A",
                "LIGAND_LINK": f"https://www.rcsb.org/ligand/{ligand_ccd}" if ligand_ccd != "N/A" else "N/A",
                "LIGAND_SMILES": "N/A"
            }
    except Exception as e:
        logging.error(f"Error fetching SMILE strings for {pdb_id}: {e}")
        ligand_info = {
            "LIGAND_CCD": ligand_ccd if ligand_ccd != "N/A" else "N/A",
            "LIGAND_LINK": f"https://www.rcsb.org/ligand/{ligand_ccd}" if ligand_ccd != "N/A" else "N/A",
            "LIGAND_SMILES": "N/A"
        }
    
    # Calculate and prepare molecular property fields
    mol_properties = {}
    if ligand_info["LIGAND_SMILES"] != "N/A":
        mol_properties = calculate_molecular_properties_from_smiles(ligand_info["LIGAND_SMILES"])
    
    # Find model folders with seed information for both AlphaFold and Boltz-1
    ccd_model_dict = {}
    smiles_model_dict = {}
    
    # Collect all folders in the PDB ID directory
    all_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    # Pattern for folders with seed numbers
    # Handle both regular and ternary naming patterns
    model_patterns = [
        re.compile(f"{pdb_id.lower()}_([a-z]+)_(\d+)"),
        re.compile(f"{pdb_id.lower()}_ternary_([a-z]+)_(\d+)")
    ]
    
    # Find all model folders with seeds
    for folder_name in all_folders:
        match = None
        is_ternary = False
        
        # Try both regular and ternary patterns
        for pattern in model_patterns:
            match = pattern.match(folder_name)
            if match:
                is_ternary = "ternary" in folder_name
                break
                
        if match:
            model_type, seed = match.groups()
            
            # Get model paths using the adapter
            model_path, json_path = model_adapter.get_model_paths(
                folder_path, folder_name, pdb_id, model_type, seed, is_ternary=(is_ternary)
            )
            
            if not model_path:
                logging.debug(f"No model file found for {folder_name}")
                continue
            
            # Store model information
            if model_type == "ccd":
                ccd_model_dict[seed] = {
                    "model_path": model_path,
                    "json_path": json_path
                }
                logging.debug(f"Found CCD model with seed {seed}: {model_path}")
            elif model_type == "smiles":
                smiles_model_dict[seed] = {
                    "model_path": model_path,
                    "json_path": json_path
                }
                logging.debug(f"Found SMILES model with seed {seed}: {model_path}")
    
    # Check for legacy Boltz-1 single model format as fallback
    is_boltz1 = isinstance(model_adapter, Boltz1Adapter)
    if is_boltz1 and not ccd_model_dict and not smiles_model_dict:
        logging.info(f"No seed folders found for {pdb_id}, checking for legacy Boltz-1 format...")
        
        # Try to find models directly in the main folder
        ccd_model_path, ccd_json_path = model_adapter.get_model_paths(folder_path, "", pdb_id, "ccd", "0")
        smiles_model_path, smiles_json_path = model_adapter.get_model_paths(folder_path, "", pdb_id, "smiles", "0")
        
        # Use 42 as the default seed for legacy format
        if ccd_model_path:
            ccd_model_dict["42"] = {"model_path": ccd_model_path, "json_path": ccd_json_path}
        if smiles_model_path:
            smiles_model_dict["42"] = {"model_path": smiles_model_path, "json_path": smiles_json_path}
    
    # Get all unique seeds
    all_seeds = sorted(set(list(ccd_model_dict.keys()) + list(smiles_model_dict.keys())))
    
    # Exit if no models found
    if not all_seeds:
        logging.warning(f"No models found for {pdb_id}")
        return
        
    for seed in all_seeds:
        logging.info(f"Processing {pdb_id} with seed {seed}...")
        
        # Start with PDB ID and common information
        result_row = {
            "PDB_ID": pdb_id,
            "RELEASE_DATE": get_pdb_release_date(pdb_id) or "N/A",
            "SEED": seed,
            "POI_NAME": poi_name if poi_name else "N/A",
            "POI_SEQUENCE": poi_sequence if poi_sequence else "N/A",
            "E3_NAME": e3_name if e3_name else "N/A",
            "E3_SEQUENCE": e3_sequence if e3_sequence else "N/A"
        }
        
        # Add ligand information
        result_row.update(ligand_info)
        
        # Add molecular properties
        for prop in ['Molecular_Weight', 'Heavy_Atom_Count', 'Ring_Count', 'Rotatable_Bond_Count',
                    'LogP', 'HBA_Count', 'HBD_Count', 'TPSA', 'Aromatic_Rings', 'Aliphatic_Rings']:
            result_row[prop] = mol_properties.get(prop, "N/A")
        
        # Initialize metrics with default values
        for metric in ["SMILES_RMSD", "SMILES_POI_RMSD", "SMILES_E3_RMSD", 
                       "SMILES_DOCKQ_SCORE", "SMILES_DOCKQ_iRMSD", "SMILES_DOCKQ_LRMSD",
                       "CCD_RMSD", "CCD_POI_RMSD", "CCD_E3_RMSD", 
                       "CCD_DOCKQ_SCORE", "CCD_DOCKQ_iRMSD", "CCD_DOCKQ_LRMSD",
                       "SMILES_FRACTION_DISORDERED", "SMILES_HAS_CLASH", "SMILES_IPTM", 
                       "SMILES_PTM", "SMILES_RANKING_SCORE", "CCD_FRACTION_DISORDERED", 
                       "CCD_HAS_CLASH", "CCD_IPTM", "CCD_PTM", "CCD_RANKING_SCORE"]:
            result_row[metric] = "N/A"
        
        # Process SMILES model for this seed
        if seed in smiles_model_dict:
            smiles_model_path = smiles_model_dict[seed]["model_path"]
            smiles_json_path = smiles_model_dict[seed]["json_path"]
            
            try:
                # Compute overall RMSD
                smiles_rmsd = compute_rmsd_with_pymol(smiles_model_path, ref_path, pdb_id, f"smiles_{seed}")
                result_row["SMILES_RMSD"] = smiles_rmsd
                logging.debug(f"{pdb_id} seed {seed} SMILES RMSD: {smiles_rmsd}")
                
                # Compute component-specific RMSD if sequences are available
                if poi_sequence or e3_sequence:
                    component_rmsd = calculate_component_rmsd(
                        smiles_model_path, ref_path, pdb_id, f"smiles_{seed}", 
                        poi_sequence, e3_sequence
                    )
                    result_row["SMILES_POI_RMSD"] = component_rmsd["POI_RMSD"]
                    result_row["SMILES_E3_RMSD"] = component_rmsd["E3_RMSD"]
                    logging.debug(f"{pdb_id} seed {seed} SMILES POI RMSD: {component_rmsd['POI_RMSD']}, E3 RMSD: {component_rmsd['E3_RMSD']}")
                
                # Run DockQ
                smiles_dockq_output = run_dockq(smiles_model_path, ref_path)
                if smiles_dockq_output:
                    dockq_score, irmsd, lrmsd = extract_dockq_values(smiles_dockq_output)
                    result_row["SMILES_DOCKQ_SCORE"] = dockq_score if dockq_score is not None else "N/A"
                    result_row["SMILES_DOCKQ_iRMSD"] = irmsd if irmsd is not None else "N/A"
                    result_row["SMILES_DOCKQ_LRMSD"] = lrmsd if lrmsd is not None else "N/A"
                    logging.debug(f"{pdb_id} seed {seed} SMILES DockQ: {dockq_score}, iRMSD: {irmsd}, LRMSD: {lrmsd}")
                
                # Compute ligand RMSD
                smiles_ligand_rmsd = compute_ligand_rmsd_with_pymol(smiles_model_path, ref_path, pdb_id, f"smiles_{seed}", ligand_ccd)
                result_row["SMILES_PROTAC_RMSD"] = smiles_ligand_rmsd
                logging.debug(f"{pdb_id} seed {seed} SMILES Ligand RMSD: {smiles_ligand_rmsd}")
                    
            except Exception as e:
                logging.error(f"Error processing SMILES model for {pdb_id} with seed {seed}: {e}")
            
            # Extract confidence values
            if smiles_json_path:
                try:
                    fraction_disordered, has_clash, iptm, ptm, ranking_score = model_adapter.extract_confidence_values(smiles_json_path)
                    result_row["SMILES_FRACTION_DISORDERED"] = fraction_disordered if fraction_disordered is not None else "N/A"
                    result_row["SMILES_HAS_CLASH"] = has_clash if has_clash is not None else "N/A"
                    result_row["SMILES_IPTM"] = iptm if iptm is not None else "N/A"
                    result_row["SMILES_PTM"] = ptm if ptm is not None else "N/A"
                    result_row["SMILES_RANKING_SCORE"] = ranking_score if ranking_score is not None else "N/A"
                    logging.debug(f"{pdb_id} seed {seed} SMILES confidence metrics extracted")
                except Exception as e:
                    logging.error(f"Error extracting SMILES confidence values for {pdb_id} with seed {seed}: {e}")
        
        # Process CCD model for this seed
        if seed in ccd_model_dict:
            ccd_model_path = ccd_model_dict[seed]["model_path"]
            ccd_json_path = ccd_model_dict[seed]["json_path"]
            
            try:
                # Compute overall RMSD
                ccd_rmsd = compute_rmsd_with_pymol(ccd_model_path, ref_path, pdb_id, f"ccd_{seed}")
                result_row["CCD_RMSD"] = ccd_rmsd
                logging.debug(f"{pdb_id} seed {seed} CCD RMSD: {ccd_rmsd}")
                
                # Compute component-specific RMSD if sequences are available
                if poi_sequence or e3_sequence:
                    component_rmsd = calculate_component_rmsd(
                        ccd_model_path, ref_path, pdb_id, f"ccd_{seed}", 
                        poi_sequence, e3_sequence
                    )
                    result_row["CCD_POI_RMSD"] = component_rmsd["POI_RMSD"]
                    result_row["CCD_E3_RMSD"] = component_rmsd["E3_RMSD"]
                    logging.debug(f"{pdb_id} seed {seed} CCD POI RMSD: {component_rmsd['POI_RMSD']}, E3 RMSD: {component_rmsd['E3_RMSD']}")
                
                # Run DockQ
                ccd_dockq_output = run_dockq(ccd_model_path, ref_path)
                if ccd_dockq_output:
                    dockq_score, irmsd, lrmsd = extract_dockq_values(ccd_dockq_output)
                    result_row["CCD_DOCKQ_SCORE"] = dockq_score if dockq_score is not None else "N/A"
                    result_row["CCD_DOCKQ_iRMSD"] = irmsd if irmsd is not None else "N/A"
                    result_row["CCD_DOCKQ_LRMSD"] = lrmsd if lrmsd is not None else "N/A"
                    logging.debug(f"{pdb_id} seed {seed} CCD DockQ: {dockq_score}, iRMSD: {irmsd}, LRMSD: {lrmsd}")
                
                # Compute ligand RMSD
                ccd_ligand_rmsd = compute_ligand_rmsd_with_pymol(ccd_model_path, ref_path, pdb_id, f"ccd_{seed}", ligand_ccd)
                result_row["CCD_PROTAC_RMSD"] = ccd_ligand_rmsd
                logging.debug(f"{pdb_id} seed {seed} CCD Ligand RMSD: {ccd_ligand_rmsd}")
                    
            except Exception as e:
                logging.error(f"Error processing CCD model for {pdb_id} with seed {seed}: {e}")
            
            # Extract confidence values
            if ccd_json_path:
                try:
                    fraction_disordered, has_clash, iptm, ptm, ranking_score = model_adapter.extract_confidence_values(ccd_json_path)
                    result_row["CCD_FRACTION_DISORDERED"] = fraction_disordered if fraction_disordered is not None else "N/A"
                    result_row["CCD_HAS_CLASH"] = has_clash if has_clash is not None else "N/A"
                    result_row["CCD_IPTM"] = iptm if iptm is not None else "N/A"
                    result_row["CCD_PTM"] = ptm if ptm is not None else "N/A"
                    result_row["CCD_RANKING_SCORE"] = ranking_score if ranking_score is not None else "N/A"
                    logging.debug(f"{pdb_id} seed {seed} CCD confidence metrics extracted")
                except Exception as e:
                    logging.error(f"Error extracting CCD confidence values for {pdb_id} with seed {seed}: {e}")
        
        # Create side-by-side ligand visualizations for this seed
        if seed in smiles_model_dict and seed in ccd_model_dict:
            smiles_model_path = smiles_model_dict[seed]["model_path"]
            ccd_model_path = ccd_model_dict[seed]["model_path"]
            
            try:
                # Create seed-specific image directory
                seed_images_folder = os.path.join(images_folder, f"seed_{seed}")
                os.makedirs(seed_images_folder, exist_ok=True)
                
                side_by_side_screenshots = capture_side_by_side_views(
                    f"{pdb_id}_seed_{seed}",
                    smiles_model_path,
                    ccd_model_path,
                    ref_path,
                    seed_images_folder,
                    ligand_id
                )
                
                if not side_by_side_screenshots:
                    logging.warning(f"Failed to create side-by-side ligand views for {pdb_id} with seed {seed}")
                else:
                    logging.debug(f"Created {len(side_by_side_screenshots)} side-by-side views for {pdb_id} with seed {seed}")
            except Exception as e:
                logging.error(f"Error creating side-by-side ligand views for {pdb_id} with seed {seed}: {e}")
        
        results.append(result_row)
        logging.info(f"Completed processing {pdb_id} with seed {seed}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate structure predictions against experimental structures.")
    parser.add_argument("--protac", help="Path to the folder containing PROTAC PDB ID folders")
    parser.add_argument("--glue", help="Path to the folder containing Molecular Glue PDB ID folders")
    parser.add_argument("--e3_ligand", help="Path to the folder containing E3 Ligand PDB ID folders")
    parser.add_argument("--linker", help="Path to the folder containing Linker PDB ID folders")
    parser.add_argument("--warhead", help="Path to the folder containing Warhead PDB ID folders")
    parser.add_argument("--output", "-o", default="evaluation_results.csv", 
                        help="Output CSV file name or full path (default: saves to ../data/af3_results/)")
    parser.add_argument("--log", help="Path to output log file (default: None, logs to console only)")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        default="INFO", help="Set the logging level (default: INFO)")
    parser.add_argument("--model_type", choices=["AlphaFold3", "Boltz1"], 
                        default="AlphaFold3", help="Structure prediction model type (default: AlphaFold3)")
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(args.log, log_level)
    
    # Initialize the appropriate model adapter
    model_adapter = get_model_adapter(args.model_type)
    
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
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Define folder types with their corresponding TYPE values
    folder_mapping = {
        "protac_folder": {"path": args.protac, "type": "PROTAC"},
        "glue_folder": {"path": args.glue, "type": "MOLECULAR GLUE"},
        "e3_ligand_folder": {"path": args.e3_ligand, "type": "E3 LIGAND"},
        "linker_folder": {"path": args.linker, "type": "LINKER"},
        "warhead_folder": {"path": args.warhead, "type": "WARHEAD"}
    }
    
    # Process each folder type
    for folder_key, folder_info in folder_mapping.items():
        folder_path = folder_info["path"]
        if not folder_path:
            continue
            
        if not os.path.exists(folder_path):
            logging.error(f"Error: {folder_key} {folder_path} does not exist.")
            continue
        
        logging.info(f"Processing {folder_info['type']} structures...")
        folder_results = []
        
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                logging.info(f"Processing {folder_info['type']} {item}...")
                process_pdb_folder(item_path, item, folder_results, model_adapter)
        
        # Add type information
        for result in folder_results:
            result["TYPE"] = folder_info["type"]
        
        # Add results to the final results list
        results.extend(folder_results)
    
    # Create dataframe and save to CSV
    if results:
        df = pd.DataFrame(results)
        
        # Add MODEL_TYPE column
        df["MODEL_TYPE"] = args.model_type
        
        # Define common column order
        column_order = [
            'PDB_ID', 'RELEASE_DATE', 'SEED', 'TYPE', 'MODEL_TYPE', 'POI_NAME', 'POI_SEQUENCE', 'E3_NAME', 'E3_SEQUENCE',
            'SMILES_RMSD', 'SMILES_POI_RMSD', 'SMILES_E3_RMSD', 'SMILES_DOCKQ_SCORE', 'SMILES_DOCKQ_iRMSD', 'SMILES_DOCKQ_LRMSD',
            'SMILES_PROTAC_RMSD', 'SMILES_FRACTION_DISORDERED', 'SMILES_HAS_CLASH', 'SMILES_IPTM', 'SMILES_PTM', 'SMILES_RANKING_SCORE',
            'CCD_RMSD', 'CCD_POI_RMSD', 'CCD_E3_RMSD', 'CCD_DOCKQ_SCORE', 'CCD_DOCKQ_iRMSD', 'CCD_DOCKQ_LRMSD',
            'CCD_PROTAC_RMSD', 'CCD_FRACTION_DISORDERED', 'CCD_HAS_CLASH', 'CCD_IPTM', 'CCD_PTM', 'CCD_RANKING_SCORE',
            'LIGAND_CCD', 'LIGAND_LINK', 'LIGAND_SMILES',
            'Molecular_Weight', 'Heavy_Atom_Count', 'Ring_Count', 'Rotatable_Bond_Count',
            'LogP', 'HBA_Count', 'HBD_Count', 'TPSA', 'Aromatic_Rings', 'Aliphatic_Rings'
        ]
        
        # Ensure all columns exist
        for col in column_order:
            if col not in df.columns:
                df[col] = "N/A"
        
        # Sort and save the model-specific output file
        model_df = df[column_order].sort_values(by=['PDB_ID', 'RELEASE_DATE'])
        model_df.to_csv(output_path, index=False)
        logging.info(f"Results saved to {output_path}")
    else:
        logging.warning("No results to save.")

if __name__ == "__main__":
    main()