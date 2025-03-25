import streamlit as st
import json
import io
import zipfile
from api import process_pdb_ids, retrieve_fasta_sequence, extract_ligand_ccd_from_pdb, extract_comp_ids, fetch_ligand_data
import ollama
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to store protein analysis with ollama
protein_analysis_results = {}

def extract_chain_info(header):
    """Extract and format chain information from a FASTA header"""
    chain_info = ""
    header_parts = header.split("|")

    if len(header_parts) >= 3:
        # Get the chain part (third element)
        full_chain_info = header_parts[2].strip()

        # Remove "Chains " prefix if present
        if full_chain_info.startswith("Chains "):
            chain_info = full_chain_info[7:]
        else:
            chain_info = full_chain_info

        # Process auth chain IDs if present
        if "[auth " in chain_info:
            processed_chains = []

            for part in chain_info.split(","):
                part = part.strip()
                if "[auth " in part:
                    # Extract just the auth value
                    auth_value = part.split("[auth ")[1].split("]")[0].strip()
                    processed_chains.append(auth_value)
                else:
                    processed_chains.append(part)

            chain_info = ", ".join(processed_chains)

        processed_parts = []
        for part in chain_info.split(", "):
            part = part.strip()
            if part.startswith("Chain "):
                processed_parts.append(part[6:])  # Remove "Chain " prefix
            else:
                processed_parts.append(part)
        
        chain_info = ", ".join(processed_parts)

    return chain_info

def create_alphafold_input(pdb_id, fasta_sequences, ligand_data, ligand_chain_info=None, format_type="ccd"):
    """
    Create an AlphaFold3-compatible input JSON structure

    Args:
        pdb_id: The PDB ID
        fasta_sequences: Dictionary of FASTA sequences keyed by header
        ligand_data: Dictionary of ligand data
        ligand_chain_info: Dictionary mapping ligand comp_ids to chain information
        format_type: Type of output format - "ccd" or "smiles"

    Returns:
        Dictionary representing the AlphaFold3 input JSON
    """
    logging.info(f"create_alphafold_input - PDB ID: {pdb_id}, format_type: {format_type}")
    # Initialize structure
    suffix = f"_{format_type}"
    alphafold_input = {
        "name": f"{pdb_id}{suffix}",
        "sequences": [],
        "modelSeeds": [42],
        "dialect": "alphafold3",
        "version": 1
    }

    logging.debug(f"Input fasta_sequences: {fasta_sequences}")
    logging.debug(f"Input ligand_data: {ligand_data}")
    logging.debug(f"Input ligand_chain_info: {ligand_chain_info}")

    # Process protein sequences
    logging.info("Processing protein sequences...")
    for header, sequence in fasta_sequences.items():
        logging.debug(f"  Header: {header}")
        chain_info = extract_chain_info(header)
        logging.debug(f"  Extracted chain_info: {chain_info}")

        # If there are multiple chains, use the last one
        if chain_info:
            chains = chain_info.split(", ")
            chain_id = chains[0]
        else:
            chain_id = "A"
        logging.debug(f"  Chain ID: {chain_id}")

        # Add protein entry
        protein_entry = {
            "protein": {
                "id": chain_id,
                "sequence": sequence
            }
        }
        alphafold_input["sequences"].append(protein_entry)
        logging.debug(f"  Added protein entry: {protein_entry}")

    # List of common ions and small molecules to exclude
    excluded_molecules = ["ZN", "NA", "CL", "MG", "CA", "K", "FE", "MN", "CU", "CO", "HOH", "SO4", "PO4"]

    # Process ligands
    potential_ligands = []

    # Get SMILES data dictionary
    smiles_data = ligand_chain_info.get("smiles_data", {}) if ligand_chain_info else {}

    logging.info("Processing ligands...")
    if ligand_data:
        for ligand_id, ligand_info in ligand_data.items():
            comp_id = ligand_info.get("pdbx_entity_nonpoly", {}).get("comp_id", "Unknown")
            ligand_name = ligand_info.get("pdbx_entity_nonpoly", {}).get("name", "")
            logging.debug(f"  Ligand comp_id: {comp_id}, name: {ligand_name}")

            # Skip excluded molecules
            if comp_id.upper() in excluded_molecules:
                logging.debug(f"  Skipping excluded molecule: {comp_id}")
                continue

            # Extract chain info for the ligand
            ligand_chain = "X"
            if ligand_chain_info and comp_id in ligand_chain_info and ligand_chain_info[comp_id]:
                chains = ligand_chain_info[comp_id].split(", ")
                ligand_chain = chains[0]
            logging.debug(f"  Ligand chain: {ligand_chain}")

            # Get SMILES string if available
            smiles = smiles_data.get(comp_id)
            logging.debug(f"  Ligand SMILES: {smiles}")

            # Calculate a score based on name length to prioritize complex molecules
            name_score = len(ligand_name) if ligand_name else 0

            potential_ligands.append({
                "comp_id": comp_id,
                "chain": ligand_chain,
                "name": ligand_name,
                "name_score": name_score,
                "smiles": smiles
            })
            logging.debug(f"  Potential ligand added: {potential_ligands[-1]}")

    # Input Protac or molecular glue
    if potential_ligands:
        potential_ligands.sort(key=lambda x: x["name_score"], reverse=True)
        top_ligand = potential_ligands[0]
        logging.debug(f"Top ligand selected: {top_ligand}")

        if format_type == "ccd":
            ligand_entry = {
                "ligand": {
                    "id": top_ligand["chain"],
                    "ccdCodes": [top_ligand["comp_id"]]
                }
            }
            alphafold_input["sequences"].append(ligand_entry)
            logging.debug(f"  Added CCD ligand entry: {ligand_entry}")
        else:
            if top_ligand["smiles"]:
                ligand_entry = {
                    "ligand": {
                        "id": top_ligand["chain"],
                        "smiles": top_ligand["smiles"]
                    }
                }
                alphafold_input["sequences"].append(ligand_entry)
                logging.debug(f"  Added SMILES ligand entry: {ligand_entry}")
    logging.debug(f"Final alphafold_input: {alphafold_input}")

    return alphafold_input

def create_ternary_alphafold_input(pdb_id, fasta_sequences, ligand_data, ligand_chain_info=None, format_type="ccd", analysis_result=None):
    """
    Create an AlphaFold3-compatible input JSON structure, ignoring chain ID in protein matching
    """
    logging.info(f"create_ternary_alphafold_input - PDB ID: {pdb_id}, format_type: {format_type}")
    # Initialize structure
    suffix = f"_ternary_{format_type}"
    alphafold_input = {
        "name": f"{pdb_id}{suffix}",
        "sequences": [],
        "modelSeeds": [42],
        "dialect": "alphafold3",
        "version": 1
    }

    logging.debug(f"Input fasta_sequences: {fasta_sequences}")
    logging.debug(f"Input ligand_data: {ligand_data}")
    logging.debug(f"Input ligand_chain_info: {ligand_chain_info}")
    logging.debug(f"Input analysis_result: {analysis_result}")

    if not analysis_result:
        logging.info("No analysis result provided. Returning empty structure.")
        return alphafold_input

    identified_proteins = []
    if analysis_result.get("protein_of_interest"):
        identified_proteins.append(analysis_result["protein_of_interest"])
    if analysis_result.get("e3_ubiquitin_ligase"):
        identified_proteins.append(analysis_result["e3_ubiquitin_ligase"])
    logging.debug(f"Identified proteins from analysis: {identified_proteins}")

    filtered_sequences = {}
    logging.info("Filtering FASTA sequences...")
    for header, sequence in fasta_sequences.items():
        logging.debug(f"  Header: {header}")
        match_found_for_header = False
        for identified_protein in identified_proteins:
            logging.debug(f"    Checking against identified protein: {identified_protein}")
            header_pdb_id_prefix = header.split('_')[0] if '_' in header else "" # Extract PDB ID prefix (e.g., "8FY0")
            identified_pdb_id_prefix = identified_protein.split('_')[0] if '_' in identified_protein else "" # Extract PDB ID prefix

            header_parts = header.split('|')
            header_protein_name = header_parts[1] if len(header_parts) > 1 else ""

            identified_parts = identified_protein.split('|')
            identified_protein_name = identified_parts[2] if len(identified_parts) > 2 else ""

            logging.debug(f"    Header PDB ID Prefix: {header_pdb_id_prefix}, Identified PDB ID Prefix: {identified_pdb_id_prefix}")
            logging.debug(f"    Header protein name: {header_protein_name}, Identified protein name: {identified_protein_name}")

            # --- Simplified Matching Logic: Lowercase Protein Name Exact Match + PDB ID Prefix Match (Ignoring Chain) ---
            protein_name_match = False
            if header_protein_name.strip() and identified_protein_name.strip():
                if header_protein_name.strip().lower() == identified_protein_name.strip().lower(): # Exact lowercase protein name match
                    protein_name_match = True
                    logging.debug(f"    Protein name EXACT match (lowercase) FOUND.")
                else:
                    logging.debug(f"    Protein name EXACT match (lowercase) NOT found.")
            else:
                logging.debug(f"    One or both protein names are empty, skipping name comparison.")


            if (header_pdb_id_prefix == identified_pdb_id_prefix and # Compare PDB ID prefixes
                protein_name_match): # Use exact lowercase name match
                filtered_sequences[header] = sequence
                logging.debug(f"    Match FOUND (PDB ID Prefix, lowercase protein name - IGNORING CHAIN ID). Adding to filtered sequences.")
                match_found_for_header = True
                break
            else:
                logging.debug(f"    No Match (PDB ID Prefix or lowercase protein name mismatch).") # More informative no match msg

        if not match_found_for_header:
            logging.debug(f"  No identified protein match found for header: {header}")

    logging.debug(f"Filtered sequences: {list(filtered_sequences.keys())}")

    # Process filtered protein sequences
    logging.info("Processing filtered protein sequences...")
    for header, sequence in filtered_sequences.items():
        logging.debug(f"  Header: {header}")
        chain_info = extract_chain_info(header)
        logging.debug(f"  Extracted chain_info: {chain_info}")

        if chain_info:
            chains = chain_info.split(", ")
            chain_id = chains[0]
        else:
            chain_id = "A"
        logging.debug(f"  Chain ID: {chain_id}")

        protein_entry = {
            "protein": {
                "id": chain_id,
                "sequence": sequence
            }
        }
        alphafold_input["sequences"].append(protein_entry)
        logging.debug(f"  Added protein entry: {protein_entry}")

    # Process ligands (same as before - no changes needed)
    excluded_molecules = ["ZN", "NA", "CL", "MG", "CA", "K", "FE", "MN", "CU", "CO", "HOH", "SO4", "PO4"]
    potential_ligands = []
    smiles_data = ligand_chain_info.get("smiles_data", {}) if ligand_chain_info else {}

    logging.info("Processing ligands...")
    if ligand_data:
        for ligand_id, ligand_info in ligand_data.items():
            comp_id = ligand_info.get("pdbx_entity_nonpoly", {}).get("comp_id", "Unknown")
            ligand_name = ligand_info.get("pdbx_entity_nonpoly", {}).get("name", "")
            logging.debug(f"  Ligand comp_id: {comp_id}, name: {ligand_name}")

            if comp_id.upper() in excluded_molecules:
                logging.debug(f"  Skipping excluded molecule: {comp_id}")
                continue

            ligand_chain = "X"
            if ligand_chain_info and comp_id in ligand_chain_info and ligand_chain_info[comp_id]:
                chains = ligand_chain_info[comp_id].split(", ")
                ligand_chain = chains[0]
            logging.debug(f"  Ligand chain: {ligand_chain}")

            smiles = smiles_data.get(comp_id)
            logging.debug(f"  Ligand SMILES: {smiles}")

            name_score = len(ligand_name) if ligand_name else 0

            potential_ligands.append({
                "comp_id": comp_id,
                "chain": ligand_chain,
                "name": ligand_name,
                "name_score": name_score,
                "smiles": smiles
            })
            logging.debug(f"  Potential ligand added: {potential_ligands[-1]}")

    # Input Protac or molecular glue (same as before - no changes needed)
    if potential_ligands:
        potential_ligands.sort(key=lambda x: x["name_score"], reverse=True)
        top_ligand = potential_ligands[0]
        logging.debug(f"Top ligand selected: {top_ligand}")

        if format_type == "ccd":
            ligand_entry = {
                "ligand": {
                    "id": top_ligand["chain"],
                    "ccdCodes": [top_ligand["comp_id"]]
                }
            }
            alphafold_input["sequences"].append(ligand_entry)
            logging.debug(f"  Added CCD ligand entry: {ligand_entry}")
        else:
            if top_ligand["smiles"]:
                ligand_entry = {
                    "ligand": {
                        "id": top_ligand["chain"],
                        "smiles": top_ligand["smiles"]
                    }
                }
                alphafold_input["sequences"].append(ligand_entry)
                logging.debug(f"  Added SMILES ligand entry: {ligand_entry}")

    logging.debug(f"Final ternary alphafold_input: {alphafold_input}")
    return alphafold_input


def run_protein_analysis_in_background(pdb_ids_string):
    """
    Run the protein analysis in a background thread

    Args:
        pdb_ids_string (str): Comma-separated string of PDB IDs
    """
    global protein_analysis_results
    try:
        logging.info(f"Starting protein analysis for PDB IDs: {pdb_ids_string}")
        # Call the analyze_pdb_proteins function from ollama.py
        results = ollama.analyze_pdb_proteins(pdb_ids_string)

        # Store the results in the global variable
        protein_analysis_results.update(results)

        # Log completion
        st.session_state.protein_analysis_status = "completed"
        st.session_state.protein_analysis_message = f"Protein analysis completed for {len(results)} PDB IDs"
        logging.info(f"Protein analysis completed successfully for {list(results.keys())}")

    except Exception as e:
        # Log error
        st.session_state.protein_analysis_status = "error"
        st.session_state.protein_analysis_message = f"Error in protein analysis: {str(e)}"
        logging.error(f"Error during protein analysis: {str(e)}")

def create_ternary_zip(pdb_ids, results):
    """
    Create a ZIP file containing ternary complex files

    Args:
        pdb_ids (str): Comma-separated string of PDB IDs
        results (dict): Results from process_pdb_ids

    Returns:
        io.BytesIO: ZIP file buffer
    """
    logging.info("Creating ternary ZIP file...")
    ternary_zip_buffer = io.BytesIO()

    with zipfile.ZipFile(ternary_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as ternary_zip:
        for pdb_id, data in results.items():
            logging.info(f"Processing PDB ID for ternary ZIP: {pdb_id}")
            if "error" in data:
                logging.warning(f"Skipping PDB ID {pdb_id} due to previous error: {data['error']}")
                continue

            if pdb_id not in protein_analysis_results:
                logging.warning(f"Skipping PDB ID {pdb_id} as it's not in protein_analysis_results.")
                continue

            try:
                # Get the analysis result for this PDB ID
                analysis_result = protein_analysis_results[pdb_id]
                logging.debug(f"Analysis result for {pdb_id}: {analysis_result}")

                # Skip if no proteins were identified
                if not isinstance(analysis_result, dict) or (
                    not analysis_result.get("protein_of_interest") and
                    not analysis_result.get("e3_ubiquitin_ligase")):
                    logging.info(f"Skipping PDB ID {pdb_id} - no proteins identified in analysis.")
                    continue

                # Get FASTA sequences and ligand data
                fasta_sequences = retrieve_fasta_sequence(pdb_id)
                ligand_data = extract_ligand_ccd_from_pdb(pdb_id)

                # Prepare ligand chain info
                ligand_chain_info, ligand_smiles_data = {}, {}
                for ligand_id, ligand_info in ligand_data.items():
                    comp_id = ligand_info.get("pdbx_entity_nonpoly", {}).get("comp_id", "Unknown")
                    try:
                        ligand_data_result = fetch_ligand_data(pdb_id, comp_id)
                        if "chain_info" in ligand_data_result and ligand_data_result["chain_info"]:
                            ligand_chain_info[comp_id] = ligand_data_result["chain_info"]

                        if "chemical_data" in ligand_data_result and "data" in ligand_data_result["chemical_data"]:
                            chem_data = ligand_data_result["chemical_data"]["data"]["chem_comp"]
                            if "rcsb_chem_comp_descriptor" in chem_data and chem_data["rcsb_chem_comp_descriptor"]:
                                descriptors = chem_data["rcsb_chem_comp_descriptor"]
                                if "SMILES_stereo" in descriptors and descriptors["SMILES_stereo"]:
                                    smiles = descriptors["SMILES_stereo"]
                                    ligand_smiles_data[comp_id] = smiles
                    except Exception as e:
                        logging.warning(f"Error fetching ligand data for {comp_id} in PDB {pdb_id}: {str(e)}")

                ligand_chain_info["smiles_data"] = ligand_smiles_data

                # Create the ternary AlphaFold input JSONs
                ternary_ccd_input = create_ternary_alphafold_input(
                    pdb_id, fasta_sequences, ligand_data, ligand_chain_info,
                    format_type="ccd", analysis_result=analysis_result
                )
                ternary_smiles_input = create_ternary_alphafold_input(
                    pdb_id, fasta_sequences, ligand_data, ligand_chain_info,
                    format_type="smiles", analysis_result=analysis_result
                )

                # Only add to ZIP if there are sequences
                if ternary_ccd_input["sequences"]:
                    logging.info(f"Adding ternary input files to ZIP for PDB ID {pdb_id}")
                    # Add the JSON files to the ternary ZIP
                    ccd_json = json.dumps(ternary_ccd_input, indent=2)
                    ternary_zip.writestr(f"{pdb_id}/{pdb_id}_ternary_ccd.json", ccd_json)

                    smiles_json = json.dumps(ternary_smiles_input, indent=2)
                    ternary_zip.writestr(f"{pdb_id}/{pdb_id}_ternary_smiles.json", smiles_json)
                    
                    # Get the filtered sequences for this PDB ID
                    logging.info("Filtering FASTA sequences...")
                    filtered_sequences = {}
                    identified_proteins = []
                    if analysis_result.get("protein_of_interest"):
                        identified_proteins.append(analysis_result["protein_of_interest"])
                    if analysis_result.get("e3_ubiquitin_ligase"):
                        identified_proteins.append(analysis_result["e3_ubiquitin_ligase"])
                    
                    for header, sequence in fasta_sequences.items():
                        for identified_protein in identified_proteins:
                            header_pdb_id_prefix = header.split('_')[0] if '_' in header else ""
                            identified_pdb_id_prefix = identified_protein.split('_')[0] if '_' in identified_protein else ""

                            header_parts = header.split('|')
                            header_protein_name = header_parts[1] if len(header_parts) > 1 else ""

                            identified_parts = identified_protein.split('|')
                            identified_protein_name = identified_parts[2] if len(identified_parts) > 2 else ""

                            protein_name_match = False
                            if header_protein_name.strip() and identified_protein_name.strip():
                                if header_protein_name.strip().lower() == identified_protein_name.strip().lower():
                                    protein_name_match = True

                            if (header_pdb_id_prefix == identified_pdb_id_prefix and protein_name_match):
                                # Create a simplified header for better readability
                                simplified_header = identified_protein
                                if len(identified_parts) >= 3:
                                    simplified_header = f"{identified_parts[0]}|{identified_parts[2]}|{identified_parts[1]}"
                                filtered_sequences[simplified_header] = sequence
                                break
                    
                    # Create PDBID_analysis.txt content
                    analysis_txt_content = ""
                    
                    # Add Protein of Interest
                    if analysis_result.get("protein_of_interest"):
                        poi_key = None
                        for key in filtered_sequences:
                            if analysis_result["protein_of_interest"].split("|")[2].lower() in key.lower():
                                poi_key = key
                                break
                        
                        if poi_key:
                            analysis_txt_content += f"Protein of Interest: {poi_key}\n"
                            analysis_txt_content += f"{filtered_sequences[poi_key]}\n\n"
                    
                    # Add E3 Ubiquitin Ligase
                    if analysis_result.get("e3_ubiquitin_ligase"):
                        e3_key = None
                        for key in filtered_sequences:
                            if analysis_result["e3_ubiquitin_ligase"].split("|")[2].lower() in key.lower():
                                e3_key = key
                                break
                        
                        if e3_key:
                            analysis_txt_content += f"E3 Ubiquitin Ligase: {e3_key}\n"
                            analysis_txt_content += f"{filtered_sequences[e3_key]}\n"
                    
                    # Add the PDBID_analysis.txt to ZIP
                    if analysis_txt_content:
                        ternary_zip.writestr(f"{pdb_id}/{pdb_id}_analysis.txt", analysis_txt_content)
                        logging.info(f"Added analysis_results.txt to ZIP for PDB ID {pdb_id}")
                else:
                    logging.info(f"No sequences in ternary input for PDB ID {pdb_id}, skipping ZIP entry.")

            except Exception as e:
                logging.error(f"Error processing ternary files for PDB ID {pdb_id}: {str(e)}")

    # Reset buffer position
    ternary_zip_buffer.seek(0)

    logging.info("Ternary ZIP file creation completed.")
    return None, ternary_zip_buffer

def main():
    # Set page title and header
    st.set_page_config(page_title="Input File Generator", layout="wide")

    # Initialize session state variables if they don't exist
    if 'protein_analysis_status' not in st.session_state:
        st.session_state.protein_analysis_status = "not_started"
    if 'protein_analysis_message' not in st.session_state:
        st.session_state.protein_analysis_message = ""
    if 'pdb_results' not in st.session_state:
        st.session_state.pdb_results = None
    if 'ternary_zip_buffer' not in st.session_state:
        st.session_state.ternary_zip_buffer = io.BytesIO()
    if 'ternary_debug_file' not in st.session_state:
        st.session_state.ternary_debug_file = None

    # Add logo in top-left corner
    col1, _ = st.columns([1, 5])
    with col1:
        logo_path = os.path.join(os.path.dirname(__file__), "static", "logo.png")
        st.image(logo_path)

    st.title("Input File Generator")

    # Input field for PDB IDs
    st.subheader("Enter Protein DataBank ID(s)")
    pdb_ids = st.text_area(
        "Entry ID(s)",
        placeholder="Enter one or more PDB IDs separated by commas (e.g., 1ABC, 2XYZ)",
        help="Enter one or more PDB IDs separated by commas"
    )

    # Submit button
    if st.button("Submit"):
        if not pdb_ids:
            st.error("Please enter at least one PDB ID")
        else:
            st.session_state.protein_analysis_status = "running_analysis" # Set status to running analysis
            # Clear any previous analysis messages
            st.session_state.protein_analysis_message = ""

            # Create a combined ZIP file for all PDB IDs
            combined_zip_buffer = io.BytesIO()
            with zipfile.ZipFile(combined_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as combined_zip:
                with st.spinner("Fetching data from Protein DataBank..."):
                    # Process the PDB IDs and fetch data
                    results = process_pdb_ids(pdb_ids)

                    # Store results in session state for later use
                    st.session_state.pdb_results = results

                    # Display the results
                    st.success(f"Successfully fetched data for {len(results)} PDB entries")
                    logging.info(f"Successfully fetched data for PDB IDs: {list(results.keys())}")

                    # Create tabs for each PDB ID
                    tabs = st.tabs(list(results.keys()))

                    for i, (pdb_id, data) in enumerate(results.items()):
                        with tabs[i]:
                            if "error" in data:
                                st.error(data["error"])
                                logging.error(f"Error fetching data for PDB ID {pdb_id}: {data['error']}")
                            else:
                                try:
                                    entry_data = data["data"]["entry"]

                                    # Fetch FASTA sequences for this PDB ID
                                    try:
                                        fasta_sequences = retrieve_fasta_sequence(pdb_id)
                                    except Exception as e:
                                        st.warning(f"Could not retrieve sequence data: {str(e)}")
                                        logging.warning(f"Could not retrieve sequence data for PDB ID {pdb_id}: {str(e)}")
                                        fasta_sequences = {}

                                    # Fetch ligand information for this PDB ID
                                    try:
                                        ligand_data = extract_ligand_ccd_from_pdb(pdb_id)
                                        comp_ids = extract_comp_ids(ligand_data)
                                    except Exception as e:
                                        st.warning(f"Could not retrieve ligand data: {str(e)}")
                                        logging.warning(f"Could not retrieve ligand data for PDB ID {pdb_id}: {str(e)}")
                                        ligand_data = {}
                                        comp_ids = []

                                    # Display basic information
                                    st.header(f"PDB ID: {pdb_id}")

                                    # Display protein analysis results if available (initially empty)
                                    global protein_analysis_results
                                    if pdb_id in protein_analysis_results:
                                        st.subheader("Protein Analysis")
                                        analysis_result = protein_analysis_results[pdb_id]

                                        if isinstance(analysis_result, dict):
                                            # New format with both protein of interest and E3 ligase
                                            if analysis_result.get("protein_of_interest"):
                                                st.write(f"**Protein of Interest:** {analysis_result['protein_of_interest']}")
                                            if analysis_result.get("e3_ubiquitin_ligase"):
                                                st.write(f"**E3 Ubiquitin Ligase:** {analysis_result['e3_ubiquitin_ligase']}")
                                        else:
                                            # Old format with just one protein
                                            st.write(f"**Identified Protein:** {analysis_result}")

                                    if entry_data and "struct" in entry_data and entry_data["struct"]:
                                        st.subheader("Structure Title")
                                        st.write(entry_data["struct"]["title"])

                                    # Create columns for metadata
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.subheader("Basic Information")
                                        if "exptl" in entry_data and entry_data["exptl"]:
                                            st.write(f"**Method:** {entry_data['exptl'][0]['method']}")

                                        if "rcsb_accession_info" in entry_data and entry_data["rcsb_accession_info"]:
                                            st.write(f"**Deposit Date:** {entry_data['rcsb_accession_info']['deposit_date']}")
                                            st.write(f"**Release Date:** {entry_data['rcsb_accession_info']['initial_release_date']}")

                                    with col2:
                                        st.subheader("Citation")
                                        if "rcsb_primary_citation" in entry_data and entry_data["rcsb_primary_citation"]:
                                            citation = entry_data["rcsb_primary_citation"]
                                            if "pdbx_database_id_DOI" in citation and citation["pdbx_database_id_DOI"]:
                                                st.write(f"**DOI:** {citation['pdbx_database_id_DOI']}")

                                    # Display polymer entities
                                    if "polymer_entities" in entry_data and entry_data["polymer_entities"]:
                                        st.subheader("Polymer Entities")
                                        for entity_idx, entity in enumerate(entry_data["polymer_entities"]):
                                            entity_title = f"Entity {entity_idx+1}"
                                            if "rcsb_polymer_entity" in entity and entity["rcsb_polymer_entity"]:
                                                if entity["rcsb_polymer_entity"]["pdbx_description"]:
                                                    entity_title = entity["rcsb_polymer_entity"]["pdbx_description"]

                                            with st.expander(entity_title):
                                                if "entity_poly" in entity and entity["entity_poly"]:
                                                    st.write(f"**Type:** {entity['entity_poly']['type']}")
                                                    st.write(f"**Polymer Type:** {entity['entity_poly']['rcsb_entity_polymer_type']}")

                                                if "rcsb_entity_source_organism" in entity and entity["rcsb_entity_source_organism"]:
                                                    organisms = entity["rcsb_entity_source_organism"]
                                                    if organisms:
                                                        st.write(f"**Source Organism:** {organisms[0]['scientific_name']}")

                                                # Find and display matching FASTA sequence
                                                if fasta_sequences:
                                                    entity_description = entity["rcsb_polymer_entity"]["pdbx_description"] if "rcsb_polymer_entity" in entity else ""

                                                    # Try to find matching sequences
                                                    matching_sequences = []
                                                    for header, sequence in fasta_sequences.items():
                                                        if entity_description and entity_description.lower() in header.lower():
                                                            matching_sequences.append((header, sequence))

                                                    # If we found matching sequences, display them
                                                    if matching_sequences:
                                                        for header, sequence in matching_sequences:
                                                            chain_info = extract_chain_info(header)

                                                            if chain_info:
                                                                st.write(f"**Chains:** {chain_info}")

                                                            st.write(f"**Amino Acid Sequence:**")
                                                            st.code(sequence, language=None)
                                                    else:
                                                        # Fallback
                                                        if entity_idx < len(fasta_sequences):
                                                            header, sequence = list(fasta_sequences.items())[entity_idx]
                                                            chain_info = extract_chain_info(header)

                                                            if chain_info:
                                                                st.write(f"**Chains:** {chain_info}")

                                                            st.write(f"**Amino Acid Sequence:**")
                                                            st.code(sequence, language=None)

                                    # Display ligand entities
                                    ligand_chain_info, ligand_smiles_data = {}, {}
                                    if ligand_data:
                                        st.subheader("Ligand Entities")

                                        for ligand_id, ligand_info in ligand_data.items():
                                            comp_id = ligand_info.get("pdbx_entity_nonpoly", {}).get("comp_id", "Unknown")
                                            ligand_name = ligand_info.get("pdbx_entity_nonpoly", {}).get("name", comp_id)
                                            ligand_title = f"{comp_id}"

                                            with st.expander(ligand_title):
                                                # Display ligand details
                                                st.write(f"**Component ID:** {comp_id}")
                                                st.write(f"**Name:** {ligand_name}")

                                                # Fetch and display combined data
                                                try:
                                                    ligand_data_result = fetch_ligand_data(pdb_id, comp_id)

                                                    # Store chain info for AlphaFold input
                                                    if "chain_info" in ligand_data_result and ligand_data_result["chain_info"]:
                                                        ligand_chain_info[comp_id] = ligand_data_result["chain_info"]
                                                        st.write(f"**Chains:** {ligand_data_result['chain_info']}")

                                                    # Display chemical information
                                                    if "chemical_data" in ligand_data_result and "data" in ligand_data_result["chemical_data"]:
                                                        chem_data = ligand_data_result["chemical_data"]["data"]["chem_comp"]

                                                        # Display chemical information
                                                        if "chem_comp" in chem_data and chem_data["chem_comp"]:
                                                            chem_info = chem_data["chem_comp"]
                                                            if "formula" in chem_info:
                                                                st.write(f"**Formula:** {chem_info['formula']}")
                                                            if "formula_weight" in chem_info:
                                                                st.write(f"**Molecular Weight:** {chem_info['formula_weight']}")

                                                        # Display and store SMILES strings
                                                        if "rcsb_chem_comp_descriptor" in chem_data and chem_data["rcsb_chem_comp_descriptor"]:
                                                            descriptors = chem_data["rcsb_chem_comp_descriptor"]
                                                            if "SMILES_stereo" in descriptors and descriptors["SMILES_stereo"]:
                                                                smiles = descriptors["SMILES_stereo"]
                                                                ligand_smiles_data[comp_id] = smiles
                                                except Exception as e:
                                                    st.warning(f"Could not retrieve ligand data: {str(e)}")
                                                    logging.warning(f"Could not retrieve ligand data for ligand {comp_id} in PDB {pdb_id}: {str(e)}")

                                    # Create the AlphaFold input JSON
                                    ligand_chain_info["smiles_data"] = ligand_smiles_data
                                    alphafold_ccd_input = create_alphafold_input(pdb_id, fasta_sequences, ligand_data, ligand_chain_info, format_type="ccd")
                                    alphafold_smiles_input = create_alphafold_input(pdb_id, fasta_sequences, ligand_data, ligand_chain_info, format_type="smiles")

                                    # Add the JSON files to the combined ZIP
                                    ccd_json = json.dumps(alphafold_ccd_input, indent=2)
                                    combined_zip.writestr(f"{pdb_id}/{pdb_id}_ccd.json", ccd_json)

                                    smiles_json = json.dumps(alphafold_smiles_input, indent=2)
                                    combined_zip.writestr(f"{pdb_id}/{pdb_id}_smiles.json", smiles_json)

                                except Exception as e:
                                    st.error(f"Error processing data for {pdb_id}: {str(e)}")
                                    st.json(data)
                                    logging.error(f"Error processing data for PDB ID {pdb_id}: {str(e)}, Data: {data}")

                # Show information message at the bottom
                info_container = st.empty()
                info_container.info("Processing data... Running Protein Analysis in background (this may take a few minutes)...") # Updated message

                # Reset buffer position for combined zip
                combined_zip_buffer.seek(0)

                # Create an initial empty ternary ZIP file (before analysis, will be updated later)
                ternary_debug_file_initial, ternary_zip_buffer_initial = create_ternary_zip(pdb_ids, results)
                st.session_state.ternary_zip_buffer = ternary_zip_buffer_initial
                st.session_state.ternary_debug_file = ternary_debug_file_initial # This will be None now

                # **Run protein analysis synchronously here (no threading)**
                run_protein_analysis_in_background(pdb_ids) # Run analysis in main thread

                # After analysis is done, create the TERTIARY ZIP (now with analysis results)
                ternary_debug_file, ternary_zip_buffer = create_ternary_zip(pdb_ids, results)
                st.session_state.ternary_zip_buffer = ternary_zip_buffer
                st.session_state.ternary_debug_file = ternary_debug_file # Still None

                # Replace the info message with success message (after analysis and ternary zip)
                info_container.success("Processing complete! Click the buttons below to download your files.")
                logging.info("Processing complete for all PDB IDs.")

                # Create a columns layout for the download buttons
                download_cols = st.columns(2)

                # Add the download button for all entities
                with download_cols[0]:
                    st.download_button(
                        label="Download AlphaFold Input Files (All Entities)",
                        data=combined_zip_buffer,
                        file_name=f"alphafold_inputs_full_structure_{time.strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )

                # Add the download button for ternary complexes
                with download_cols[1]:
                    st.download_button(
                        label="Download AlphaFold Input Files (Ternary Only)",
                        data=st.session_state.ternary_zip_buffer, # Use session state ternary_zip_buffer
                        file_name=f"alphafold_inputs_ternary_{time.strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )

            st.session_state.protein_analysis_status = "completed" # Set status to completed after everything

    # Display protein analysis status message if available (now only for errors)
    if st.session_state.protein_analysis_status == "error":
        st.error(st.session_state.protein_analysis_message)
    elif st.session_state.protein_analysis_status == "running_analysis":
        st.info("Running Protein Analysis in the main thread, please wait...") # Indicate analysis is running

if __name__ == "__main__":
    main()