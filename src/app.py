import streamlit as st
import json
import io
import zipfile
import threading
from api import process_pdb_ids, retrieve_fasta_sequence, extract_ligand_ccd_from_pdb, extract_comp_ids, fetch_ligand_data
import ollama
import time
import os

# Global variable to store protein analysis with ollama
protein_analysis_results = {}
DEBUG_DIR = "/home/2024/piza/PROTACFold/debug_files"

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
    # Initialize structure
    suffix = f"_{format_type}"
    alphafold_input = {
        "name": f"{pdb_id}{suffix}",
        "sequences": [],
        "modelSeeds": [42],
        "dialect": "alphafold3",
        "version": 1
    }

    # Create debug files to understand what's happening
    os.makedirs(DEBUG_DIR, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    debug_file = f"{DEBUG_DIR}/create_alphafold_input_{pdb_id}_{format_type}_{timestamp}.txt"

    with open(debug_file, "w") as f:
        f.write(f"create_alphafold_input - PDB ID: {pdb_id}, format_type: {format_type}\n")
        f.write("========================================\n\n")
        f.write("Input fasta_sequences:\n")
        f.write(json.dumps(fasta_sequences, indent=2))
        f.write("\n\nInput ligand_data:\n")
        f.write(json.dumps(ligand_data, indent=2))
        f.write("\n\nInput ligand_chain_info:\n")
        f.write(json.dumps(ligand_chain_info, indent=2))
        f.write("\n\n")

        # Process protein sequences
        f.write("Processing protein sequences...\n")
        for header, sequence in fasta_sequences.items():
            f.write(f"  Header: {header}\n")
            chain_info = extract_chain_info(header)
            f.write(f"  Extracted chain_info: {chain_info}\n")

            # If there are multiple chains, use the last one
            if chain_info:
                chains = chain_info.split(", ")
                chain_id = chains[-1]
            else:
                chain_id = "A"
            f.write(f"  Chain ID: {chain_id}\n")

            # Add protein entry
            protein_entry = {
                "protein": {
                    "id": chain_id,
                    "sequence": sequence
                }
            }
            alphafold_input["sequences"].append(protein_entry)
            f.write(f"  Added protein entry: {protein_entry}\n")
        f.write("\n")

        # List of common ions and small molecules to exclude
        excluded_molecules = ["ZN", "NA", "CL", "MG", "CA", "K", "FE", "MN", "CU", "CO", "HOH", "SO4", "PO4"]

        # Process ligands
        potential_ligands = []

        # Get SMILES data dictionary
        smiles_data = ligand_chain_info.get("smiles_data", {}) if ligand_chain_info else {}

        f.write("Processing ligands...\n")
        if ligand_data:
            for ligand_id, ligand_info in ligand_data.items():
                comp_id = ligand_info.get("pdbx_entity_nonpoly", {}).get("comp_id", "Unknown")
                ligand_name = ligand_info.get("pdbx_entity_nonpoly", {}).get("name", "")
                f.write(f"  Ligand comp_id: {comp_id}, name: {ligand_name}\n")

                # Skip excluded molecules
                if comp_id.upper() in excluded_molecules:
                    f.write(f"  Skipping excluded molecule: {comp_id}\n")
                    continue

                # Extract chain info for the ligand
                ligand_chain = "X"
                if ligand_chain_info and comp_id in ligand_chain_info and ligand_chain_info[comp_id]:
                    chains = ligand_chain_info[comp_id].split(", ")
                    ligand_chain = chains[-1]
                f.write(f"  Ligand chain: {ligand_chain}\n")

                # Get SMILES string if available
                smiles = smiles_data.get(comp_id)
                f.write(f"  Ligand SMILES: {smiles}\n")

                # Calculate a score based on name length to prioritize complex molecules
                name_score = len(ligand_name) if ligand_name else 0

                potential_ligands.append({
                    "comp_id": comp_id,
                    "chain": ligand_chain,
                    "name": ligand_name,
                    "name_score": name_score,
                    "smiles": smiles
                })
                f.write(f"  Potential ligand added: {potential_ligands[-1]}\n")
        f.write("\n")

        # Input Protac or molecular glue
        if potential_ligands:
            potential_ligands.sort(key=lambda x: x["name_score"], reverse=True)
            top_ligand = potential_ligands[0]
            f.write(f"Top ligand selected: {top_ligand}\n")

            if format_type == "ccd":
                ligand_entry = {
                    "ligand": {
                        "id": top_ligand["chain"],
                        "ccdCodes": [top_ligand["comp_id"]]
                    }
                }
                alphafold_input["sequences"].append(ligand_entry)
                f.write(f"  Added CCD ligand entry: {ligand_entry}\n")
            else:
                if top_ligand["smiles"]:
                    ligand_entry = {
                        "ligand": {
                            "id": top_ligand["chain"],
                            "smiles": top_ligand["smiles"]
                        }
                    }
                    alphafold_input["sequences"].append(ligand_entry)
                    f.write(f"  Added SMILES ligand entry: {ligand_entry}\n")
        f.write("\nFinal alphafold_input:\n")
        f.write(json.dumps(alphafold_input, indent=2))

    return alphafold_input

def create_ternary_alphafold_input(pdb_id, fasta_sequences, ligand_data, ligand_chain_info=None, format_type="ccd", analysis_result=None):
    """
    Create an AlphaFold3-compatible input JSON structure with only proteins identified in the analysis results

    Args:
        pdb_id: The PDB ID
        fasta_sequences: Dictionary of FASTA sequences keyed by header
        ligand_data: Dictionary of ligand data
        ligand_chain_info: Dictionary mapping ligand comp_ids to chain information
        format_type: Type of output format - "ccd" or "smiles"
        analysis_result: Dictionary containing protein analysis results

    Returns:
        Dictionary representing the AlphaFold3 input JSON with only identified proteins
    """
    # Initialize structure
    suffix = f"_ternary_{format_type}"
    alphafold_input = {
        "name": f"{pdb_id}{suffix}",
        "sequences": [],
        "modelSeeds": [42],
        "dialect": "alphafold3",
        "version": 1
    }

    # Create debug files to understand what's happening
    os.makedirs(DEBUG_DIR, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    debug_file = f"{DEBUG_DIR}/create_ternary_alphafold_input_{pdb_id}_{format_type}_{timestamp}.txt"

    with open(debug_file, "w") as f:
        f.write(f"create_ternary_alphafold_input - PDB ID: {pdb_id}, format_type: {format_type}\n")
        f.write("========================================\n\n")
        f.write("Input fasta_sequences:\n")
        f.write(json.dumps(fasta_sequences, indent=2))
        f.write("\n\nInput ligand_data:\n")
        f.write(json.dumps(ligand_data, indent=2))
        f.write("\n\nInput ligand_chain_info:\n")
        f.write(json.dumps(ligand_chain_info, indent=2))
        f.write("\n\nInput analysis_result:\n")
        f.write(json.dumps(analysis_result, indent=2))
        f.write("\n\n")

        # If no analysis result, return empty structure
        if not analysis_result:
            f.write("No analysis result provided. Returning empty structure.\n")
            return alphafold_input

        # Extract identified proteins
        identified_proteins = []
        if analysis_result.get("protein_of_interest"):
            identified_proteins.append(analysis_result["protein_of_interest"])
        if analysis_result.get("e3_ubiquitin_ligase"):
            identified_proteins.append(analysis_result["e3_ubiquitin_ligase"])
        f.write(f"Identified proteins from analysis: {identified_proteins}\n")

        # Filter FASTA sequences to only include identified proteins
        filtered_sequences = {}
        f.write("Filtering FASTA sequences...\n")
        for header, sequence in fasta_sequences.items():
            f.write(f"  Header: {header}\n")
            # Check if this sequence matches any identified protein
            for identified_protein in identified_proteins:
                # Extract PDB ID from header
                header_pdb_id = header.split('_')[0] if '_' in header else ""
                identified_pdb_id = identified_protein.split('_')[0] if '_' in identified_protein else ""

                # Extract protein name and chain from header
                header_parts = header.split('|')
                header_protein_name = header_parts[1] if len(header_parts) > 1 else "" # Use index 1 for protein name
                header_chain = extract_chain_info(header) # Extract chain info properly

                # Extract protein name and chain from identified protein
                identified_parts = identified_protein.split('|')
                identified_protein_name = identified_parts[2] if len(identified_parts) > 2 else "" # Use index 2 for protein name
                identified_chain_full = identified_parts[1] if len(identified_parts) > 1 else "" # e.g., "Chain D"
                identified_chain = identified_chain_full.split("Chain ")[-1] if "Chain " in identified_chain_full else "" # Extract just "D"


                f.write(f"    Checking against identified protein: {identified_protein}\n")
                f.write(f"    Header PDB ID: {header_pdb_id}, Identified PDB ID: {identified_pdb_id}\n")
                f.write(f"    Header protein name: {header_protein_name}, Identified protein name: {identified_protein_name}\n")
                f.write(f"    Header chain: {header_chain}, Identified chain: {identified_chain}\n")


                # Check if PDB ID, protein name and Chain match (more accurate comparison)
                if (header_pdb_id == identified_pdb_id and
                    header_protein_name.strip() == identified_protein_name.strip() and # Exact match for protein name
                    header_chain == identified_chain): # Exact match for chain
                    filtered_sequences[header] = sequence
                    f.write(f"    Match found. Adding to filtered sequences.\n")
                    break
            else:
                f.write(f"    No match found for this header.\n")
        f.write(f"Filtered sequences: {list(filtered_sequences.keys())}\n")

        # Process filtered protein sequences
        f.write("Processing filtered protein sequences...\n")
        for header, sequence in filtered_sequences.items():
            f.write(f"  Header: {header}\n")
            chain_info = extract_chain_info(header)
            f.write(f"  Extracted chain_info: {chain_info}\n")

            # If there are multiple chains, use the last one
            if chain_info:
                chains = chain_info.split(", ")
                chain_id = chains[-1]
            else:
                chain_id = "A"
            f.write(f"  Chain ID: {chain_id}\n")

            # Add protein entry
            protein_entry = {
                "protein": {
                    "id": chain_id,
                    "sequence": sequence
                }
            }
            alphafold_input["sequences"].append(protein_entry)
            f.write(f"  Added protein entry: {protein_entry}\n")
        f.write("\n")

        # List of common ions and small molecules to exclude
        excluded_molecules = ["ZN", "NA", "CL", "MG", "CA", "K", "FE", "MN", "CU", "CO", "HOH", "SO4", "PO4"]

        # Process ligands (Ligand processing remains the same as it was correct)
        potential_ligands = []

        # Get SMILES data dictionary
        smiles_data = ligand_chain_info.get("smiles_data", {}) if ligand_chain_info else {}

        f.write("Processing ligands...\n")
        if ligand_data:
            for ligand_id, ligand_info in ligand_data.items():
                comp_id = ligand_info.get("pdbx_entity_nonpoly", {}).get("comp_id", "Unknown")
                ligand_name = ligand_info.get("pdbx_entity_nonpoly", {}).get("name", "")
                f.write(f"  Ligand comp_id: {comp_id}, name: {ligand_name}\n")

                # Skip excluded molecules
                if comp_id.upper() in excluded_molecules:
                    f.write(f"  Skipping excluded molecule: {comp_id}\n")
                    continue

                # Extract chain info for the ligand
                ligand_chain = "X"
                if ligand_chain_info and comp_id in ligand_chain_info and ligand_chain_info[comp_id]:
                    chains = ligand_chain_info[comp_id].split(", ")
                    ligand_chain = chains[-1]
                f.write(f"  Ligand chain: {ligand_chain}\n")

                # Get SMILES string if available
                smiles = smiles_data.get(comp_id)
                f.write(f"  Ligand SMILES: {smiles}\n")

                # Calculate a score based on name length to prioritize complex molecules
                name_score = len(ligand_name) if ligand_name else 0

                potential_ligands.append({
                    "comp_id": comp_id,
                    "chain": ligand_chain,
                    "name": ligand_name,
                    "name_score": name_score,
                    "smiles": smiles
                })
                f.write(f"  Potential ligand added: {potential_ligands[-1]}\n")
        f.write("\n")

        # Input Protac or molecular glue (Ligand selection remains the same as it was correct)
        if potential_ligands:
            potential_ligands.sort(key=lambda x: x["name_score"], reverse=True)
            top_ligand = potential_ligands[0]
            f.write(f"Top ligand selected: {top_ligand}\n")

            if format_type == "ccd":
                ligand_entry = {
                    "ligand": {
                        "id": top_ligand["chain"],
                        "ccdCodes": [top_ligand["comp_id"]]
                    }
                }
                alphafold_input["sequences"].append(ligand_entry)
                f.write(f"  Added CCD ligand entry: {ligand_entry}\n")
            else:
                if top_ligand["smiles"]:
                    ligand_entry = {
                        "ligand": {
                            "id": top_ligand["chain"],
                            "smiles": top_ligand["smiles"]
                        }
                    }
                    alphafold_input["sequences"].append(ligand_entry)
                    f.write(f"  Added SMILES ligand entry: {ligand_entry}\n")

        f.write("\nFinal ternary alphafold_input:\n")
        f.write(json.dumps(alphafold_input, indent=2))

    return alphafold_input

def run_protein_analysis_in_background(pdb_ids_string):
    """
    Run the protein analysis in a background thread

    Args:
        pdb_ids_string (str): Comma-separated string of PDB IDs
    """
    global protein_analysis_results
    try:
        # Call the analyze_pdb_proteins function from ollama.py
        results = ollama.analyze_pdb_proteins(pdb_ids_string)

        # Store the results in the global variable
        protein_analysis_results.update(results)

        # Log completion
        st.session_state.protein_analysis_status = "completed"
        st.session_state.protein_analysis_message = f"Protein analysis completed for {len(results)} PDB IDs"

        # Create debug files to understand what's happening
        os.makedirs(DEBUG_DIR, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Save protein_analysis_results to a file
        with open(f"{DEBUG_DIR}/protein_analysis_results_{timestamp}.txt", "w") as f:
            f.write("Protein Analysis Results:\n")
            f.write("=======================\n\n")
            for pdb_id, result in protein_analysis_results.items():
                f.write(f"PDB ID: {pdb_id}\n")
                if isinstance(result, dict):
                    if result.get("protein_of_interest"):
                        f.write(f"  Protein of Interest: {result['protein_of_interest']}\n")
                    if result.get("e3_ubiquitin_ligase"):
                        f.write(f"  E3 Ubiquitin Ligase: {result['e3_ubiquitin_ligase']}\n")
                else:
                    f.write(f"  Result: {result}\n")
                f.write("\n")

        # Create a debug file for ternary processing
        ternary_debug_file = f"{DEBUG_DIR}/ternary_processing_{timestamp}.txt"
        with open(ternary_debug_file, "w") as debug_f:
            debug_f.write("Ternary Processing Debug (After Analysis Complete):\n")
            debug_f.write("=======================\n\n")
            debug_f.write(f"Protein analysis completed with {len(results)} results\n\n")

            for pdb_id, result in results.items():
                debug_f.write(f"PDB ID: {pdb_id}\n")
                if isinstance(result, dict):
                    if result.get("protein_of_interest"):
                        debug_f.write(f"  Protein of Interest: {result['protein_of_interest']}\n")
                    if result.get("e3_ubiquitin_ligase"):
                        debug_f.write(f"  E3 Ubiquitin Ligase: {result['e3_ubiquitin_ligase']}\n")
                else:
                    debug_f.write(f"  Result: {result}\n")
                debug_f.write("\n")

    except Exception as e:
        # Log error
        st.session_state.protein_analysis_status = "error"
        st.session_state.protein_analysis_message = f"Error in protein analysis: {str(e)}"

def create_ternary_zip(pdb_ids, results):
    """
    Create a ZIP file containing ternary complex files

    Args:
        pdb_ids (str): Comma-separated string of PDB IDs
        results (dict): Results from process_pdb_ids

    Returns:
        io.BytesIO: ZIP file buffer
    """
    # Create a new ZIP file for ternary complexes
    ternary_zip_buffer = io.BytesIO()

    # Create debug files to understand what's happening
    os.makedirs(DEBUG_DIR, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    with zipfile.ZipFile(ternary_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as ternary_zip:
        # Create a debug file for ternary processing
        ternary_debug_file = f"{DEBUG_DIR}/ternary_processing_{timestamp}.txt"
        with open(ternary_debug_file, "w") as debug_f:
            debug_f.write("Ternary Processing Debug:\n")
            debug_f.write("=======================\n\n")

            for pdb_id, data in results.items():
                debug_f.write(f"Processing PDB ID: {pdb_id}\n")

                if "error" in data:
                    debug_f.write(f"  Error in data: {data['error']}\n")
                    debug_f.write("\n")
                    continue

                debug_f.write(f"  PDB ID in protein_analysis_results: {pdb_id in protein_analysis_results}\n")

                if pdb_id not in protein_analysis_results:
                    debug_f.write("  Skipping - not in protein_analysis_results\n\n")
                    continue

                try:
                    # Get the analysis result for this PDB ID
                    analysis_result = protein_analysis_results[pdb_id]
                    debug_f.write(f"  Analysis result type: {type(analysis_result)}\n")
                    debug_f.write(f"  Analysis result: {analysis_result}\n")

                    # Skip if no proteins were identified
                    if not isinstance(analysis_result, dict) or (
                        not analysis_result.get("protein_of_interest") and
                        not analysis_result.get("e3_ubiquitin_ligase")):
                        debug_f.write("  Skipping - no proteins identified\n\n")
                        continue

                    # Get FASTA sequences and ligand data
                    fasta_sequences = retrieve_fasta_sequence(pdb_id)
                    debug_f.write(f"  FASTA sequences count: {len(fasta_sequences)}\n")
                    for header in fasta_sequences.keys():
                        debug_f.write(f"    FASTA header: {header}\n")

                    ligand_data = extract_ligand_ccd_from_pdb(pdb_id)
                    debug_f.write(f"  Ligand data count: {len(ligand_data)}\n")

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
                            debug_f.write(f"    Error fetching ligand data for {comp_id}: {str(e)}\n")

                    ligand_chain_info["smiles_data"] = ligand_smiles_data
                    debug_f.write(f"  Ligand chain info: {ligand_chain_info}\n")

                    # Create the ternary AlphaFold input JSONs
                    ternary_ccd_input = create_ternary_alphafold_input(
                        pdb_id, fasta_sequences, ligand_data, ligand_chain_info,
                        format_type="ccd", analysis_result=analysis_result
                    )

                    # Debug the filtered sequences
                    debug_f.write(f"  Ternary CCD input sequences count: {len(ternary_ccd_input['sequences'])}\n")
                    for i, seq in enumerate(ternary_ccd_input['sequences']):
                        debug_f.write(f"    Sequence {i+1} type: {list(seq.keys())[0]}\n")

                    ternary_smiles_input = create_ternary_alphafold_input(
                        pdb_id, fasta_sequences, ligand_data, ligand_chain_info,
                        format_type="smiles", analysis_result=analysis_result
                    )

                    # Only add to ZIP if there are sequences
                    if ternary_ccd_input["sequences"]:
                        debug_f.write(f"  Adding to ZIP - has {len(ternary_ccd_input['sequences'])} sequences\n")
                        # Add the JSON files to the ternary ZIP
                        ccd_json = json.dumps(ternary_ccd_input, indent=2)
                        ternary_zip.writestr(f"{pdb_id}/{pdb_id}_ternary_ccd.json", ccd_json)

                        smiles_json = json.dumps(ternary_smiles_input, indent=2)
                        ternary_zip.writestr(f"{pdb_id}/{pdb_id}_ternary_smiles.json", smiles_json)
                    else:
                        debug_f.write("  Not adding to ZIP - no sequences\n")

                    debug_f.write("\n")
                except Exception as e:
                    debug_f.write(f"  Error processing ternary files: {str(e)}\n\n")

    # Reset buffer position
    ternary_zip_buffer.seek(0)

    # Return the debug file path and the ZIP buffer
    return ternary_debug_file, ternary_zip_buffer

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
        st.image("/home/2024/piza/PROTACFold/src/static/logo.png")

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

                    # Create tabs for each PDB ID
                    tabs = st.tabs(list(results.keys()))

                    for i, (pdb_id, data) in enumerate(results.items()):
                        with tabs[i]:
                            if "error" in data:
                                st.error(data["error"])
                            else:
                                try:
                                    entry_data = data["data"]["entry"]

                                    # Fetch FASTA sequences for this PDB ID
                                    try:
                                        fasta_sequences = retrieve_fasta_sequence(pdb_id)
                                    except Exception as e:
                                        st.warning(f"Could not retrieve sequence data: {str(e)}")
                                        fasta_sequences = {}

                                    # Fetch ligand information for this PDB ID
                                    try:
                                        ligand_data = extract_ligand_ccd_from_pdb(pdb_id)
                                        comp_ids = extract_comp_ids(ligand_data)
                                    except Exception as e:
                                        st.warning(f"Could not retrieve ligand data: {str(e)}")
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

                # Show information message at the bottom
                info_container = st.empty()
                info_container.info("Processing data... Running Protein Analysis in background (this may take a few minutes)...") # Updated message

                # Reset buffer position for combined zip
                combined_zip_buffer.seek(0)

                # Create an initial empty ternary ZIP file (before analysis, will be updated later)
                ternary_debug_file_initial, ternary_zip_buffer_initial = create_ternary_zip(pdb_ids, results)
                st.session_state.ternary_zip_buffer = ternary_zip_buffer_initial
                st.session_state.ternary_debug_file = ternary_debug_file_initial

                # **Run protein analysis synchronously here (no threading)**
                run_protein_analysis_in_background(pdb_ids) # Run analysis in main thread

                # After analysis is done, create the TERTIARY ZIP (now with analysis results)
                ternary_debug_file, ternary_zip_buffer = create_ternary_zip(pdb_ids, results)
                st.session_state.ternary_zip_buffer = ternary_zip_buffer
                st.session_state.ternary_debug_file = ternary_debug_file

                # Replace the info message with success message (after analysis and ternary zip)
                info_container.success("Processing complete! Click the buttons below to download your files.")

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

                # Add a button to download debug files
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="Download Debug Files",
                    data=open(st.session_state.ternary_debug_file, "rb"), # Use session state ternary_debug_file
                    file_name=f"debug_files_{timestamp}.txt",
                    mime="text/plain"
                )
            st.session_state.protein_analysis_status = "completed" # Set status to completed after everything

    # Display protein analysis status message if available (now only for errors)
    if st.session_state.protein_analysis_status == "error":
        st.error(st.session_state.protein_analysis_message)
    elif st.session_state.protein_analysis_status == "running_analysis":
        st.info("Running Protein Analysis in the main thread, please wait...") # Indicate analysis is running

if __name__ == "__main__":
    main()