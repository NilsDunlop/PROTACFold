import streamlit as st
import json
import io
import zipfile
import threading
from api import process_pdb_ids, retrieve_fasta_sequence, extract_ligand_ccd_from_pdb, extract_comp_ids, fetch_ligand_data
import ollama
import time

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
    
    # Process protein sequences
    for header, sequence in fasta_sequences.items():
        chain_info = extract_chain_info(header)
        
        # If there are multiple chains, use the last one
        if chain_info:
            chains = chain_info.split(", ")
            chain_id = chains[-1]
        else:
            chain_id = "A"
        
        # Add protein entry
        protein_entry = {
            "protein": {
                "id": chain_id,
                "sequence": sequence
            }
        }
        alphafold_input["sequences"].append(protein_entry)
    
    # List of common ions and small molecules to exclude
    excluded_molecules = ["ZN", "NA", "CL", "MG", "CA", "K", "FE", "MN", "CU", "CO", "HOH", "SO4", "PO4"]
    
    # Process ligands
    potential_ligands = []
    
    # Get SMILES data dictionary
    smiles_data = ligand_chain_info.get("smiles_data", {}) if ligand_chain_info else {}
    
    if ligand_data:
        for ligand_id, ligand_info in ligand_data.items():
            comp_id = ligand_info.get("pdbx_entity_nonpoly", {}).get("comp_id", "Unknown")
            ligand_name = ligand_info.get("pdbx_entity_nonpoly", {}).get("name", "")
            
            # Skip excluded molecules
            if comp_id.upper() in excluded_molecules:
                continue
                
            # Extract chain info for the ligand
            ligand_chain = "X"
            if ligand_chain_info and comp_id in ligand_chain_info and ligand_chain_info[comp_id]:
                chains = ligand_chain_info[comp_id].split(", ")
                ligand_chain = chains[-1]
            
            # Get SMILES string if available
            smiles = smiles_data.get(comp_id)
            
            # Calculate a score based on name length to prioritize complex molecules
            name_score = len(ligand_name) if ligand_name else 0
            
            potential_ligands.append({
                "comp_id": comp_id,
                "chain": ligand_chain,
                "name": ligand_name,
                "name_score": name_score,
                "smiles": smiles
            })
    
    # Input Protac or molecular glue
    if potential_ligands:
        potential_ligands.sort(key=lambda x: x["name_score"], reverse=True)
        top_ligand = potential_ligands[0]
        
        if format_type == "ccd":
            ligand_entry = {
                "ligand": {
                    "id": top_ligand["chain"],
                    "ccdCodes": [top_ligand["comp_id"]]
                }
            }
            alphafold_input["sequences"].append(ligand_entry)
        else:
            if top_ligand["smiles"]:
                ligand_entry = {
                    "ligand": {
                        "id": top_ligand["chain"],
                        "smiles": top_ligand["smiles"]
                    }
                }
                alphafold_input["sequences"].append(ligand_entry)
    
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
    except Exception as e:
        # Log error
        st.session_state.protein_analysis_status = "error"
        st.session_state.protein_analysis_message = f"Error in protein analysis: {str(e)}"

def main():
    # Set page title and header
    st.set_page_config(page_title="Input File Generator", layout="wide")
    
    # Initialize session state variables if they don't exist
    if 'protein_analysis_status' not in st.session_state:
        st.session_state.protein_analysis_status = "not_started"
    if 'protein_analysis_message' not in st.session_state:
        st.session_state.protein_analysis_message = ""
    
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
            # Start protein analysis in background
            st.session_state.protein_analysis_status = "running"
            
            # Start the background thread for protein analysis
            threading.Thread(target=run_protein_analysis_in_background, args=(pdb_ids,)).start()

            # Create a combined ZIP file for all PDB IDs
            combined_zip_buffer = io.BytesIO()
            with zipfile.ZipFile(combined_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as combined_zip:
                with st.spinner("Fetching data from Protein DataBank..."):
                    # Process the PDB IDs and fetch data
                    results = process_pdb_ids(pdb_ids)
                    
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

                                    # Display protein analysis results if available
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
                                                                st.write(f"**Canonical SMILES:** {smiles}")
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
                info_container.info("Processing data... Your files are ready for download.")
                
                # Reset buffer position
                combined_zip_buffer.seek(0)
                
                # Replace the info message with success message
                info_container.success("Processing complete! Click the button below to download your files.")
                
                # Add back the download button
                st.download_button(
                    label="Download AlphaFold Input Files",
                    data=combined_zip_buffer,
                    file_name=f"alphafold_inputs_full_structure_{time.strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )

    # Display protein analysis status message if available
    if st.session_state.protein_analysis_status != "not_started":
        # Only show completed or error messages, not progress
        if st.session_state.protein_analysis_status == "completed":
            st.success(st.session_state.protein_analysis_message)
        elif st.session_state.protein_analysis_status == "error":
            st.error(st.session_state.protein_analysis_message)

if __name__ == "__main__":
    main()
