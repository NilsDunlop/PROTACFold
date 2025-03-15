import streamlit as st
import json
from api import process_pdb_ids, retrieve_fasta_sequence, extract_ligand_ccd_from_pdb, extract_comp_ids

def main():
    # Set page title and header
    st.set_page_config(page_title="Input File Generator", layout="wide")
    
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
                                        # Use polymer description as the expander title
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
                                                        st.write(f"**Amino Acid Sequence:**")
                                                        st.code(sequence, language=None)
                                                else:
                                                    # Fallback
                                                    if entity_idx < len(fasta_sequences):
                                                        header, sequence = list(fasta_sequences.items())[entity_idx]
                                                        st.write(f"**Amino Acid Sequence:**")
                                                        st.code(sequence, language=None)
                                
                                # Display ligand entities
                                if ligand_data:
                                    st.subheader("Ligand Entities")
                                    
                                    for ligand_id, ligand_info in ligand_data.items():
                                        comp_id = ligand_info.get("pdbx_entity_nonpoly", {}).get("comp_id", "Unknown")
                                        ligand_name = ligand_info.get("pdbx_entity_nonpoly", {}).get("name", comp_id)
                                        ligand_title = f"{comp_id} - {ligand_name}"
                                        
                                        with st.expander(ligand_title):
                                            # Display ligand details
                                            st.write(f"**Component ID:** {comp_id}")
                                            st.write(f"**Name:** {ligand_name}")
                                
                                # Option to download the raw JSON data
                                st.download_button(
                                    label="Download Raw Data (JSON)",
                                    data=json.dumps(data, indent=2),
                                    file_name=f"{pdb_id}_data.json",
                                    mime="application/json"
                                )
                            
                            except Exception as e:
                                st.error(f"Error processing data for {pdb_id}: {str(e)}")
                                st.json(data)

if __name__ == "__main__":
    main()