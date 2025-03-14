import streamlit as st
import json
from api import process_pdb_ids

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
                                    for i, entity in enumerate(entry_data["polymer_entities"]):
                                        with st.expander(f"Entity {i+1}"):
                                            if "rcsb_polymer_entity" in entity and entity["rcsb_polymer_entity"]:
                                                st.write(f"**Description:** {entity['rcsb_polymer_entity']['pdbx_description']}")
                                            
                                            if "entity_poly" in entity and entity["entity_poly"]:
                                                st.write(f"**Type:** {entity['entity_poly']['type']}")
                                                st.write(f"**Polymer Type:** {entity['entity_poly']['rcsb_entity_polymer_type']}")
                                            
                                            if "rcsb_entity_source_organism" in entity and entity["rcsb_entity_source_organism"]:
                                                organisms = entity["rcsb_entity_source_organism"]
                                                if organisms:
                                                    st.write(f"**Source Organism:** {organisms[0]['scientific_name']}")
                                
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