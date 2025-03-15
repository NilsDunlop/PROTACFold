import requests
import json
from typing import Dict, List
from io import StringIO
from Bio import SeqIO

def fetch_pdb_data(pdb_id):
    """
    Fetch data from the Protein Data Bank GraphQL API for a given PDB ID
    """
    url = "https://data.rcsb.org/graphql"
    
    # The GraphQL query
    query = """
    query structure($id: String!) {
      entry(entry_id: $id) {
        rcsb_id
        entry {
          id
        }
        struct {
          title
        }
        exptl {
          method
        }
        rcsb_accession_info {
          deposit_date
          initial_release_date
        }
        rcsb_primary_citation {
          id
          pdbx_database_id_DOI
        }
        polymer_entities {
          rcsb_polymer_entity {
            pdbx_description
          }
          entity_poly {
            type
            rcsb_entity_polymer_type
          }
          rcsb_entity_source_organism {
            scientific_name
          }
        }
      }
    }
    """
    
    # Variables for the query
    variables = {"id": pdb_id}
    
    # Make the request
    response = requests.post(
        url,
        json={"query": query, "variables": variables}
    )
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch data: {response.status_code}"}

def process_pdb_ids(pdb_ids_string):
    """
    Process a comma-separated string of PDB IDs and fetch data for each
    """
    # Split the string by commas and strip whitespace
    pdb_ids = [pdb_id.strip() for pdb_id in pdb_ids_string.split(',')]
    
    # Fetch data for each PDB ID
    results = {}
    for pdb_id in pdb_ids:
        if pdb_id:  # Skip empty strings
            results[pdb_id] = fetch_pdb_data(pdb_id)
    
    return results

def retrieve_fasta_sequence(pdb_id: str) -> Dict[str, str]:
    """
    Retrieve FASTA sequences for a given PDB ID from RCSB
    
    Args:
        pdb_id: The PDB ID to retrieve sequences for
        
    Returns:
        A dictionary mapping sequence headers to sequences
    """
    pdb_id = pdb_id.upper()  # ensure PDB id is uppercase
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/display"
    response = requests.get(url)
    response.raise_for_status()
    fasta_text = response.text
    sequences = {}
    fasta_io = StringIO(fasta_text)
    
    for record in SeqIO.parse(fasta_io, "fasta"):
        header_parts = record.description.split("|")
        if len(header_parts) >= 3:
            # Construct header as Protein name|Chains
            custom_header = f"{header_parts[0].strip()}|{header_parts[2].strip()}|{header_parts[1].strip()}"
        else:
            custom_header = record.id

        sequences[custom_header] = str(record.seq)
    
    return sequences

def get_nonpolymer_entity_ids(pdb_id: str) -> List[str]:
    """
    Retrieve the list of non-polymer entity indices from the entry's container identifiers.
    
    Args:
        pdb_id (str): The PDB identifier (e.g., "6TD3").
        
    Returns:
        list: A list of non-polymer entity indices as strings.
    """
    pdb_id = pdb_id.upper()
    entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(entry_url)
    response.raise_for_status()
    entry_data = response.json()
    return entry_data.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids", [])

def get_ligand_ccd_info_rest(pdb_id: str, entity_id: str) -> Dict:
    """
    Retrieve the complete CCD information for a given ligand (non-polymer entity)
    using the REST API.
    
    Args:
        pdb_id (str): The PDB identifier.
        entity_id (str): The internal non-polymer entity index (e.g., "4").
        
    Returns:
        dict: A dictionary with the ligand's CCD details.
    """
    pdb_id = pdb_id.upper()
    url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{entity_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def extract_ligand_ccd_from_pdb(pdb_id: str) -> Dict:
    """
    Extract the CCD (ligand) information from a given PDB entry.
    
    It first gets the non-polymer entity indices from the entry and then, for each one,
    queries the nonpolymer_entity REST endpoint to get the CCD details.
    
    Args:
        pdb_id (str): The PDB identifier (e.g., "6TD3").
    
    Returns:
        dict: A dictionary mapping each ligand's full ID (e.g., "6TD3_4")
              to its detailed CCD information.
    """
    pdb_id = pdb_id.upper()
    ligand_info = {}
    indices = get_nonpolymer_entity_ids(pdb_id)
    if not indices:
        return {}
    
    for entity in indices:
        full_ligand_id = f"{pdb_id}_{entity}"
        try:
            ccd_info = get_ligand_ccd_info_rest(pdb_id, entity)
            ligand_info[full_ligand_id] = ccd_info
        except requests.HTTPError as e:
            # Could log the error here instead of printing
            pass
    return ligand_info

def extract_comp_ids(ligand_ccd: Dict) -> List[str]:
    """
    Extract the comp_id (CCD code) for each ligand from the full ligand CCD info.
    
    Args:
        ligand_ccd (dict): A dictionary of ligand CCD details.
        
    Returns:
        list: A list of unique comp_ids.
    """
    comp_ids = []
    for key, ligand in ligand_ccd.items():
        comp_id = ligand.get("pdbx_entity_nonpoly", {}).get("comp_id")
        if comp_id and comp_id not in comp_ids:
            comp_ids.append(comp_id)
    return comp_ids

def fetch_ligand_smiles(comp_id):
    """
    Fetch SMILES and other chemical data for a ligand using its component ID
    """
    url = "https://data.rcsb.org/graphql"
    
    # The GraphQL query
    query = """
    query molecule($id: String!) {
        chem_comp(comp_id:$id){
            chem_comp {
                id
                name
                formula
                pdbx_formal_charge
                formula_weight
                type
            }
            rcsb_chem_comp_descriptor {
                InChI
                InChIKey
                SMILES
                SMILES_stereo
            }
        }
    }
    """
    
    # Variables for the query
    variables = {"id": comp_id}
    
    # Make the request
    response = requests.post(
        url,
        json={"query": query, "variables": variables}
    )
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch chemical data: {response.status_code}"}