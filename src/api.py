import requests
import json
from typing import Dict
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