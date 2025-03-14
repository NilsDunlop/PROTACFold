import requests
import json

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