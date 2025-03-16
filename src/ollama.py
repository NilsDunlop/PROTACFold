import requests
import json
import logging
import time
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("protein_analysis.log"),
        logging.StreamHandler()
    ]
)

# Define debug directory using relative path
# Since ollama.py is in src/ and debug_files is at the project root
DEBUG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "debug_files"))
# Ensure debug directory exists
os.makedirs(DEBUG_DIR, exist_ok=True)

def create_prompt_with_sequences(fasta_text):
    """
    Create the prompt for Ollama with the FASTA text inserted
    
    Args:
        fasta_text (str): Raw FASTA text
        
    Returns:
        str: The complete prompt with sequences inserted
    """
    base_prompt = """You are an expert in targeted protein degradation (TPD), specializing in molecular glues and proteolysis-targeting chimeras (PROTACs). Your task is to analyze protein data and correctly identify two key components: the protein of interest (degradation target) and the E3 ubiquitin ligase component.

# INPUT FORMAT
You will receive a list of proteins with their amino acid sequences between <protein_list> tags.

# TASK CONTEXT
In targeted protein degradation:
- Protein of interest (POI): The disease-relevant target protein we want to degrade
- E3 ubiquitin ligase: The protein that facilitates ubiquitination of the POI, marking it for proteasomal degradation

# IDENTIFICATION GUIDELINES
1. Protein of Interest:
   - Disease-relevant proteins or traditionally undruggable targets
   - Common examples: BRD4/7/9, BRAF, BTK, FAK,  SMARCA2/4, FKBP51, WDR5, KRAS, BCL-family, PBRM1, transcription factors, kinases
   - Often the larger protein in the complex

2. E3 Ligase Components:
   - Primary examples: VHL, CRBN, MDM2, DCAF15, cIAP1, KEAP1, DCAF16, RNF114, DCAF1
   - May appear with associated complex proteins (e.g., DDB1 with CRBN)
   - Function in recruiting the ubiquitination machinery

3. Special Cases:
   - Adaptor/scaffold proteins (e.g., Elongin-B, Elongin-C, Cullin, DDB1) are NOT the core E3 ligase
   - Some E3 ligases have substrate receptor domains (e.g., CRBN, VHL) that are part of larger complexes
   - Neo-substrates (e.g., p53, IKZF1/2/3, CK1α, FKBP12, GSPT1/eRF3a, RBM39, Sal-like protein 4, CD01) are proteins of interest when targeted by molecular glues

# OUTPUT FORMAT
<output>
<protein_of_interest>
[EXACTLY as named in the protein list - if not present, leave empty but keep tags]
</protein_of_interest>

<e3_ubiquitin_ligase>
[EXACTLY as named in the protein list - if not present, leave empty but keep tags]
</e3_ubiquitin_ligase>
</output>

# CRITICAL REQUIREMENTS
- Use ONLY the exact names from the input list
- Provide NO explanations or additional text
- If one component is missing, include its tags but leave them empty
- Include full protein identifiers exactly as shown in the input

# Example valid output:
<output>
<protein_of_interest>
8C13_1|Chain A[auth J]|Some Target Protein|Homo sapiens (9606)
</protein_of_interest>

<e3_ubiquitin_ligase>
8C13_3|Chain C[auth L]|von Hippel-Lindau disease tumor suppressor|Homo sapiens (9606)
</e3_ubiquitin_ligase>
</output>

<protein_list>
"""
    
    # Add the raw FASTA text directly to the prompt
    complete_prompt = base_prompt + fasta_text + "</protein_list>"
    
    return complete_prompt

def query_ollama(prompt, model="deepseek-r1:14b"):
    """
    Send a prompt to Ollama API and get the response
    
    Args:
        prompt (str): The prompt to send to Ollama
        model (str): The model to use, defaults to deepseek-r1:14b
        
    Returns:
        str: The response from Ollama
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 8192,  # Set context window
            "temperature": 0.3,  # Moderate temperature for balanced responses
            "top_p": 0.95,
            "top_k": 40,
            "seed": 42,  # Set a fixed seed for more reproducible results
            "reset": True  # This ensures we start a fresh conversation
        }
    }
    
    logging.info(f"Sending request to Ollama with model: {model}")
    start_time = time.time()
    
    response = requests.post(url, json=payload)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Ollama response received in {elapsed_time:.2f} seconds")
    
    if response.status_code == 200:
        return response.json()["response"]
    else:
        error_msg = f"Error querying Ollama: {response.status_code}, {response.text}"
        logging.error(error_msg)
        raise Exception(error_msg)

def parse_ollama_response(response):
    """
    Parse the Ollama response to extract protein of interest and E3 ligase
    
    Args:
        response (str): The response from Ollama
        
    Returns:
        tuple: (protein_of_interest, e3_ubiquitin_ligase)
    """
    # Extract content between tags
    poi_start = response.find("<protein_of_interest>")
    poi_end = response.find("</protein_of_interest>")
    
    e3_start = response.find("<e3_ubiquitin_ligase>")
    e3_end = response.find("</e3_ubiquitin_ligase>")
    
    protein_of_interest = ""
    e3_ubiquitin_ligase = ""
    
    if poi_start != -1 and poi_end != -1:
        protein_of_interest = response[poi_start + len("<protein_of_interest>"):poi_end].strip()
    
    if e3_start != -1 and e3_end != -1:
        e3_ubiquitin_ligase = response[e3_start + len("<e3_ubiquitin_ligase>"):e3_end].strip()
    
    return protein_of_interest, e3_ubiquitin_ligase

def retrieve_raw_fasta_text(pdb_id: str) -> str:
    """
    Retrieve raw FASTA text for a given PDB ID from RCSB
    
    Args:
        pdb_id: The PDB ID to retrieve sequences for
        
    Returns:
        str: The complete raw FASTA text
    """
    pdb_id = pdb_id.upper()  # ensure PDB id is uppercase
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/display"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
        'Accept': 'text/plain',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Log the raw response for debugging
        fasta_text = response.text
        logging.info(f"Retrieved FASTA text length: {len(fasta_text)} characters")
        
        # Save the raw FASTA response to a debug file
        fasta_debug_file = os.path.join(DEBUG_DIR, f"fasta_raw_{pdb_id}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        with open(fasta_debug_file, 'w') as f:
            f.write(f"=== RAW FASTA RESPONSE FOR {pdb_id} ===\n\n")
            f.write(fasta_text)
        logging.info(f"Saved raw FASTA response to {fasta_debug_file}")
        
        # Check if we have multiple entries by counting '>' characters
        protein_count = fasta_text.count('>')
        logging.info(f"Found {protein_count} protein entries in FASTA text")
        
        if protein_count == 0:
            logging.warning(f"No protein entries found in FASTA response for PDB ID: {pdb_id}")
            return ""
            
        # Also log the first 200 characters to see what we're getting
        preview = fasta_text[:200].replace('\n', '\\n')
        logging.info(f"FASTA text preview: {preview}...")
            
        return fasta_text
    except requests.RequestException as e:
        logging.error(f"Error retrieving FASTA text for PDB ID {pdb_id}: {str(e)}")
        # Try an alternative approach with a custom GET request
        try:
            import urllib.request
            
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                fasta_text = response.read().decode('utf-8')
                logging.info(f"Retrieved FASTA text using urllib length: {len(fasta_text)} characters")
                
                # Save the raw FASTA response from urllib to a debug file
                fasta_debug_file = os.path.join(DEBUG_DIR, f"fasta_raw_urllib_{pdb_id}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
                with open(fasta_debug_file, 'w') as f:
                    f.write(f"=== RAW FASTA RESPONSE (URLLIB) FOR {pdb_id} ===\n\n")
                    f.write(fasta_text)
                logging.info(f"Saved raw FASTA response (urllib) to {fasta_debug_file}")
                
                return fasta_text
        except Exception as e2:
            logging.error(f"Alternative method also failed: {str(e2)}")
            raise Exception(f"Failed to retrieve FASTA data for {pdb_id}: {str(e)}, {str(e2)}")

def identify_proteins(pdb_ids):
    """
    Identify protein of interest and E3 ubiquitin ligase for each PDB ID
    
    Args:
        pdb_ids (list): List of PDB IDs to analyze
        
    Returns:
        dict: Dictionary with PDB IDs as keys and identified proteins information as values
    """
    results = {}
    raw_outputs = {}
    prompts = {}  # Store the prompts for each PDB ID
    
    total_pdb_ids = len(pdb_ids)
    logging.info(f"Starting protein analysis for {total_pdb_ids} PDB IDs: {', '.join(pdb_ids)}")
    
    for idx, pdb_id in enumerate(pdb_ids):
        try:
            logging.info(f"Processing PDB ID {pdb_id} ({idx+1}/{total_pdb_ids})")
            
            # Get raw FASTA text for this PDB ID
            fasta_text = retrieve_raw_fasta_text(pdb_id)
            
            if not fasta_text:
                logging.warning(f"No sequences found for PDB ID: {pdb_id}")
                continue
            
            logging.info(f"Retrieved FASTA text for PDB ID: {pdb_id}")
            
            # Create prompt with sequences
            prompt = create_prompt_with_sequences(fasta_text)
            
            # Save the prompt for later
            prompts[pdb_id] = prompt
            
            # Query Ollama
            logging.info(f"Querying Ollama model for PDB ID: {pdb_id}")
            response = query_ollama(prompt)
            
            # Save raw model output for debugging
            raw_outputs[pdb_id] = response
            
            # Parse response
            protein_of_interest, e3_ubiquitin_ligase = parse_ollama_response(response)
            
            logging.info(f"PDB ID {pdb_id} - Protein of Interest: {protein_of_interest or 'None'}")
            logging.info(f"PDB ID {pdb_id} - E3 Ubiquitin Ligase: {e3_ubiquitin_ligase or 'None'}")
            
            # Add to results dictionary - now store both proteins if available
            results[pdb_id] = {
                "protein_of_interest": protein_of_interest,
                "e3_ubiquitin_ligase": e3_ubiquitin_ligase
            }
            
            if not protein_of_interest and not e3_ubiquitin_ligase:
                logging.warning(f"No proteins identified for PDB ID: {pdb_id}")
                
        except Exception as e:
            error_msg = f"Error processing PDB ID {pdb_id}: {str(e)}"
            logging.error(error_msg)
            print(error_msg)
    
    # Save raw model outputs and prompts to files for debugging
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Save the model outputs
    output_file = os.path.join(DEBUG_DIR, f"model_raw_outputs_{timestamp}.txt")
    with open(output_file, "w") as f:
        for pdb_id, output in raw_outputs.items():
            f.write(f"===== PDB ID: {pdb_id} =====\n\n")
            f.write(output)
            f.write("\n\n")
    
    logging.info(f"Raw model outputs saved to {output_file}")
    
    # Save the prompts
    prompts_file = os.path.join(DEBUG_DIR, f"model_prompts_{timestamp}.txt")
    with open(prompts_file, "w") as f:
        for pdb_id, prompt_text in prompts.items():
            f.write(f"===== PROMPT FOR PDB ID: {pdb_id} =====\n\n")
            f.write(prompt_text)
            f.write("\n\n" + "="*80 + "\n\n")
    
    logging.info(f"Model prompts saved to {prompts_file}")
    
    return results

def save_results_to_file(results, filename="protein_analysis_results.txt"):
    """
    Save analysis results to a text file
    
    Args:
        results (dict): Dictionary with PDB IDs as keys and identified proteins info as values
        filename (str): Name of the file to save results to
    """
    # Use the debug directory for results file if not an absolute path
    if not os.path.isabs(filename):
        filename = os.path.join(DEBUG_DIR, filename)
        
    with open(filename, 'w') as f:
        f.write("PDB ID Analysis Results\n")
        f.write("----------------------\n\n")
        
        for pdb_id, proteins in results.items():
            f.write(f"PDB ID: {pdb_id}\n")
            
            if proteins["protein_of_interest"]:
                f.write(f"Protein of Interest: {proteins['protein_of_interest']}\n")
            
            if proteins["e3_ubiquitin_ligase"]:
                f.write(f"E3 Ubiquitin Ligase: {proteins['e3_ubiquitin_ligase']}\n")
            
            f.write("\n")
    
    print(f"Results saved to {filename}")

def analyze_pdb_proteins(pdb_ids_string):
    """
    Main function to analyze PDB proteins from a comma-separated string of PDB IDs
    
    Args:
        pdb_ids_string (str): Comma-separated string of PDB IDs
        
    Returns:
        dict: Dictionary with PDB IDs as keys and identified proteins information as values
    """
    # Split the string by commas and strip whitespace
    pdb_ids = [pdb_id.strip() for pdb_id in pdb_ids_string.split(',') if pdb_id.strip()]
    
    if not pdb_ids:
        logging.warning("No valid PDB IDs provided")
        print("No valid PDB IDs provided")
        return {}
    
    logging.info(f"Starting analysis for PDB IDs: {', '.join(pdb_ids)}")
    
    # Identify proteins
    detailed_results = identify_proteins(pdb_ids)
    
    # Save results to file
    results_file = os.path.join(DEBUG_DIR, f"protein_analysis_results_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    save_results_to_file(detailed_results, results_file)
    
    # Return the detailed results
    return detailed_results 