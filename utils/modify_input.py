import argparse
import json
import os
import yaml
from pathlib import Path

def update_json_with_seed(json_path, new_seed, output_folder=None):
    """Update modelSeeds and name in JSON with a new seed."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Update seed
    data['modelSeeds'] = [new_seed]  
    if 'name' in data:
        base_name = data['name']
        if '_' in base_name and base_name.split('_')[-1].isdigit():
            base_name = '_'.join(base_name.split('_')[:-1])
        data['name'] = f"{base_name}_{new_seed}"
    
    # Generate new filename
    file_stem = json_path.stem
    if '_' in file_stem and file_stem.split('_')[-1].isdigit():
        file_stem = '_'.join(file_stem.split('_')[:-1])
    new_filename = f"{file_stem}_{new_seed}.json"
    
    # Define output path and ensure the directory exists
    output_path = Path(output_folder) / new_filename if output_folder else json_path.parent / new_filename
    os.makedirs(output_folder, exist_ok=True) if output_folder else None

    # Write the modified JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_path

def convert_to_boltz1_yaml(json_path, output_folder=None):
    """Convert AF3 JSON file to Boltz-1 YAML format."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Initialize YAML structure
    boltz_data = {"version": 1, "sequences": []}  
    
    if 'sequences' in data:
        for sequence in data['sequences']:
            # Process protein sequences
            if 'protein' in sequence:
                chain_id = sequence['protein'].get('id')
                seq = sequence['protein'].get('sequence', '')
                id_list = chain_id if isinstance(chain_id, list) else [chain_id] if chain_id else []
                boltz_data["sequences"].append({"protein": {"id": id_list, "sequence": seq}})
                
            # Process ligand sequences
            elif 'ligand' in sequence:
                ligand_data = sequence['ligand']
                ligand_id = ligand_data.get('id')
                id_list = ligand_id if isinstance(ligand_id, list) else [ligand_id] if ligand_id else []
                ligand_entry = {"ligand": {"id": id_list}}

                # Prioritize identifiers (CCD or SMILES)
                if 'ccdCodes' in ligand_data and ligand_data['ccdCodes']:
                    ligand_entry["ligand"]["ccd"] = ligand_data['ccdCodes'][0]
                elif 'smiles' in ligand_data:
                    ligand_entry["ligand"]["smiles"] = ligand_data['smiles']
                else:  # Fallback to filename for identifiers
                    file_stem = json_path.stem
                    if 'ccd' in file_stem.lower():
                        parts = file_stem.split('_')
                        league_code = parts[1] if len(parts) > 1 and not parts[1].isdigit() else "UNK"
                        ligand_entry["ligand"]["ccd"] = league_code
                    else:
                        ligand_entry["ligand"]["smiles"] = ''

                boltz_data["sequences"].append(ligand_entry)

    # Define output path for YAML file
    yaml_filename = f"{json_path.stem}.yaml"
    output_path = Path(output_folder) / yaml_filename if output_folder else json_path.parent / yaml_filename
    os.makedirs(output_folder, exist_ok=True) if output_folder else None

    # Write the YAML file with formatted content
    with open(output_path, 'w') as f:
        yaml_str = yaml.dump(boltz_data, default_flow_style=False, sort_keys=False)
        lines = yaml_str.split('\n')
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'id:' in line and i + 1 < len(lines) and lines[i + 1].strip().startswith('- '):
                id_values = []
                i += 1
                while i < len(lines) and lines[i].strip().startswith('- '):
                    id_values.append(lines[i].strip()[2:])
                    i += 1
                result.append(f"    id: [{', '.join(id_values)}]")
            elif 'ccd:' in line and "'" in line:
                ccd_value = line.split("'")[1]
                result.append(f"    ccd: {ccd_value}")
                i += 1
            elif 'smiles:' in line and not "'" in line:
                smiles_value = line.split("smiles:")[1].strip()
                result.append(f"    smiles: '{smiles_value}'")
                i += 1
            else:
                result.append(line)
                i += 1
        
        f.write('\n'.join(result))
    
    return output_path

def main():
    """Main function to process command-line inputs."""
    parser = argparse.ArgumentParser(description='Modify AlphaFold3 input files for different purposes.')
    parser.add_argument('--input', required=True, help='Folder containing AlphaFold3 JSON files')
    parser.add_argument('--output', help='Optional folder to save modified files')
    parser.add_argument('--seed', type=int, help='New seed value to use for model seeds')
    parser.add_argument('--yaml', action='store_true', help='Convert JSON files to Boltz-1 YAML format')

    args = parser.parse_args()
    
    input_folder = Path(args.input)
    output_folder = args.output
    
    if not input_folder.exists() or not input_folder.is_dir():
        return
    
    # Create output folder if needed
    if output_folder:
        os.makedirs(output_folder, exist_ok=True) 
    
    # Gather JSON files from the input folder
    json_files = list(input_folder.glob('*.json'))
    if not json_files:
        return
    
    # Update seeds in JSON files if specified
    if args.seed is not None:
        for json_file in json_files:
            update_json_with_seed(json_file, args.seed, output_folder)
        
        if not args.yaml:
            return

        # Refresh the list of JSON files after seeds update
        if output_folder:
            json_files = list(Path(output_folder).glob('*.json'))
    
    # Convert JSON to YAML if specified
    if args.yaml:
        for json_file in json_files:
            convert_to_boltz1_yaml(json_file, output_folder)

if __name__ == "__main__":
    main()