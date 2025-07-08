from abc import ABC, abstractmethod
import os
import json
import logging

class ModelAdapter(ABC):
    """Base adapter for different structure prediction tools"""
    
    @abstractmethod
    def get_model_paths(self, folder_path, folder_name, pdb_id, model_type, seed, is_ternary=False):
        """Get the model file path and JSON confidence file path"""
        pass
    
    @abstractmethod
    def extract_confidence_values(self, json_file):
        """Extract confidence metrics from a JSON file"""
        pass

class AlphaFoldAdapter(ModelAdapter):
    """Adapter for AlphaFold outputs"""
    
    def get_model_paths(self, folder_path, folder_name, pdb_id, model_type, seed, is_ternary=False):
        # Handle both naming conventions
        if is_ternary:
            name_prefix = f"{pdb_id.lower()}_ternary_{model_type}"
        else:
            name_prefix = f"{pdb_id.lower()}_{model_type}"
        
        # Try both naming conventions for model
        model_path1 = os.path.join(folder_path, folder_name, f"{folder_name}_model.cif")
        model_path2 = os.path.join(folder_path, folder_name, f"{name_prefix}_model.cif")
        
        # Check which model file exists
        if os.path.exists(model_path1):
            model_path = model_path1
        elif os.path.exists(model_path2):
            model_path = model_path2
        else:
            model_path = None
        
        # Try both naming conventions for JSON
        json_path1 = os.path.join(folder_path, folder_name, f"{folder_name}_summary_confidences.json")
        json_path2 = os.path.join(folder_path, folder_name, f"{name_prefix}_summary_confidences.json")
        
        if os.path.exists(json_path1):
            json_path = json_path1
        elif os.path.exists(json_path2):
            json_path = json_path2
        else:
            json_path = None
        
        return model_path, json_path
    
    def extract_confidence_values(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        fraction_disordered = data.get('fraction_disordered', None)
        has_clash = data.get('has_clash', None)
        iptm = data.get('iptm', None)
        ptm = data.get('ptm', None)
        ranking_score = data.get('ranking_score', None)
        
        return fraction_disordered, has_clash, iptm, ptm, ranking_score

class Boltz1Adapter(ModelAdapter):
    """Adapter for Boltz-1 structure prediction model outputs"""
    
    def get_model_paths(self, folder_path, folder_name, pdb_id, model_type, seed, is_ternary=False):
        """
        Locate Boltz-1 model and confidence files using a robust search strategy.
        
        Handles various folder structures and naming conventions that may be present
        in Boltz-1 output directories, including ternary complexes.
        """
        # If folder_name is provided, look in the specific subfolder first
        if folder_name:
            subfolder_path = os.path.join(folder_path, folder_name)
            if os.path.isdir(subfolder_path):
                # Try both naming patterns in the subfolder
                patterns_to_try = [
                    (f"{pdb_id}_ternary_{model_type}_model_0.cif", f"confidence_{pdb_id}_ternary_{model_type}_model_0.json"),
                    (f"{pdb_id.upper()}_ternary_{model_type}_model_0.cif", f"confidence_{pdb_id.upper()}_ternary_{model_type}_model_0.json"),
                    (f"{pdb_id}_{model_type}_model_0.cif", f"confidence_{pdb_id}_{model_type}_model_0.json"),
                    (f"{pdb_id.upper()}_{model_type}_model_0.cif", f"confidence_{pdb_id.upper()}_{model_type}_model_0.json")
                ]
                
                for model_filename, json_filename in patterns_to_try:
                    model_path = os.path.join(subfolder_path, model_filename)
                    json_path = os.path.join(subfolder_path, json_filename)
                    
                    if os.path.exists(model_path) and os.path.exists(json_path):
                        logging.debug(f"Found Boltz-1 files in subfolder {folder_name}: {model_filename}")
                        return model_path, json_path
                
                # If exact patterns don't work, search for any model files in the subfolder
                try:
                    subfolder_files = os.listdir(subfolder_path)
                    model_files = [f for f in subfolder_files if f.endswith('_model_0.cif')]
                    json_files = [f for f in subfolder_files if f.startswith('confidence_') and f.endswith('.json')]
                    
                    # Try to match model and json files
                    for model_file in model_files:
                        # Extract the base name to find matching json
                        if model_file.endswith('_model_0.cif'):
                            base_name = model_file[:-10]  # Remove '_model_0.cif'
                            expected_json = f"confidence_{base_name}_model_0.json"
                            
                            if expected_json in json_files:
                                model_path = os.path.join(subfolder_path, model_file)
                                json_path = os.path.join(subfolder_path, expected_json)
                                logging.debug(f"Found Boltz-1 files via search in {folder_name}: {model_file}")
                                return model_path, json_path
                except Exception as e:
                    logging.debug(f"Error searching subfolder {folder_name}: {e}")
        
        # Fallback: Try direct paths in the main folder
        standard_model_path = os.path.join(folder_path, f"{pdb_id}_{model_type}_model_0.cif")
        standard_json_path = os.path.join(folder_path, f"confidence_{pdb_id}_{model_type}_model_0.json")
        
        ternary_model_path = os.path.join(folder_path, f"{pdb_id}_ternary_{model_type}_model_0.cif")
        ternary_json_path = os.path.join(folder_path, f"confidence_{pdb_id}_ternary_{model_type}_model_0.json")
        
        # Check if files exist at either direct path pattern
        if os.path.exists(standard_model_path) and os.path.exists(standard_json_path):
            return standard_model_path, standard_json_path
            
        if os.path.exists(ternary_model_path) and os.path.exists(ternary_json_path):
            return ternary_model_path, ternary_json_path
        
        # Legacy fallback: Look through subdirectories to find appropriate files
        try:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if not os.path.isdir(item_path):
                    continue
                    
                # Determine if this subdirectory represents a ternary complex
                is_subdir_ternary = "ternary" in item.lower()
                
                # Construct the file paths based on detected ternary status
                if is_subdir_ternary:
                    subdir_prefix = f"{pdb_id}_ternary_{model_type}"
                else:
                    subdir_prefix = f"{pdb_id}_{model_type}"
                    
                subdir_model_path = os.path.join(item_path, f"{subdir_prefix}_model_0.cif")
                subdir_json_path = os.path.join(item_path, f"confidence_{subdir_prefix}_model_0.json")
                
                if os.path.exists(subdir_model_path) and os.path.exists(subdir_json_path):
                    logging.debug(f"Found Boltz-1 files in {model_type} subdirectory: {item}" + 
                                 (" (ternary)" if is_subdir_ternary else ""))
                    return subdir_model_path, subdir_json_path
        except Exception as e:
            logging.debug(f"Error in legacy search: {e}")
        
        # No files were found
        logging.debug(f"No Boltz-1 files found for {pdb_id} with model type {model_type}")
        return None, None
    
    def extract_confidence_values(self, json_file):
        """
        Parse Boltz-1 confidence metrics from JSON output file.
        
        Returns standardized confidence metrics, with None values for metrics
        that are specific to AlphaFold but not present in Boltz-1 output.
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Boltz-1 uses a different confidence schema than AlphaFold
        # These fields are AlphaFold-specific and not present in Boltz-1
        fraction_disordered = None
        has_clash = None
        
        # Extract standard confidence metrics that are present in Boltz-1 output
        iptm = data.get('iptm', None)
        ptm = data.get('ptm', None)
        confidence_score = data.get('confidence_score', None)
        
        return fraction_disordered, has_clash, iptm, ptm, confidence_score

def get_model_adapter(model_type="alphafold"):
    """Factory function to get the appropriate adapter"""
    if model_type.lower() in ["boltz1", "boltz-1", "boltz"]:
        return Boltz1Adapter()
    else:
        return AlphaFoldAdapter()