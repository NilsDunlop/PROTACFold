# PROTACFold Utilities

This directory contains utility scripts for preparing inputs, evaluating models, and processing data for PROTAC structure prediction.

## Script Overview

| Script                                               | Description                                                                                                | PyMOL Required |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------- |
| [`pdb_assembly_downloader.py`](#pdb_assembly_downloaderpy) | Downloads biological assembly files from the PDB.                                                          | No             |
| [`cif_converter.py`](#cif_converterpy)               | Processes CIF files for compatibility with AlphaFold's `userCCD` field.                                    | No             |
| [`modify_input.py`](#modify_inputpy)                 | Modifies AlphaFold 3 JSON inputs (e.g., updates seeds) and converts them to Boltz-1 YAML format.           | No             |
| [`evaluation.py`](#evaluationpy)                     | Evaluates prediction models by calculating RMSD, DockQ, and other metrics against reference structures.    | Yes            |
| [`model_adapters.py`](#model_adapterspy)             | Provides helper classes to handle outputs from different prediction models (AlphaFold 3, Boltz-1).           | No             |

## Detailed Descriptions

### pdb_assembly_downloader.py

**Description**: Downloads biological assembly CIF files from the RCSB PDB for given PDB IDs. It can download a single file or batch download multiple files from a list or a default list in the script. The files are downloaded as `.cif.gz` and decompressed by default.

**Key Functions**:
- `download_pdb_assembly(pdb_id, output_dir, assembly_id, decompress)`: Downloads a single PDB assembly.
- `batch_download(pdb_ids, output_dir, ...)`: Downloads multiple PDB assemblies.
- `main()`: Parses command-line arguments for single and batch modes.

**Usage**:
For a single PDB ID:
```bash
python utils/pdb_assembly_downloader.py single 9DLW -o path/to/output
```
For batch downloading from a file:
```bash
python utils/pdb_assembly_downloader.py batch --input-file pdb_ids.txt --output-dir path/to/output
```

**Dependencies**: `requests`

### cif_converter.py

**Description**: Processes mmCIF CCD (Chemical Component Dictionary) files to produce valid strings for the userCCD field in AlphaFold.

**Key Functions**:
- `process_atom_loop(lines, header_start, header_end)`: Processes atom coordinates in CIF files.
- `process_cif_file(input_path, output_path)`: Processes a CIF file for compatibility.
- `main()`: Main function to parse command-line arguments and process files.

**Usage**:
```
python utils/cif_converter.py input.cif output.txt
```

### modify_input.py

**Description**: A script to modify AlphaFold 3 JSON input files. It can update the random seed used for predictions and convert the JSON files to the YAML format required by Boltz-1.

**Key Functions**:
- `update_json_with_seed(json_path, new_seed, output_folder)`: Updates the `modelSeeds` and `name` in a JSON file.
- `convert_to_boltz1_yaml(json_path, output_folder)`: Converts an AlphaFold 3 JSON file to a Boltz-1 compatible YAML file.
- `main()`: Main function to process arguments for updating seed and/or converting to YAML.

**Usage**:
To update a seed:
```bash
python utils/modify_input.py --input path/to/json --seed 42 --output path/to/output
```
To convert to YAML:
```bash
python utils/modify_input.py --input path/to/json --yaml --output path/to/output
```

**Dependencies**: `PyYAML`

### evaluation.py

**Description**: This script automates the comprehensive evaluation of structure predictions from models like AlphaFold 3 and Boltz-1. It processes prediction outputs, compares them to reference structures, and calculates a wide range of metrics:
- Overall RMSD (C-alpha atoms)
- Component-wise RMSD for Protein of Interest (POI) and E3 Ligase
- PROTAC/Ligand RMSD
- DockQ, iRMSD, and LRMSD scores for interface quality
- Model confidence scores (pTM, ipTM, etc.)
- Physicochemical properties of ligands from SMILES strings.

The script is highly automated, using analysis files to identify protein chains and a sophisticated method to identify the primary ligand. It compiles all results into a single CSV file.

**Key Functions**:
- `process_pdb_folder(...)`: Processes all prediction results for a given PDB ID.
- `compute_rmsd_with_pymol(...)`: Computes C-alpha RMSD between two structures.
- `compute_ligand_rmsd_with_pymol(...)`: Computes RMSD for the ligand.
- `calculate_component_rmsd(...)`: Calculates RMSD for specific components like POI and E3.
- `run_dockq(...)`: Executes the DockQ tool.
- `calculate_molecular_properties_from_smiles(...)`: Calculates molecular properties using RDKit.

**Usage**:
The script can be run on directories containing prediction outputs, categorized by molecule type (e.g., PROTAC, glue).
```bash
# Analyze AlphaFold 3 predictions
python utils/evaluation.py --protac path/to/protac_predictions --model_type AlphaFold3 --output results.csv

# Analyze Boltz-1 predictions
python utils/evaluation.py --glue path/to/glue_predictions --model_type Boltz1 --output results.csv
```

**Note**: This script requires a local installation of PyMOL for structural alignments and the DockQ command-line tool.

**Dependencies**: `PyMOL`, `RDKit`, `Pandas`, `NumPy`, `requests`, `DockQ` (command-line tool).

### model_adapters.py

**Description**: This module provides a set of adapter classes to create a unified interface for handling outputs from different structure prediction models, namely AlphaFold 3 and Boltz-1. It abstracts away the differences in file naming conventions and confidence score formats, allowing evaluation scripts to process outputs from either model seamlessly. This script is not intended to be run directly but is used as a helper module by `evaluation.py`.

**Key Components**:
- `ModelAdapter`: An abstract base class defining the adapter interface.
- `AlphaFoldAdapter`: An adapter specifically for handling AlphaFold 3 output files.
- `Boltz1Adapter`: An adapter specifically for handling Boltz-1 output files.
- `get_model_adapter(model_type)`: A factory function that returns the appropriate adapter based on the model name.

**Usage**:
This module is imported and used by other scripts. For example, in `evaluation.py`:
```python
from model_adapters import get_model_adapter

# ...
model_adapter = get_model_adapter("AlphaFold3")
model_path, json_path = model_adapter.get_model_paths(...)
``` 