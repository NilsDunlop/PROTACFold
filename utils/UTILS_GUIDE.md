# PROTACFold Utilities

This directory contains utility scripts for working with protein structures, molecular data, and AlphaFold results. These tools help with various tasks such as file format conversion, structure comparison, and property calculation.

## Script Overview

| Script | Description | PyMOL Required |
|--------|-------------|----------------|
| [cif_to_pdb_converter.py](#cif_to_pdb_converterpy) | Converts CIF files to PDB format | Yes |
| [superpose.py](#superposepy) | Superposes protein structures and calculates RMSD | No |
| [rmsd_calculator.py](#rmsd_calculatorpy) | Calculates RMSD between protein chains | Yes |
| [molecular_properties.py](#molecular_propertiespy) | Calculates molecular properties for compounds | No |
| [compute_dockq.py](#compute_dockqpy) | Computes DockQ scores for structure comparisons | No |
| [cif_converter.py](#cif_converterpy) | Processes CIF files for compatibility with other tools | No |
| [compare_predictions.py](#compare_predictionspy) | Compares predicted and experimental structures | No |
| [summary_confidences_merger.py](#summary_confidences_mergerpy) | Merges AlphaFold confidence information | No |

## Detailed Descriptions

### cif_to_pdb_converter.py

**Description**: Converts CIF (Crystallographic Information File) files to PDB (Protein Data Bank) format using PyMOL.

**Key Functions**:
- `convert_cif_to_pdb(input_dir)`: Converts all CIF files in a directory to PDB format.
- `main()`: Main function to run the conversion process.

**Usage**:
```
python cif_to_pdb_converter.py
```

**Note**: Must be run from PyMOL since it relies on PyMOL's command interface.

### superpose.py

**Description**: Superposes AlphaFold model structures onto experimental structures and calculates RMSD (Root Mean Square Deviation).

**Key Functions**:
- `superpose_cif(pdb_code, exp_dir, pred_dir, output_dir)`: Superposes structures and calculates RMSD.
- `parse_arguments()`: Parses command-line arguments.
- `main()`: Main function to run the superposition workflow.

**Usage**:
```
python superpose.py -pdb PDB_CODE [PDB_CODE ...]
```

**Dependencies**: ProDy, MDTraj, NumPy

### rmsd_calculator.py

**Description**: Calculates RMSD between protein structures in PyMOL, specifically comparing models generated with different approaches.

**Key Functions**:
- `find_chain_by_sequence(sequence, model_name)`: Finds a chain matching a specific sequence.
- `detect_models()`: Detects original, SMILES, and CCD models loaded in PyMOL.
- `process_protein_chain(protein_type, sequence, original_model, smiles_model, ccd_model)`: Processes protein chains and calculates RMSD.
- `calculate_rmsd()`: Main function to calculate RMSD between structures.

**Usage**: This script should be run within PyMOL:
```
# In PyMOL after loading models
run utils/rmsd_calculator.py
calculate_rmsd
```

**Note**: Must be run from PyMOL as it depends on the PyMOL environment.

### molecular_properties.py

**Description**: Calculates various molecular properties for compounds using RDKit.

**Key Functions**:
- `calculate_molecular_properties(df, smiles_column)`: Calculates properties like MW, LogP, etc.
- `parse_arguments()`: Parses command-line arguments.
- `print_summary(df, property_columns)`: Prints a summary of calculated properties.
- `main()`: Main function to run the property calculation workflow.

**Usage**:
```
python molecular_properties.py --input <input_csv> --output <output_csv> [--smiles_column <column_name>]
```

**Dependencies**: RDKit, Pandas, NumPy

### compute_dockq.py

**Description**: Analyzes protein structure models by computing DockQ scores between AlphaFold models and reference PDB structures.

**Key Functions**:
- `run_dockq(model_path, reference_path, output_json)`: Runs DockQ analysis for a single model.
- `extract_pdb_id(filename)`: Extracts the PDB ID from a filename.
- `process_models(af_models_dir, reference_dir, output_dir)`: Processes multiple models.
- `main()`: Main function to run the DockQ analysis workflow.

**Usage**:
```
python compute_dockq.py
```

**Dependencies**: DockQ command-line tool

### cif_converter.py

**Description**: Processes mmCIF CCD (Chemical Component Dictionary) files to produce valid strings for the userCCD field in AlphaFold.

**Key Functions**:
- `process_atom_loop(lines, header_start, header_end)`: Processes atom coordinates in CIF files.
- `process_cif_file(input_path, output_path)`: Processes a CIF file for compatibility.
- `main()`: Main function to parse command-line arguments and process files.

**Usage**:
```
python cif_converter.py input.cif output.txt
```

### compare_predictions.py

**Description**: Compares predicted protein structures with experimental structures using RMSD of alpha carbon atoms.

**Key Functions**:
- `get_parser(file_path)`: Determines the appropriate parser based on file extension.
- `compare_structures(pred_path, exp_path)`: Compares structures using RMSD.
- `process_files(exp_file, pred_file)`: Processes a pair of structure files.
- `parse_arguments()`: Parses command-line arguments.
- `main()`: Main function to run the comparison workflow.

**Usage**:
```
python compare_predictions.py [-id PDB_ID] [-e EXP_FILE] [-p PRED_FILE]
```

**Dependencies**: Biopython, NumPy

### summary_confidences_merger.py

**Description**: Walks through a directory structure containing AlphaFold3 confidence files and merges them into a single Excel spreadsheet.

**Key Functions**:
- `process_confidence_files(base_dir)`: Processes AlphaFold confidence files.
- `create_excel_report(rows, output_path)`: Creates an Excel report from processed data.
- `main()`: Main function to execute the script.

**Usage**:
```
python summary_confidences_merger.py
```

**Dependencies**: Pandas

## Running Scripts that Require PyMOL

For scripts that require PyMOL (`cif_to_pdb_converter.py` and `rmsd_calculator.py`), you need to run them from within the PyMOL application:

1. Launch PyMOL
2. Use the PyMOL command line to run the script:
   ```
   run /path/to/utils/script_name.py
   ```
3. For `rmsd_calculator.py`, after loading the script, run the command:
   ```
   calculate_rmsd
   ``` 