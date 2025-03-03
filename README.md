# PROTACFold

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

![PROTACFold Workflow](docs/images/PROTACFold.png)

## Overview

PROTACFold is a comprehensive toolkit for analyzing and predicting Proteolysis Targeting Chimera (PROTAC) structures using AlphaFold 3 and related tools. This project focuses on understanding PROTAC-mediated ternary complexes between target proteins and E3 ligases, enabling better insights into protein degradation mechanisms.

## Features

- **AlphaFold 3 Integration**: Streamlined setup and usage of AlphaFold 3 for PROTAC ternary complex prediction
- **Structure Analysis Tools**: Utilities for calculating RMSD, DockQ scores, and TM-scores for protein structure comparison
- **Molecular Property Analysis**: Calculate and analyze important molecular properties of PROTACs using RDKit
- **Visualization Workflows**: Jupyter notebooks for data analysis and visualization of prediction results
- **Data Integration**: Support for PROTAC databases including ProtacDB and ProtacPedia
- **Format Conversion**: Tools for converting between different molecular structure formats (PDB, CIF)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for AlphaFold 3)
- Docker (recommended for AlphaFold 3 setup)

### Using Docker (Recommended)

Detailed instructions for setting up AlphaFold 3 using Docker can be found in the [installation guide](docs/installation_docker.md).

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/NilsDunlop/PROTACFold.git
cd PROTACFold
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Directory Structure

- `data/`: Contains datasets and analysis results
  - `protacdb/`: PROTAC database files
  - `protacpedia/`: ProtacPedia database files
  - `af3_input/`: Input files for AlphaFold 3
  - `af3_results/`: Consolidated results from AlphaFold 3 predictions
- `utils/`: Utility scripts for structure analysis and property calculation
- `notebooks/`: Jupyter notebooks for analysis and visualization
- `docs/`: Documentation including installation guides

## Usage

### PROTAC Structure Prediction

Use AlphaFold 3 to predict the structure of PROTAC-mediated ternary complexes:

1. Prepare your input JSON files similar to the examples in the `data/af3_input/` directory
2. Run AlphaFold 3 using Docker (see installation guide)
3. Analyze results using the provided utility scripts (if needed)

### Analyzing Prediction Results

```bash
# Calculate RMSD between predicted and reference structures
python utils/rmsd_calculator.py --pred path/to/prediction.pdb --ref path/to/reference.pdb

# Calculate DockQ score
python utils/compute_dockq.py

# Calculate molecular properties from SMILES
python utils/molecular_properties.py --input data/smiles_file.csv --output results.csv
```

### Visualization and Analysis

Explore the Jupyter notebooks in the `notebooks/` directory for examples of how to analyze and visualize prediction results:

```bash
jupyter notebook notebooks/af3_analysis.ipynb
```

## Tools

### Protein Structure Prediction
- **[AlphaFold 3](https://github.com/google-deepmind/alphafold3)** - DeepMind's state-of-the-art protein structure prediction model, used for generating PROTAC-mediated ternary complex structures

### Structure Analysis and Comparison
- **[DockQ](https://github.com/bjornwallner/DockQ)** - A quality measure for protein-protein docking models, used for evaluating prediction accuracy

### Visualization
- **[PyMOL](https://www.pymol.org/)** - Molecular visualization system for rendering and analyzing 3D molecular structures

### Chemoinformatics
- **[RDKit](https://www.rdkit.org/)** - Open-source chemoinformatics toolkit for calculating molecular properties and manipulating chemical structures

## Data Sources

This project integrates data from:
- [Protein Data Bank](https://www.rcsb.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The AlphaFold team at Google DeepMind
- RDKit and other open-source tools used in this project
- PyMOL and DockQ for their respective tools

## Citation

If you use PROTACFold in your research, please cite this repository.