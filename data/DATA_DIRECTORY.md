#  Data Directory Structure

## Main Data Directory (`data/`)
Contains datasets and analysis results for the PROTACFold project.

## Subdirectories:

### 1. `af3_input/`
Contains input files for AlphaFold 3 predictions in two formats:
- **SMILES format** (e.g., `7KHH_smiles.json`): ~45 JSON files containing SMILES string representations of molecules
- **CCD format** (e.g., `7KHH_ccd.json`): ~45 JSON files containing Chemical Component Dictionary format data
- Files are named with PDB IDs followed by the format indicator

### 2. `af3_results/`
Contains consolidated results from AlphaFold 3 predictions:
- `af3_data.csv` (6.8KB): Main results data file
- `af3_poi_e3.csv` (2.2KB): Data specific to POI-E3 interactions
- `molecular_descriptors.csv` (12KB): Contains molecular property data for the PROTACs

### 3. `plots/`
Contains visualization outputs:
- Multiple PNG files showing results from the PROTACFold project

### 4. `hal_04732948/`
Contains data from [Pereira et al., 2024](https://www.biorxiv.org/content/10.1101/2024.03.19.585735v2) for comparison purposes:
- `hal_data.csv` (1.8KB): Main data file from the paper
- Two subdirectories with specific AlphaFold 3 results:

  #### 4.1 `AF3_DIMERS/`
  - Contains ~28 JSON files with docking quality scores (`*_dockq.json`)
  - Each file corresponds to a different PDB structure
  - Files are consistently sized (1.1KB) and follow the naming pattern `[PDB_ID]_dockq.json`

  #### 4.2 `AF3_CONTEXT/`
  - Contains ~28 JSON files with context-specific docking quality scores
  - Files vary in size (1KB to 4.1KB) and follow the same naming pattern as the DIMERS folder
  - Provides additional context information for the same PDB structures


