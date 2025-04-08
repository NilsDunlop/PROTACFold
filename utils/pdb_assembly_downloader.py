#!/usr/bin/env python3

import os
import sys
import argparse
import requests
import time
from pathlib import Path
import gzip
import shutil


def download_pdb_assembly(pdb_id, output_dir=None, assembly_id=1, decompress=True):
    """
    Download the biological assembly CIF file for a given PDB ID.
    
    Parameters:
    -----------
    pdb_id : str
        The PDB ID of the structure (e.g., '9DLW')
    output_dir : str, optional
        Directory to save the downloaded file. If None, uses current directory.
    assembly_id : int, optional
        The assembly ID to download. Default is 1.
    decompress : bool, optional
        Whether to decompress the .gz file. Default is True.
        
    Returns:
    --------
    str
        Path to the downloaded file
    """
    # Convert PDB ID to uppercase
    pdb_id = pdb_id.upper()
    
    # Create the URL for the assembly file
    url = f"https://files.rcsb.org/download/{pdb_id}-assembly{assembly_id}.cif.gz"
    
    # Set up the output directory
    if output_dir is None:
        output_dir = os.getcwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the output filename - without the "-assembly" part
    gz_file_path = output_dir / f"{pdb_id}.cif.gz"
    
    # Download the file
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        print(f"Error: Could not download {url}")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        sys.exit(1)
    
    # Save the downloaded file
    with open(gz_file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded to {gz_file_path}")
    
    # Decompress the file if requested
    if decompress:
        cif_file_path = output_dir / f"{pdb_id}.cif"
        
        with gzip.open(gz_file_path, 'rb') as f_in:
            with open(cif_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"Decompressed to {cif_file_path}")
        return str(cif_file_path)
    
    return str(gz_file_path)


def batch_download(pdb_ids, output_dir, assembly_id=1, decompress=True, delay=0.5):
    """
    Download multiple PDB assembly files.
    
    Parameters:
    -----------
    pdb_ids : list
        List of PDB IDs to download
    output_dir : str
        Directory to save the downloaded files
    assembly_id : int, optional
        The assembly ID to download. Default is 1.
    decompress : bool, optional
        Whether to decompress the .gz files. Default is True.
    delay : float, optional
        Delay between downloads in seconds. Default is 0.5.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download each assembly
    successful = 0
    failed = []
    
    print(f"Starting download of {len(pdb_ids)} PDB assemblies...")
    
    for i, pdb_id in enumerate(pdb_ids, 1):
        print(f"[{i}/{len(pdb_ids)}] Processing {pdb_id}...")
        try:
            # Download the assembly file
            file_path = download_pdb_assembly(
                pdb_id=pdb_id,
                output_dir=output_dir,
                assembly_id=assembly_id,
                decompress=decompress
            )
            successful += 1
            print(f"Successfully downloaded and saved to {file_path}")
            
            # Add a small delay to avoid overwhelming the server
            if i < len(pdb_ids):
                time.sleep(delay)
                
        except Exception as e:
            print(f"Error downloading {pdb_id}: {str(e)}")
            failed.append(pdb_id)
    
    # Print summary
    print("\nDownload Summary:")
    print(f"Total PDB IDs: {len(pdb_ids)}")
    print(f"Successfully downloaded: {successful}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed PDB IDs:")
        for pdb_id in failed:
            print(f"  - {pdb_id}")
    
    print(f"\nAll files have been saved to: {output_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Download PDB assembly files in CIF format')
    
    # Create subparsers for single and batch modes
    subparsers = parser.add_subparsers(dest='mode', help='Download mode')
    
    # Single PDB download
    single_parser = subparsers.add_parser('single', help='Download a single PDB assembly')
    single_parser.add_argument('pdb_id', help='PDB ID (e.g., 9DLW)')
    single_parser.add_argument('-o', '--output-dir', help='Output directory (default: current directory)')
    single_parser.add_argument('-a', '--assembly-id', type=int, default=1, 
                        help='Assembly ID (default: 1)')
    single_parser.add_argument('-k', '--keep-compressed', action='store_true',
                        help='Keep the compressed .gz file only (default: decompress)')
    
    # Batch download
    batch_parser = subparsers.add_parser('batch', help='Download multiple PDB assemblies')
    batch_parser.add_argument('-i', '--input-file', help='File containing PDB IDs (one per line)')
    batch_parser.add_argument('-o', '--output-dir', required=True, help='Output directory')
    batch_parser.add_argument('-a', '--assembly-id', type=int, default=1, 
                       help='Assembly ID (default: 1)')
    batch_parser.add_argument('-k', '--keep-compressed', action='store_true',
                       help='Keep the compressed .gz files only (default: decompress)')
    batch_parser.add_argument('-d', '--delay', type=float, default=0.5,
                       help='Delay between downloads in seconds (default: 0.5)')
    batch_parser.add_argument('--use-default-list', action='store_true',
                       help='Use the default list of PDB IDs included in the script')
    
    return parser.parse_args()


# Default list of PDB IDs
DEFAULT_PDB_IDS = [
    '6BN8', '6BN9', '6BNB', '6DL2', '6EW6', '6EW7', '6EW8', '6GMN', '6GMQ', 
    '6GMX', '6HAZ', '6R0Q', '6R0U', '6R0V', '6R11', '6R12', '6R13', '6R18', 
    '6R19', '6R1A', '6R1C', '6R1D', '6R1K', '6R1W', '6R1X', '6W74', '6WTP', 
    '6WTQ', '6WWB', '7RMD', '7RN2', '7TVA', '7TVB', '7UBT', '7UC6', '7UC7', 
    '7Z76', '7Z77', '7Z78', '8AOP', '8AOQ', '8BD8', '8BD9', '8BDI', '8BDJ', 
    '8BDL', '8BDM', '8BDN', '8BDO', '8BDY', '8BFM', '8EBK', '8EMU', '8EXC', 
    '8EXG', '8EYL', '8FTQ', '8OIZ', '8OJH', '8OO5', '8OOD', '8OU3', '8OU4', 
    '8OU5', '8OU6', '8OU7', '8OU9', '8OUA', '8PC2', '8QU8', '8R5H', '8RQ1', 
    '8RQ8', '8RQ9', '8RQA', '8RQC', '8S6F', '8S75', '8S76', '8S77', '8T2H', 
    '8U0H', '8V1O', '8V2F', '8V2L', '8WDK', '8WFP', '8X71', '8X73', '8Y58', 
    '8Y59', '8Y5B', '8YMB', '9BA2', '9BEQ', '9BIG', '9D11', '9D12', '9EQJ', 
    '9EQM', '9FJX', '9GAO'
]


def main():
    args = parse_arguments()
    
    # Handle single PDB download
    if args.mode == 'single':
        download_pdb_assembly(
            args.pdb_id,
            args.output_dir,
            args.assembly_id,
            not args.keep_compressed
        )
    
    # Handle batch download
    elif args.mode == 'batch':
        # Determine which PDB IDs to use
        if args.use_default_list:
            pdb_ids = DEFAULT_PDB_IDS
        elif args.input_file:
            # Read PDB IDs from file
            with open(args.input_file, 'r') as f:
                pdb_ids = [line.strip() for line in f if line.strip()]
        else:
            print("Error: Either --input-file or --use-default-list must be provided")
            sys.exit(1)
        
        batch_download(
            pdb_ids,
            args.output_dir,
            args.assembly_id,
            not args.keep_compressed,
            args.delay
        )
    
    # If no mode specified, show help
    else:
        print("Error: Please specify either 'single' or 'batch' mode")
        print("For help, use: python pdb_assembly_downloader.py -h")
        sys.exit(1)


if __name__ == "__main__":
    main() 