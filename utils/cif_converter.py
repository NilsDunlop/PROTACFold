#!/usr/bin/env python3
"""
This script processes an mmCIF CCD file to produce a valid string for the userCCD field.
It performs the following tasks:
  1. Reads the entire CIF file and strips any leading whitespace so that the file starts with "data_".
  2. Searches for atom loop blocks (detected by "loop_" followed by header lines)
     and, if the header contains the ideal coordinate fields (_chem_comp_atom.pdbx_model_Cartn_x_ideal,
     _chem_comp_atom.pdbx_model_Cartn_y_ideal, _chem_comp_atom.pdbx_model_Cartn_z_ideal), then in the corresponding
     data lines any token that is "?" is replaced with "0.0".
  3. Converts all double quotes to single quotes (per the specification).
  4. Replaces actual newline characters with the literal '\n' so that the entire file appears as a one-line JSON string.
  5. Writes the processed text to an output file.

Usage:
    python fix_cif.py input.cif output.txt

Example:
    python fix_cif.py af_input/YF8.cif af_input/YF8_escaped.txt
"""

import sys

def process_atom_loop(lines, header_start, header_end):
    """
    Process the data lines of an atom loop.
    Assumes that the loop header (from header_start to header_end) includes:
      _chem_comp_atom.pdbx_model_Cartn_x_ideal
      _chem_comp_atom.pdbx_model_Cartn_y_ideal
      _chem_comp_atom.pdbx_model_Cartn_z_ideal
    This function replaces any token equal to "?" in the x/y/z columns with "0.0".
    """
    # Collect header tokens from the header block.
    header_tokens = []
    for i in range(header_start, header_end):
        token = lines[i].strip()
        if token:
            header_tokens.append(token)
    
    # Determine the indices of the ideal coordinate columns.
    try:
        idx_x = header_tokens.index("_chem_comp_atom.pdbx_model_Cartn_x_ideal")
        idx_y = header_tokens.index("_chem_comp_atom.pdbx_model_Cartn_y_ideal")
        idx_z = header_tokens.index("_chem_comp_atom.pdbx_model_Cartn_z_ideal")
    except ValueError:
        # If any of these fields is missing, do nothing.
        return
    
    # Process the data lines that follow the header.
    i = header_end
    while i < len(lines):
        current_line = lines[i].strip()
        # Stop processing if we hit an empty line, a new "loop_" or a comment line.
        if current_line == "" or current_line.startswith("loop_") or current_line.startswith("#"):
            break
        # Split the line into tokens by whitespace.
        tokens = current_line.split()
        # Replace coordinate tokens if they are "?". Only do this if we have enough tokens.
        for idx in [idx_x, idx_y, idx_z]:
            if idx < len(tokens) and tokens[idx] == "?":
                tokens[idx] = "0.0"
        # Reconstruct the line.
        lines[i] = " ".join(tokens)
        i += 1

def main():
    if len(sys.argv) != 3:
        print("Usage: {} input.cif output.txt".format(sys.argv[0]))
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Read and strip leading whitespace.
    with open(input_path, "r") as infile:
        content = infile.read()
    content = content.lstrip()
    if not content.startswith("data_"):
        print("Error: The CIF file does not start with 'data_' after stripping.")
        sys.exit(1)

    # Split the file into lines.
    lines = content.splitlines()

    # Look for "loop_" blocks that contain the atom ideal coordinate header.
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "loop_":
            header_start = i + 1
            header_end = header_start
            # Collect header lines (those starting with an underscore).
            while header_end < len(lines) and lines[header_end].strip().startswith("_"):
                header_end += 1
            # If this loop contains the coordinate header, process its data lines.
            header_block = " ".join(lines[header_start:header_end])
            if "_chem_comp_atom.pdbx_model_Cartn_x_ideal" in header_block:
                process_atom_loop(lines, header_start, header_end)
            i = header_end
        else:
            i += 1

    # Join the processed lines back together.
    fixed_content = "\n".join(lines)

    # Replace double quotes with single quotes.
    fixed_content = fixed_content.replace('"', "'")
    
    # Replace actual newline characters with the literal "\n" (backslash and n).
    fixed_content = fixed_content.replace("\n", "\\n")

    # Write the final processed string to the output file.
    with open(output_path, "w") as outfile:
        outfile.write(fixed_content)

    print("Processed CIF content has been written to '{}'.".format(output_path))

if __name__ == '__main__':
    main()