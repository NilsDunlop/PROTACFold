from pymol import cmd

# ------------------------------------------------------------------------------
# Provide your amino acid chain sequences here.
#
# IMPORTANT: If these variables are left blank the script will exit.
# ------------------------------------------------------------------------------
POI_SEQUENCE = """MHHHHHHSSGRENLYFQGTQSKPTPVKPNYALKFTLAGHTKAVSSVKFSPNGEWLASSSADKLIKIWGAYDGKFEKTISGHKLGISDVAWSSDSNLLVSASDDKTLKIWDVSSGKCLKTLKGHSNYVFCCNFNPQSNLIVSGSFDESVRIWDVKTGKCLKTLPAHSDPVSAVHFNRDGSLIVSSSYDGLCRIWDTASGQCLKTLIDDDNPPVSFVKFSPNGKYILAATLDNTLKLWDYSKGKCLKTYTGHKNEKYCIFANFSVTGGKWIVSGSEDNLVYIWNLQTKEIVQKLQGHTDVVISTACHPTENIIASAALENDKTIKLWKSDC"""

E3_SEQUENCE = """MGSSHHHHHHSSGRENLYFQGSSRASAFRPISVFREANEDESGFTCCAFSARERFLMLGTCTGQLKLYNVFSGQEEASYNCHNSAITHLEPSRDGSLLLTSATWSQPLSALWGMKSVFDMKHSFTEDHYVEFSKHSQDRVIGTKGDIAHIYDIQTGNKLLTLFNPDLANNYKRNCATFNPTDDLVLNDGVLWDVRSAQAIHKFDKFNMNISGVFHPNGLEVIINTEIWDLRTFHLLHTVPALDQCRVVFNHTGTVMYGAMLQADDEDDLMEERMKSPFGSSFRTFNATDYKPIATIDVKRNIFDLCTDTKDCYLAVIENQGSMDALNMDTVCRLYEVG"""

# ------------------------------------------------------------------------------
# Helper function: Look for a chain that contains the given sequence.
#
# Both the provided sequence and the FASTA string from PyMOL have whitespace
# (newlines/spaces) removed to avoid formatting issues.
# ------------------------------------------------------------------------------
def find_chain_by_sequence(sequence, model_name):
    """
    Searches for a chain within the specified model that contains the given
    amino acid sequence.
    Returns the chain ID if found, or raises ValueError otherwise.
    """
    seq_clean = "".join(sequence.split())  # Remove all whitespace
    for chain in cmd.get_chains(model_name):
        fasta = cmd.get_fastastr(f"{model_name} and chain {chain}")
        fasta_clean = "".join(fasta.split())
        if seq_clean in fasta_clean:
            return chain
    raise ValueError(f"Sequence not found in model '{model_name}'.")

# ------------------------------------------------------------------------------
# Main routine: Automatically detect the models, find the target chains (POI and
# E3) in each, create selections and compute the RMSDs using align.
#
# The names of the models are determined from the filename:
#   - Original model: name does NOT contain 'smiles_model' or 'ccd_model'
#   - Smiles model: name contains 'smiles_model'
#   - CCD model: name contains 'ccd_model'
# ------------------------------------------------------------------------------
def calculate_rmsd():
    # Get all loaded objects (models) in PyMOL.
    models = cmd.get_object_list("all")
    if not models:
        print("Error: No models loaded. Please load your model files first.")
        return

    # Auto-detect model names based on naming conventions.
    orig_models = [m for m in models if "smiles_model" not in m and "ccd_model" not in m]
    smiles_models = [m for m in models if "smiles_model" in m]
    ccd_models = [m for m in models if "ccd_model" in m]

    if len(orig_models) != 1:
        print("Error: Expected exactly one original model, found:", ", ".join(orig_models))
        return
    if len(smiles_models) != 1:
        print("Error: Expected exactly one smiles model, found:", ", ".join(smiles_models))
        return
    if len(ccd_models) != 1:
        print("Error: Expected exactly one ccd model, found:", ", ".join(ccd_models))
        return

    original_model = orig_models[0]
    smiles_model   = smiles_models[0]
    ccd_model      = ccd_models[0]

    print(f"Identified models: Original='{original_model}', Smiles='{smiles_model}', CCD='{ccd_model}'")

    # ----- Check Sequences -----
    poi_seq = POI_SEQUENCE.strip()
    e3_seq  = E3_SEQUENCE.strip()

    if not poi_seq:
        print("Error: POI sequence is empty. Please provide a sequence in the script.")
        return
    if not e3_seq:
        print("Error: E3 sequence is empty. Please provide a sequence in the script.")
        return

    # ----- Process POI (Protein Of Interest) -----
    print("\n--- Processing POI chain ---")
    try:
        poi_chain_orig = find_chain_by_sequence(poi_seq, original_model)
        poi_chain_smiles = find_chain_by_sequence(poi_seq, smiles_model)
        poi_chain_ccd = find_chain_by_sequence(poi_seq, ccd_model)
    except ValueError as e:
        print("POI error:", e)
        return

    sel_poi_orig   = f"chain{poi_chain_orig}_POI"
    sel_poi_smiles = f"chain{poi_chain_smiles}_POI_smiles"
    sel_poi_ccd    = f"chain{poi_chain_ccd}_POI_ccd"

    cmd.select(sel_poi_orig,   f"{original_model} and chain {poi_chain_orig}")
    cmd.select(sel_poi_smiles, f"{smiles_model} and chain {poi_chain_smiles}")
    cmd.select(sel_poi_ccd,    f"{ccd_model} and chain {poi_chain_ccd}")

    # Align the Smiles and CCD POI selections to the Original POI selection.
    rmsd_smiles = cmd.align(sel_poi_smiles, sel_poi_orig)[0]
    rmsd_ccd    = cmd.align(sel_poi_ccd, sel_poi_orig)[0]

    print(f"\nRMSD for POI (smiles vs. original): {rmsd_smiles:.3f} Å")
    print(f"RMSD for POI (ccd vs. original): {rmsd_ccd:.3f} Å")

    # ----- Process E3 Ligase -----
    print("\n--- Processing E3 Ligase chain ---")
    try:
        e3_chain_orig = find_chain_by_sequence(e3_seq, original_model)
        e3_chain_smiles = find_chain_by_sequence(e3_seq, smiles_model)
        e3_chain_ccd = find_chain_by_sequence(e3_seq, ccd_model)
    except ValueError as e:
        print("E3 error:", e)
        return

    sel_e3_orig   = f"chain{e3_chain_orig}_E3"
    sel_e3_smiles = f"chain{e3_chain_smiles}_E3_smiles"
    sel_e3_ccd    = f"chain{e3_chain_ccd}_E3_ccd"

    cmd.select(sel_e3_orig,   f"{original_model} and chain {e3_chain_orig}")
    cmd.select(sel_e3_smiles, f"{smiles_model} and chain {e3_chain_smiles}")
    cmd.select(sel_e3_ccd,    f"{ccd_model} and chain {e3_chain_ccd}")

    rmsd_e3_smiles = cmd.align(sel_e3_smiles, sel_e3_orig)[0]
    rmsd_e3_ccd    = cmd.align(sel_e3_ccd, sel_e3_orig)[0]

    print(f"\nRMSD for E3 (smiles vs. original): {rmsd_e3_smiles:.3f} Å")
    print(f"RMSD for E3 (ccd vs. original): {rmsd_e3_ccd:.3f} Å")

# ------------------------------------------------------------------------------
# Register as a PyMOL command.
# After running this script (e.g. run pymol_scripts/calculate_rmsd.py) in PyMOL,
# simply type "calculate_rmsd" in the PyMOL command prompt.
# ------------------------------------------------------------------------------
cmd.extend("calculate_rmsd", calculate_rmsd)