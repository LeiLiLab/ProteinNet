import json 
import os 
import argparse

from Bio.PDB import MMCIFParser

three2one_dict = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                   "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                   "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


parser = argparse.ArgumentParser()
parser.add_argument("--pdb_file_path", default="3DWZ.cif", type=str, help='PDB file path')
parser.add_argument("--protein_id", default="3DWZ", type=str, help='PDB or Uniprot id')
parser.add_argument("--motif", default="6,8,9,14-18,23,30-44,47,49-52,54,60,64,66,70,75-78,80-87,94-118,125,126,129,130,132,134-136,139-143,147-156,160,163,166,168-172,177-184,187-190,194-197,199,200,202,204,206-209,211,213,214,217,218-223,229,231,234,238,245,246,251-256", type=str, help="motif fragment of the provided protein")
parser.add_argument("--pdb", default=1, type=int, help="indicate the provided pdb file is from PDB (1) or Uniprot(0)")
parser.add_argument("--ncbi_tag", default="83332", type=str, help="NCBI tag for the designed protein")
parser.add_argument("--output_path", default="example.json", type=str, help="output path for the extracted example data")
args = parser.parse_args()


def extract_seq_coor_from_cif_file(cif_file_path):
    """
    Parses a CIF file to extract the amino acid sequence and C-alpha coordinates.
    """
    parser = MMCIFParser()
    structure = parser.get_structure('protein_structure', cif_file_path)

    sequence = ""
    ca_coordinates = []
    target_chain_id = 'A'

    for model in structure:
        for chain in model:
            if chain.id != target_chain_id:
                continue
            
            for residue in chain:
                if residue.has_id("CA") and residue.get_resname() in three2one_dict:
                    
                    res_name = residue.get_resname()
                    
                    # Extract Sequence (Amino Acid)
                    try:
                        one_letter_code = three2one_dict[res_name]
                        sequence += one_letter_code
                    except KeyError:
                        continue
                        
                    # Extract C-alpha Coordinates
                    ca_atom = residue['CA']
                    coords = ca_atom.get_coord()
                    ca_coordinates.extend([coords[0].item(), coords[1].item(), coords[2].item()])
    assert len(sequence) == int(len(ca_coordinates)/3)
    return sequence, ca_coordinates


def prepare_example_data(pdb_file_path, protein_id, motif, pdb, ncbi_tag, output_path):
    data = {ncbi_tag: {"test": {"seq": [], "coor": [], "motif": [], "protein_id": [protein_id], "pdb": [pdb]}}}
    seq, coor = extract_seq_coor_from_cif_file(pdb_file_path)
    data[ncbi_tag]['test']['seq'].append(seq)
    data[ncbi_tag]['test']['coor'].append(",".join([str(coord) for coord in coor]))
    
    motif_frags = motif.split(",")
    motif_list = []
    for frag in motif_frags:
        inds = frag.split("-")
        if len(inds) == 1:
            motif_list.append(int(inds[0].strip()))
        else:
            start_id, end_id = int(inds[0].strip()), int(inds[1].strip())
            for index in range(start_id, end_id+1):
                motif_list.append(index)
    data[ncbi_tag]['test']['motif'].append(",".join([str(idx) for idx in motif_list]))
    
    with open(output_path, "w") as fw:
        json.dump(data, fw)


if __name__ == "__main__":
    prepare_example_data(args.pdb_file_path,
                         args.protein_id,
                         args.motif,
                         args.pdb,
                         args.ncbi_tag,
                         args.output_path)
            
    
