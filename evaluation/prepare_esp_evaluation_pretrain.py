import numpy as np
import os 
import json

ecs = ["2.7.7.7", "3.2.1.17", "3.5.2.6", "3.4.23.22", "3.6.4.13", "2.4.2.29", "3.1.31.1", "4.6.1.24", "3.4.22.69", "1.1.1.1"]
ec2smiles = {"2.7.7.7": "C00677",
             "3.2.1.17": "C04628",
             "3.5.2.6": "C00038",
             "3.4.23.22": "[O-]S([O-])(=O)=O",
             "3.6.4.13": "[Zn+2]",
             "2.4.2.29": "C16675",
             "3.1.31.1": "[Ca+2]",
             "4.6.1.24": "[H]O[H]", # RNA
             "3.4.22.69": "[H]O[H]", # peptide
             "1.1.1.1": "C00003"}

pdb2ec = json.load(open("data/protein_ligand_enzyme_test_pdb2ec.json"))

# generation path
data_path = "models/output/ProteinNet/pretrain"

pdbs = open(os.path.join(data_path, "pdb.txt")).readlines()
proteins = open(os.path.join(data_path, "protein.txt")).readlines()
fw = open(os.path.join(data_path, "protein_substrate.txt"), "w")

for pdb, protein in zip(pdbs, proteins):
    line = protein.strip() + " " + ec2smiles[pdb2ec[pdb.strip()]]
    fw.write(line + "\n")
fw.close()