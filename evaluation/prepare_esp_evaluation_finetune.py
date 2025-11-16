import numpy as np
import os 
import json


ec2smiles = {"Thiopurine_S_methyltransferas": "C[S+](CC[C@H]([NH3+])C([O-])=O)C[C@H]1O[C@H]([C@H](O)[C@@H]1O)n1cnc2c(N)ncnc12",
            "18421": "InChI=1S/C23H38N7O17P3S/c1-12(31)51-7-6-25-14(32)4-5-26-21(35)18(34)23(2,3)9-44-50(41,42)47-49(39,40)43-8-13-17(46-48(36,37)38)16(33)22(45-13)30-11-29-15-19(24)27-10-28-20(15)30/h10-11,13,16-18,22,33-34H,4-9H2,1-3H3,(H,25,32)(H,26,35)(H,39,40)(H,41,42)(H2,24,27,28)(H2,36,37,38)/p-4/t13-,16-,17-,18+,22-/m1/s1",
            "20245": "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP([O-])(=O)OP([O-])(=O)OP([O-])([O-])=O)[C@@H](O)[C@H]1O",
            }

data_path = 'models/output/finetune'
for key in ec2smiles:
    proteins = open(os.path.join(data_path, f"{key}_topp0.4/protein.txt")).readlines()
    fw = open(os.path.join(data_path, f"{key}_topp0.4/protein_substrate.txt"), "w")

    for protein in proteins:
        line = protein.strip() + " " + ec2smiles[key]
        fw.write(line + "\n")
    fw.close()