#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

data_path=data/protein_ligand_enzyme_test.json

local_root=models
output_path=${local_root}/ProteinNet
out_pdb_path=${local_root}/output/ProteinNet
mkdir -p ${out_pdb_path}

proteins=(10665 11698 9796 11706 573 11686 83332 11676 186497 273057 4081 510516 287 5116 36329 273063 11678 694009 93062 264203 2697049 9031 9606 83333 1280 562)
for element in ${proteins[@]}
do
generation_path=${out_pdb_path}/${element}

mkdir -p ${generation_path}
mkdir -p ${generation_path}/pred_pdbs
mkdir -p ${generation_path}/tgt_pdbs

python3 fairseq_cli/generate.py ${data_path} \
--task geometric_protein_design \
--protein-task ${element} \
--dataset-impl-source "raw" \
--dataset-impl-target "coor" \
--data-stage "pretraining-motif" \
--path ${output_path}/checkpoint_best.pt \
--batch-size 1 \
--results-path ${generation_path} \
--skip-invalid-size-inputs-valid-test \
--valid-subset test \
--generation \
--decoding-strategy "greedy"
done
