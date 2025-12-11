#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

data_path=example/example.json

local_root=models
output_path=${local_root}/ProteinNet
out_pdb_path=${local_root}/output/ProteinNet
mkdir -p ${out_pdb_path}

proteins=(83332)
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
--path ${output_path}/checkpoint_best.pt \
--batch-size 1 \
--results-path ${generation_path} \
--skip-invalid-size-inputs-valid-test \
--valid-subset test \
--generation \
--decoding-strategy "top-p" \
--topp-probability 0.2
done