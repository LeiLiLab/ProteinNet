#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

data_path=data/rhea_18421_final.json
enzyme="18421"
topp_probability=0.4

local_root=models
output_path=${local_root}/rhea_${enzyme}_finetune
generation_path=${local_root}/output/finetune/${enzyme}_topp${topp_probability}

mkdir -p ${generation_path}
mkdir -p ${generation_path}/tgt_pdbs
mkdir -p ${generation_path}/pred_pdbs

python3 fairseq_cli/generate.py ${data_path} \
--task geometric_protein_design \
--protein-task ${enzyme} \
--dataset-impl-source "raw" \
--dataset-impl-target "coor" \
--data-stage "finetuning" \
--path ${output_path}/checkpoint_best.pt \
--batch-size 1 \
--results-path ${generation_path} \
--skip-invalid-size-inputs-valid-test \
--valid-subset test \
--generation \
--decoding-strategy "top-p" \
--topp-probability 0.4