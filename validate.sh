#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --error=err.txt
#SBATCH --output=test.txt
#SBATCH --partition=taurus
#SBATCH --gpus=1
#SBATCH --time=7-0:0:0
#SBATCH --account=zhenqiaosong
#SBATCH --mail-type=fail
#SBATCH --mail-user=zhenqiao@ucsb.edu

export CUDA_VISIBLE_DEVICES=7

data_path=test_data.json

local_root=models
output_path=${local_root}/33layer_model
generation_path=${local_root}/output/33layer_model
mkdir -p ${generation_path}

python3 fairseq_cli/validate.py ${data_path} \
--task protein_protein_complex_design \
--protein-task "PDB" \
--dataset-impl "protein_complex" \
--path ${output_path}/checkpoint_best.pt \
--batch-size 1 \
--results-path ${generation_path} \
--skip-invalid-size-inputs-valid-test \
--valid-subset test \
--eval-aa-recovery