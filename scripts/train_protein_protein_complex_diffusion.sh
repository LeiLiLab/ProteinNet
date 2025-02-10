#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --error=err.18421.txt
#SBATCH --output=18421.txt
#SBATCH --partition=aries
#SBATCH --gres=gpu:1
#SBATCH --time=7-0:0:0
#SBATCH --account=zhenqiaosong
#SBATCH --mail-type=fail
#SBATCH --mail-user=zhenqiao@ucsb.edu

export CUDA_VISIBLE_DEVICES=7

data_path=/mnt/taurus/data1/zhenqiaosong/protein/protein_protein_complex_design/proein_protein_complex_data.json
# data_path=/mnt/taurus/data1/zhenqiaosong/protein/protein_protein_complex_design/test.json

local_root=/mnt/taurus/data1/zhenqiaosong/protein/protein_protein_complex_design/models
pretrained_model="esm2_t30_150M_UR50D"
output_path=${local_root}/30layerESM_5EGNN_diffusion_stable
rm -rf ${output_path}
mkdir ${output_path}

python3 fairseq_cli/train.py ${data_path} \
--save-dir ${output_path} \
--task protein_protein_complex_design \
--protein-task "PDB" \
--dataset-impl "protein_complex" \
--criterion protein_complex_diffusion_loss \
--arch protein_protein_complex_diffusion_model_esm \
--encoder-embed-dim 640 \
--egnn-mode "rm-node" \
--decoder-layers 5 \
--pretrained-esm-model ${pretrained_model} \
--model-mean-type "C0" \
--sample-time-method "symmetric" \
--beta-schedule "sigmoid" \
--num-diffusion-timesteps 1000 \
--pos-beta-s 2 --beta-start 1e-7 --beta-end 2e-3 \
--r-beta-schedule "cosine" --r-beta-s 0.01 \
--time-emb-dim 0 --time-emb-mode "simple" \
--loss-r-weight 50 \
--max-source-positions 1024 \
--knn 16 \
--dropout 0.3 \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 5e-6 --lr-scheduler inverse_sqrt \
--stop-min-lr '1e-12' --warmup-updates 4000 \
--warmup-init-lr '1e-6' \
--clip-norm 0.0001 \
--ddp-backend legacy_ddp \
--log-format 'simple' --log-interval 10 \
--max-tokens 4096 \
--update-freq 1 \
--max-update 2000000 \
--max-epoch 100 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--valid-subset valid \
--max-sentences-valid 8 \
--validate-interval 1 \
--save-interval 1 \
--keep-interval-updates	10 \
--skip-invalid-size-inputs-valid-test

