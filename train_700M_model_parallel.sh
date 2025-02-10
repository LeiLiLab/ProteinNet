#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --error=err.txt
#SBATCH --output=train.txt
#SBATCH --partition=aries
#SBATCH --gres=gpu:2
#SBATCH --time=7-0:0:0
#SBATCH --account=zhenqiaosong
#SBATCH --mail-type=fail
#SBATCH --mail-user=zhenqiao@ucsb.edu

export CUDA_VISIBLE_DEVICES=0,1

data_path=/mnt/ml_data/zhsong/test_data.json

local_root=/mnt/ml_data/zhsong/models
pretrained_model="esm2_t33_650M_UR50D"
output_path=${local_root}/680M_model
rm -rf ${output_path}
mkdir ${output_path}

python3 -m torch.distributed.launch fairseq_cli/train_parallel.py ${data_path} \
--profile \
--num-workers 0 \
--distributed-world-size 2 \
--save-dir ${output_path} \
--task protein_protein_complex_design \
--protein-task "PDB" \
--dataset-impl "protein_complex" \
--criterion protein_complex_diffusion_loss \
--arch protein_protein_complex_diffusion_model_esm \
--encoder-embed-dim 1280 \
--egnn-mode "rm-node" \
--decoder-layers 3 \
--autoregressive-layer 1 \
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
--knn 32 \
--dropout 0.3 \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 5e-6 --lr-scheduler inverse_sqrt \
--stop-min-lr '1e-12' --warmup-updates 4000 \
--warmup-init-lr '1e-6' \
--clip-norm 0.0001 \
--ddp-backend legacy_ddp \
--log-format 'simple' --log-interval 10 \
--max-tokens 1024 \
--update-freq 1 \
--max-update 1000000 \
--max-epoch 100 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--train-subset test \
--valid-subset valid \
--max-sentences-valid 8 \
--validate-interval 1 \
--save-interval 1 \
--keep-interval-updates	10 \
--skip-invalid-size-inputs-valid-test

