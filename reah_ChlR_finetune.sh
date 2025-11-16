#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

data_path=data/rhea_18421_final.json
reaction="18421"

local_root=models
pretrained_model="esm2_t33_650M_UR50D"
save_path=${local_root}/ProteinNet
output_path=${local_root}/rhea_18421_finetune
rm -rf ${output_path}
mkdir ${output_path}

python3 fairseq_cli/finetune.py ${data_path} \
--finetune-from-model ${save_path}/checkpoint_best.pt \
--save-dir ${output_path} \
--task geometric_protein_design \
--protein-task ${reaction} \
--dataset-impl-source "raw" \
--dataset-impl-target "coor" \
--data-stage "finetuning" \
--criterion geometric_protein_ncbi_loss --encoder-factor 1.0 --decoder-factor 1e-2 \
--arch geometric_protein_model_ncbi_esm \
--encoder-embed-dim 1280 \
--egnn-mode "rm-node" \
--decoder-layers 3 \
--pretrained-esm-model ${pretrained_model} \
--knn 30 \
--dropout 0.3 \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 5e-5 --lr-scheduler inverse_sqrt \
--stop-min-lr '1e-10' --warmup-updates 4000 \
--warmup-init-lr '1e-5' \
--clip-norm 0.0001 \
--ddp-backend legacy_ddp \
--log-format 'simple' --log-interval 10 \
--max-tokens 1024 \
--update-freq 1 \
--max-update 300000 \
--max-epoch 50 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--valid-subset valid \
--max-sentences-valid 8 \
--validate-interval 1 \
--save-interval 1 \
--keep-interval-updates	10 \
--skip-invalid-size-inputs-valid-test