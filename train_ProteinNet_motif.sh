#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

data_path=data/pdb_swissprot_data_ligand.json

local_root=models
pretrained_model="esm2_t33_650M_UR50D"
save_path=${local_root}/ProteinNet_MLM
output_path=${local_root}/ProteinNet_motif
mkdir ${output_path}

python3 fairseq_cli/train.py ${data_path} \
--profile \
--num-workers 0 \
--distributed-world-size 8 \
--finetune-from-model ${save_path}/checkpoint_best.pt \
--save-dir ${output_path} \
--task geometric_protein_design \
--dataset-impl-source "raw" \
--dataset-impl-target "coor" \
--data-stage "pretraining-motif" \
--criterion geometric_protein_ncbi_loss --encoder-factor 1.0 --decoder-factor 1e-2 \
--arch geometric_protein_model_ncbi_esm \
--encoder-embed-dim 1280 \
--egnn-mode "rm-node" \
--decoder-layers 3 \
--pretrained-esm-model ${pretrained_model} \
--knn 30 \
--dropout 0.3 \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 1e-4 --lr-scheduler inverse_sqrt \
--stop-min-lr '1e-10' --warmup-updates 4000 \
--warmup-init-lr '5e-5' \
--clip-norm 1.5 \
--ddp-backend legacy_ddp \
--log-format 'simple' --log-interval 10 \
--max-tokens 1024 \
--update-freq 1 \
--max-update 1000000 \
--max-epoch 10000 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--valid-subset valid \
--max-sentences-valid 8 \
--validate-interval 1 \
--save-interval 1 \
--keep-interval-updates	10 \
--skip-invalid-size-inputs-valid-test



