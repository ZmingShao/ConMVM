#!/bin/bash
set -x  # print the commands

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'0'}  # Set the GPU you want to use

ROOT_DIR=/path/to/project
OUTPUT_DIR=${ROOT_DIR}/path/to/workdir 
DATA_PATH=${ROOT_DIR}/path/to/csv 
DATA_ROOT=${ROOT_DIR}/path/to/data

GPUS_PER_NODE=${GPUS_PER_NODE:-1}  # Number of GPUs in each node

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=29500 \
        run_mae_pretraining.py \
        --fname_tmpl '{}.jpg' \
        --data_root ${DATA_ROOT} \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.0 \
        --model pretrain_videomae_small_patch16_224 \
        --teacher_model pretrain_vit_large_patch16_224 \
        --teacher_ckpt ${ROOT_DIR}/path/to/teacher \
        --ct_loss_type infoNCE \
        --decoder_depth 2 \
        --batch_size 64 \
        --num_sample 1 \
        --num_frames 16 \
        --sampling_rate 3 \
        --input_size 224 \
        --shift_frames \
        --shift_pixel \
        --num_workers 10 \
        --lr 1.5e-4 \
        --min_lr 1e-4 \
        --drop_path 0.1 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 20 \
        --save_ckpt_freq 50 \
        --epochs 200 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --ct_weight 1.0 