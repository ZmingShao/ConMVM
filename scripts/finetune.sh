#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'0'}  # Set the GPU you want to use

ROOT_DIR=/path/to/project
OUTPUT_DIR=${ROOT_DIR}/path/to/workdir 
DATA_ROOT=${ROOT_DIR}/path/to/data
MODEL_PATH=${ROOT_DIR}/path/to/ckpt

GPUS_PER_NODE=${GPUS_PER_NODE:-1}

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=29500 \
        run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set CAG \
        --nb_classes 2 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 30 \
        --num_frames 16 \
        --sampling_rate 3 \
        --num_workers 10 \
        --opt adamw \
        --lr 7e-4 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --layer_decay 0.75 \
        --epochs 10 \
        --warmup_epochs 0 \
        --dist_eval \
        --no_auto_resume \
        --mixup 0. --cutmix 0. \
        --smoothing 0.
