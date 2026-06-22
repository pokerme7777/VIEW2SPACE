#!/usr/bin/env bash

# Replace the paths below with your own local paths before running.
# Keep the training arguments below unchanged unless you intentionally want to tune them.

SRC_ROOT="/path/to/VIEW2SPACE_DEV/src"
TRAIN_FILE="/path/to/output/train.jsonl"
OUTPUT_DIR="/path/to/your/trained_model"
DEEPSPEED_CONFIG="$SRC_ROOT/deepspeed_zero3.yaml"

conda activate view2space
export WANDB_PROJECT='PROJECT_NAME'  
export WANDB_NAME='WANDB_RUN_NAME'
export CUDA_HOME=$CONDA_PREFIX

VERSION=v1.0
# e.g., VERSION_NAME v1.0

accelerate launch --config_file "$DEEPSPEED_CONFIG" "$SRC_ROOT/train/train_qwen3vl_trl_sft.py" \
        --train_file "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --dataloader_num_workers 10 \
        --dataloader_prefetch_factor 8 \
        --dataloader_pin_memory \
        --dataloader_persistent_workers \
        --model_name Qwen/Qwen3-VL-4B-Instruct \
        --max_seq_length 8192 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 64\
        --learning_rate 3e-5\
        --warmup_ratio 0.05\
        --logging_steps 1 \
        --output_model_folder_subname $VERSION \
        --attn_impl flash_attention_2 \
        --report_to wandb
