#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

PY="motion_exp/train_velocity.py"
dataset_dir="data_NeuROK_sim/laptop_images"
video_dir="data_NeuROK_sim/laptop_videos"
output_dir="output/laptop_inverse_sim"
wandb_name="laptop_velo_pretrain"


python "$PY" \
  --dataset_dir "$dataset_dir" \
  --video_dir "$video_dir" \
  --output_dir "$output_dir" \
  --use_wandb \
  --wandb_name "$wandb_name" \
  --wandb_entity "ueoo-cs" \
  --wandb_project "phys_dreamer_inverse_sim"  \
