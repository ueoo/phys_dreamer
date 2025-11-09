#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

PY="motion_exp/train_velocity.py"
dataset_dir="data_NeuROK_sim/box_images"
video_dir="data_NeuROK_sim/box_videos"
output_dir="output/box_inverse_sim"
wandb_name="box_velo_pretrain"


python "$PY" \
  --dataset_dir "$dataset_dir" \
  --video_dir "$video_dir" \
  --output_dir "$output_dir" \
  --use_wandb \
  --wandb_name "$wandb_name" \
  --wandb_entity "ueoo-cs" \
  --wandb_project "phys_dreamer_inverse_sim"  \
