#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

PY="motion_exp/train_velocity.py"
dataset_dir="data_NeuROK_sim/cloth_images"
video_dir="data_NeuROK_sim/cloth_videos"
output_dir="output/cloth_inverse_sim"
wandb_name="cloth_velo_pretrain"


python "$PY" \
  --dataset_dir "$dataset_dir" \
  --video_dir "$video_dir" \
  --output_dir "$output_dir" \
  --use_wandb \
  --wandb_name "$wandb_name" \
  --wandb_entity "ueoo-cs" \
  --wandb_project "phys_dreamer_inverse_sim"  \
