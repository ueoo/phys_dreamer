#!/usr/bin/env bash
set -euo pipefail

PY="motion_exp/fast_train_velocity.py"
dataset_dir="data_NeuROK_sim/flower_images"
video_dir="data_NeuROK_sim/flower_videos"
output_dir="output/flower_inverse_sim"
wandb_name="flower_velo_pretrain"


python "$PY" \
  --dataset_dir "$dataset_dir" \
  --video_dir "$video_dir" \
  --downsample_scale 0.02 \
  --output_dir "$output_dir" \
  --use_wandb \
  --wandb_name "$wandb_name" \
  --wandb_entity "ueoo-cs" \
  --wandb_project "phys_dreamer_inverse_sim"  \
