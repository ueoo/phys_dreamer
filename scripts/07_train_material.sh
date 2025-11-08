#!/usr/bin/env bash
set -euo pipefail

PY="./scripts/train_material.py"
dataset_dir="./data_NeuROK_sim/flower_images"
video_dir="./data_NeuROK_sim/flower_videos"
output_dir="./output/scene_1"
wandb_name="flower_material_train"


# Set this to the velocity checkpoint directory containing model.pt
VELO_CKPT="./output/inverse_sim/scene_1/seed0/checkpoint_model_000199"

python "$PY" \
  --dataset_dir "$dataset_dir" \
  --video_dir "$video_dir" \
  --checkpoint_path "$VELO_CKPT" \
  --output_dir "$output_dir" \
  --use_wandb \
  --wandb_name "$wandb_name" \
  --wandb_entity "ueoo-cs" \
  --wandb_project "phys_dreamer_inverse_sim"  \
