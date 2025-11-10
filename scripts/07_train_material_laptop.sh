#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1


PY="motion_exp/train_material.py"
dataset_dir="data_NeuROK_sim/laptop_images"
video_dir="data_NeuROK_sim/laptop_videos"
output_dir="output/laptop_inverse_sim_material"
wandb_name="laptop_material_train"


# Set this to the velocity checkpoint directory containing model.pt
VELO_CKPT="output/laptop_inverse_sim/laptop_velo_pretraindecay_1.0_substep_96_se3_field_lr_0.01_tv_0.01_iters_300_sw_2_cw_2/seed0/checkpoint_model_000300"

python "$PY" \
  --dataset_dir "$dataset_dir" \
  --video_dir "$video_dir" \
  --checkpoint_path "$VELO_CKPT" \
  --output_dir "$output_dir" \
  --downsample_scale 0.01 \
  --use_wandb \
  --wandb_name "$wandb_name" \
  --wandb_entity "ueoo-cs" \
  --wandb_project "phys_dreamer_inverse_sim"  \
