#!/usr/bin/env bash
set -euo pipefail

PY="scripts/train_material.py"
dataset_dir="data_NeuROK_sim/newton_images"
video_dir="data_NeuROK_sim/newton_videos"
output_dir="output/newton_inverse_sim_material"
wandb_name="newton_material_train"


# Set this to the velocity checkpoint directory containing model.pt
VELO_CKPT="output/newton_inverse_sim/newton_velo_pretrain_whitebgdecay_1.0_substep_96_se3_field_lr_0.01_tv_0.01_iters_300_sw_2_cw_2/seed0/checkpoint_model_000199"

python "$PY" \
  --dataset_dir "$dataset_dir" \
  --video_dir "$video_dir" \
  --checkpoint_path "$VELO_CKPT" \
  --output_dir "$output_dir" \
  --use_wandb \
  --wandb_name "$wandb_name" \
  --wandb_entity "ueoo-cs" \
  --wandb_project "phys_dreamer_inverse_sim"  \
