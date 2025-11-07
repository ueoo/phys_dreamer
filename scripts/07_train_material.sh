#!/usr/bin/env bash
set -euo pipefail

PY="/svl/u/yuegao/NeuROK/PhysDreamer/phys_dreamer/motion_exp/train_material.py"
DATASET_DIR="/abs/path/to/your_scene"
OUT_DIR="/abs/path/to/output/inverse_sim"

# Set this to the velocity checkpoint directory containing model.pt
VELO_CKPT="/abs/path/to/output/inverse_sim/<exp>/seed0/checkpoint_model_000199"

python "$PY" \
  --dataset_dir "$DATASET_DIR" \
  --video_dir_name "videos" \
  --checkpoint_path "$VELO_CKPT" \
  --num_frames 14 \
  --grid_size 64 \
  --substep 384 \
  --downsample_scale 0.06 \
  --output_dir "$OUT_DIR" \
  --wandb_name "your_scene_material"
