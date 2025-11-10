#!/usr/bin/env bash
{
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

PY="motion_exp/inference.py"
SCENE_CFG="box"  # adjust; see phys_dreamer/configs/*.py
DATASET_DIR="data_NeuROK_sim/box_images_infer"
OUT_DIR="output/box_results"

python "$PY" \
  --scene_name "$SCENE_CFG" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUT_DIR" \
  --point_id 0 \
  --velo_scaling 1.0 \
  --cam_id 4 \
  --run_eval

exit 0
}
