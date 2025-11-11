#!/usr/bin/env bash
{
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

PY="motion_exp/inference.py"
SCENE_CFG="shirt"  # adjust; see phys_dreamer/configs/*.py
DATASET_DIR="data_NeuROK_sim/shirt_images"
OUT_DIR="output/shirt_results"

python "$PY" \
  --scene_name "$SCENE_CFG" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUT_DIR" \
  --point_id 1 \
  --velo_scaling 1.0 \
  --cam_id 2 \
  --run_eval

exit 0
}
