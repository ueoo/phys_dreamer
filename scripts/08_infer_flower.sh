#!/usr/bin/env bash
{
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3

PY="motion_exp/inference.py"
SCENE_CFG="flower"  # adjust; see phys_dreamer/configs/*.py
DATASET_DIR="data_NeuROK_sim/flower_images"
OUT_DIR="output/flower_results_3"

python "$PY" \
  --scene_name "$SCENE_CFG" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUT_DIR" \
  --apply_force \
  --force_id 0 \
  --force_mag 1.0 \
  --point_id 0 \
  --velo_scaling 1.0 \
  --cam_id 2 \
  --run_eval

exit 0
}
