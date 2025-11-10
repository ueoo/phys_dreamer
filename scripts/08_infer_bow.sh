#!/usr/bin/env bash
{
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3

PY="motion_exp/inference.py"
SCENE_CFG="bow"  # adjust; see phys_dreamer/configs/*.py
DATASET_DIR="data_NeuROK_sim/bow_images"
OUT_DIR="output/bow_results"

python "$PY" \
  --scene_name "$SCENE_CFG" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUT_DIR" \
  --apply_force \
  --force_id 1 \
  --force_mag 0.1 \
  --point_id 1 \
  --velo_scaling 5.0 \
  --cam_id 2 \
  --run_eval

exit 0
}
