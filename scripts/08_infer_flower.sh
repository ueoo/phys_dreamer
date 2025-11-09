#!/usr/bin/env bash
set -euo pipefail

PY="motion_exp/inference.py"
SCENE_CFG="flower"  # adjust; see phys_dreamer/configs/*.py
DATASET_DIR="data_NeuROK_sim/flower_images"
OUT_DIR="output/flower_results"

python "$PY" \
  --scene_name "$SCENE_CFG" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUT_DIR" \
  --run_eval
