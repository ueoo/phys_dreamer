#!/usr/bin/env bash
set -euo pipefail

PY="./demo.py"
SCENE_CFG="hat"  # adjust; see phys_dreamer/configs/*.py
DATASET_DIR="/abs/path/to/your_scene"
OUT_DIR="/abs/path/to/output/inverse_sim"

python "$PY" \
  --scene_name "$SCENE_CFG" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUT_DIR" \
  --run_eval
