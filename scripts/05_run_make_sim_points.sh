#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="/abs/path/to/your_scene"

# Approximate a point on the object that should remain free (others frozen)
python "$(dirname "$0")/05_make_sim_points.py" --dataset_dir "$DATASET_DIR" \
  --move_center -0.39 0.14 -0.18 \
  --move_radius 0.06 \
  --fill
