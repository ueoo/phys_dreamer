#!/usr/bin/env bash
set -euo pipefail

# Adjust these paths
DATASET_DIR="/abs/path/to/your_scene"   # has images/ + sparse/0 or transforms_train.json
MODEL_DIR="/abs/path/to/gs_models/your_scene"
ITER="30000"                              # desired iteration to export

mkdir -p "$MODEL_DIR"

# Run embedded Gaussian Splatting training
python "$(dirname "$0")/../gaussian_3d/train.py" \
  --source_path "$DATASET_DIR" \
  --model_path "$MODEL_DIR" \
  --iterations "$ITER"

# Symlink the chosen iteration point cloud into dataset_dir for PhysDreamer trainers
ln -sf "$MODEL_DIR/point_cloud/iteration_${ITER}/point_cloud.ply" "$DATASET_DIR/point_cloud.ply"
echo "Linked point_cloud.ply to $DATASET_DIR/point_cloud.ply"
