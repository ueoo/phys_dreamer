#!/usr/bin/env bash
{
set -euo pipefail

data_root="data_NeuROK_sim"
DATASET_DIR="$data_root/laptop_images"
OUTPUT_DIR="$data_root/laptop_svd_videos"

python "$(dirname "$0")/04_generate_svd_videos.py" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR"

exit 0
}
