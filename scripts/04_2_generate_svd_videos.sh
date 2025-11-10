#!/usr/bin/env bash
{
set -euo pipefail

data_root="data_NeuROK_sim"
DATASET_DIR="$data_root/laptop_images"
OUTPUT_DIR="$data_root/laptop_svd_videos"

GPU_COUNT=8
for gpu_id in $(seq 0 $((GPU_COUNT-1))); do
CUDA_VISIBLE_DEVICES="$gpu_id" python "$(dirname "$0")/04_2_generate_svd_videos.py" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --gpu_id "$gpu_id" \
  --gpu_num "$GPU_COUNT" &
echo "[launched] gpu=$gpu_id → $OUTPUT_DIR"
done

wait
echo "[all done] launched $GPU_COUNT jobs for $DATASET_DIR"

exit 0
}
