import os
import sys

import torch

from datasets.multiview_video_dataset import MultiviewVideoDataset


if __name__ == "__main__":
    # dataset_dir = "data/multiview/dragon/merged"
    # dataset_dir = "/tmp/tmp_tyz_data/ficus/"

    dataset_dir = "data_NeuROK_sim/flower_images"
    video_dir = "data_NeuROK_sim/flower_videos"

    dataset = MultiviewVideoDataset(
        dataset_dir,
        video_dir,
    )

    data = dataset[0]

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            print(key, val.shape)
        else:
            print(key, type(val))
