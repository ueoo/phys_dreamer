import os
import shutil

from tqdm import tqdm


# scene_names = ["cloth", "flower", "newton", "box", "bow", "shirt", "lamp"]
scene_names = ["laptop"]


for scene_name in tqdm(scene_names):
    data_root = f"data_NeuROK_sim/{scene_name}_gs_iteration_30k"
    images_dataset_dir = f"data_NeuROK_sim/{scene_name}_images"

    shutil.copy(f"{data_root}/point_cloud.ply", f"{images_dataset_dir}/point_cloud.ply")
    shutil.copy(f"{data_root}/moving_points.ply", f"{images_dataset_dir}/moving_part_points.ply")
    if os.path.exists(f"{data_root}/stem.ply"):
        shutil.copy(f"{data_root}/stem.ply", f"{images_dataset_dir}/stem_points.ply")
