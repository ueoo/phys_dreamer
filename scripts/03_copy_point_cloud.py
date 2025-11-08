import os
import shutil


scene_names = ["cloth", "flower", "newton", "box", "bow"]


for scene_name in scene_names:
    data_root = f"./data_NeuROK_sim/{scene_name}_gs_iteration_30k"
    point_cloud_path = f"{data_root}/point_cloud.ply"
    moving_particles_path = f"{data_root}/moving_points.ply"

    images_dataset_dir = f"./data_NeuROK_sim/{scene_name}_images"

    shutil.copy(point_cloud_path, f"{images_dataset_dir}/point_cloud.ply")
    shutil.copy(moving_particles_path, f"{images_dataset_dir}/moving_part_points.ply")
