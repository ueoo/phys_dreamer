import os

from shutil import copyfile


svd_all_videos_path = "data_NeuROK_sim/flower_svd_videos"

svd_selected_videos_path = "data_NeuROK_sim/flower_videos"

os.makedirs(svd_selected_videos_path, exist_ok=True)

video_files = [
    "039_svd_motion_6",
    "040_svd_motion_8",
    "055_svd_motion_5",
    "059_svd_motion_7",
    "067_svd_motion_9",
    "071_svd_motion_10",
    "093_svd_motion_9",
    "103_svd_motion_8",
    "157_svd_motion_8",
    "166_svd_motion_10",
    "168_svd_motion_8",
]

for video_file in video_files:
    base_name = video_file.split("_")[0]
    new_name = f"{base_name}.mp4"
    copyfile(
        os.path.join(svd_all_videos_path, f"{video_file}.mp4"),
        os.path.join(svd_selected_videos_path, f"{new_name}"),
    )
