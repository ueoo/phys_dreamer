import os

from shutil import copyfile


# scene_names = ["flower", "bow", "box", "cloth", "newton", "shirt", "lamp"]
scene_names = ["laptop"]
for scene_name in scene_names:
    svd_all_videos_path = f"data_NeuROK_sim/{scene_name}_svd_videos"

    svd_selected_videos_path = f"data_NeuROK_sim/{scene_name}_videos"

    os.makedirs(svd_selected_videos_path, exist_ok=True)

    if "flower" == scene_name:
        video_files = [
            "039_motion_6",
            "040_motion_6",
            "055_motion_5",
            "059_motion_8",
            "067_motion_9",
            "071_motion_7",
            "093_motion_6",
            "103_motion_8",
            "157_motion_8",
            "166_motion_10",
            "168_motion_8",
        ]
    elif "bow" == scene_name:
        video_files = [
            "017_motion_5",
            "025_motion_10",
            "033_motion_10",
            "036_motion_5",
            "041_motion_10",
            "048_motion_9",
            "120_motion_9",
            "121_motion_7",
            "129_motion_9",
            "144_motion_5",
            "145_motion_10",
        ]
    elif "box" == scene_name:
        video_files = [
            "065_motion_5",
            "072_motion_6",
            "080_motion_5",
            "081_motion_7",
            "088_motion_7",
            "089_motion_6",
            "097_motion_8",
            "154_motion_10",
            "158_motion_7",
        ]
    elif "cloth" == scene_name:
        video_files = [
            "055_motion_5",
            "059_motion_10",
            "060_motion_9",
            "061_motion_10",
            "063_motion_10",
            "067_motion_10",
            "069_motion_6",
            "072_motion_7",
            "073_motion_5",
            "075_motion_7",
            "086_motion_7",
            "088_motion_10",
        ]
    elif "newton" == scene_name:
        video_files = [
            "055_motion_7",
            "058_motion_7",
            "059_motion_7",
            "063_motion_10",
            "065_motion_10",
            "070_motion_7",
            "071_motion_7",
            "078_motion_8",
            "080_motion_8",
            "087_motion_9",
        ]
    elif "shirt" == scene_name:
        video_files = [
            "052_motion_8",
            "053_motion_9",
            "054_motion_5",
            "055_motion_8",
            "056_motion_7",
            "057_motion_10",
            "058_motion_5",
            "059_motion_7",
            "060_motion_7",
            "061_motion_5",
            "062_motion_8",
            "063_motion_8",
        ]
    elif "lamp" == scene_name:
        video_files = [
            "040_motion_5",
            "041_motion_9",
            "042_motion_6",
            "043_motion_6",
            "044_motion_7",
            "045_motion_7",
            "047_motion_10",
            "048_motion_6",
            "049_motion_10",
            "050_motion_7",
            "051_motion_6",
        ]
    elif "laptop" == scene_name:
        video_files = [
            "049_motion_5",
            "050_motion_6",
            "051_motion_6",
            "052_motion_7",
            "053_motion_8",
            "054_motion_7",
            "055_motion_9",
            "056_motion_6",
            "057_motion_10",
            "058_motion_6",
            "059_motion_8",
            "060_motion_8",
            "070_motion_7",
        ]
    else:
        raise ValueError(f"Unknown dataset: {svd_all_videos_path}")

    for video_file in video_files:
        base_name = video_file.split("_")[0]
        new_name = f"{base_name}.mp4"
        copyfile(
            os.path.join(svd_all_videos_path, f"{video_file}.mp4"),
            os.path.join(svd_selected_videos_path, f"{new_name}"),
        )
