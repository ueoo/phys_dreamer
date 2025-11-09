import json
import math
import os
import shutil

import cv2
import numpy as np

from tqdm import trange


def prepare_generative_image(
    in_path,
    out_path,
    width_new=1024,
    height_new=576,
    bg_color=[255, 255, 255],
    to_white_background=True,
):
    # Original dimensions (keep alpha channel if present)
    input_im = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    height_original, width_original = input_im.shape[:2]

    # Calculating the ratio of new dimensions
    ratio_width = width_new / width_original
    ratio_height = height_new / height_original

    # Choosing the smallest ratio to maintain aspect ratio
    ratio = min(ratio_width, ratio_height)

    # Calculating new dimensions
    new_width = int(width_original * ratio)
    new_height = int(height_original * ratio)

    # Resizing the image
    resized_image = cv2.resize(input_im, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Compute top-left corner for centered placement (letterbox)
    top = (height_new - new_height) // 2
    left = (width_new - new_width) // 2

    # Prepare background canvas
    if to_white_background:
        canvas = np.full((height_new, width_new, 3), bg_color, dtype=np.uint8)
    else:
        canvas = np.zeros((height_new, width_new, 3), dtype=np.uint8)

    # Composite/paste resized image onto canvas
    if resized_image.ndim == 3 and resized_image.shape[2] == 4:
        # Alpha composite onto background
        bgr = resized_image[..., :3].astype(np.float32)
        alpha = resized_image[..., 3:4].astype(np.float32) / 255.0
        roi = canvas[top : top + new_height, left : left + new_width].astype(np.float32)
        comp = bgr * alpha + roi * (1.0 - alpha)
        canvas[top : top + new_height, left : left + new_width] = comp.astype(np.uint8)
    else:
        # Ensure 3 channels
        if resized_image.ndim == 2 or (resized_image.ndim == 3 and resized_image.shape[2] == 1):
            bgr = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = resized_image
        canvas[top : top + new_height, left : left + new_width] = bgr

    # Save final composite (3-channel, white background where transparent)
    cv2.imwrite(out_path, canvas)


def update_blender_transforms_fov(transforms_path, ratio_canvas_to_content):
    if not os.path.exists(transforms_path):
        return
    with open(transforms_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    old = None
    for fr in frames:
        if "camera_angle_x" in fr:
            old = fr["camera_angle_x"]
            break
    if old is None:
        old = data.get("camera_angle_x", None)
    if old is None:
        return

    new_fovx = 2.0 * math.atan(ratio_canvas_to_content * math.tan(old / 2.0))

    if "camera_angle_x" in data:
        data["camera_angle_x"] = new_fovx
    for fr in frames:
        if "camera_angle_x" in fr:
            fr["camera_angle_x"] = new_fovx

    with open(transforms_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


if __name__ == "__main__":
    data_root = "data_NeuROK_sim/"
    secen_name = "cloth"
    out_scene_name = f"{secen_name}_images"
    out_dir = os.path.join(data_root, out_scene_name)
    os.makedirs(out_dir, exist_ok=True)

    width_new = 1024
    height_new = 576

    # Compute letterbox ratio (canvas/content width) from the first frame
    sample_in_path = os.path.join(data_root, secen_name, f"{0:03d}.png")
    sample = cv2.imread(sample_in_path)
    if sample is not None:
        h0, w0 = sample.shape[:2]
        scale = min(width_new / float(w0), height_new / float(h0))
        content_w = int(w0 * scale)
        ratio_canvas_to_content = width_new / float(content_w)
    else:
        ratio_canvas_to_content = 1024.0 / 576.0

    print(f"ratio_canvas_to_content: {ratio_canvas_to_content}")

    for frame_idx in trange(220):
        in_path = os.path.join(data_root, secen_name, f"{frame_idx:03d}.png")
        out_path = os.path.join(out_dir, f"{frame_idx:03d}.png")
        prepare_generative_image(
            in_path,
            out_path,
            width_new=width_new,
            height_new=height_new,
            to_white_background=True,
        )

    json_files = os.listdir(os.path.join(data_root, secen_name))
    json_files = [file for file in json_files if file.endswith(".json")]
    for json_file in json_files:
        in_path = os.path.join(data_root, secen_name, json_file)
        out_path = os.path.join(out_dir, json_file)
        shutil.copy(in_path, out_path)

    # Update horizontal FOV in Blender-style transforms files to match 576x1024 letterbox
    for name in ["transforms_train.json", "transforms_test.json", "transforms.json"]:
        update_blender_transforms_fov(os.path.join(out_dir, name), ratio_canvas_to_content)
