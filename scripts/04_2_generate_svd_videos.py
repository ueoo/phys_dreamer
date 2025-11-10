import argparse
import os

import imageio
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--num_frames", type=int, default=25)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--gpu_num", type=int, default=1)
    args = ap.parse_args()

    device = "cuda"
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass
    pipe.set_progress_bar_config(disable=True)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    #### flower
    if "flower" in args.dataset_dir:
        input_frame_ids = [39, 40, 55, 59, 67, 71, 93, 103, 157, 166, 168]
    elif "bow" in args.dataset_dir:
        input_frame_ids = [17, 25, 33, 36, 41, 48, 120, 121, 129, 137, 144, 145, 152]
    elif "box" in args.dataset_dir:
        input_frame_ids = [65, 72, 80, 81, 88, 89, 97, 106, 158, 154]
    elif "cloth" in args.dataset_dir:
        input_frame_ids = [55, 59, 60, 61, 63, 67, 69, 72, 73, 75, 86, 88]
    elif "newton" in args.dataset_dir:
        input_frame_ids = [55, 58, 59, 63, 65, 70, 71, 78, 80, 87]
    elif "shirt" in args.dataset_dir:
        input_frame_ids = [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
    elif "lamp" in args.dataset_dir:
        input_frame_ids = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    elif "laptop" in args.dataset_dir:
        input_frame_ids = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 70]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_dir}")

    input_frame_ids = input_frame_ids[args.gpu_id :: args.gpu_num]
    img_paths = [os.path.join(args.dataset_dir, f"{frame_id:03d}.png") for frame_id in input_frame_ids]
    motion_bucket_ids = list(range(5, 11))

    for ip in tqdm(img_paths, desc=f"Generating videos for GPU {args.gpu_id}/{args.gpu_num}"):
        base = os.path.splitext(os.path.basename(ip))[0]
        img = Image.open(ip).convert("RGB")
        for motion_bucket_id in motion_bucket_ids:
            result = pipe(image=img, decode_chunk_size=8, motion_bucket_id=motion_bucket_id).frames[0]
            mp4_path = os.path.join(out_dir, f"{base}_motion_{motion_bucket_id}.mp4")
            export_to_video(result, mp4_path, fps=7)


if __name__ == "__main__":
    main()
