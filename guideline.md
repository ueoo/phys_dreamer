I'll scan the core files to map the dataset format, camera conventions, and training configs before drafting the step-by-step training instructions and scripts.

[5 tools called]

Now I'll scan for how cameras and datasets are loaded, check the Gaussian module README, and identify where MPM and video priors are integrated.

[31 tools called]

I'll outline the code structure, data formats/conventions, typical issues, and then give you end-to-end, executable steps and shell scripts to train on your images (3DGS → SVD videos → velocity pretrain → MPM/material training → demos).

### Paper context
- PhysDreamer distills dynamics priors from a video generation model to infer spatially-varying material and an initial velocity field over a 3D Gaussian representation, then uses a differentiable MPM to simulate interactions and render from arbitrary views. See the paper for the full approach: [PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation](https://arxiv.org/pdf/2404.13026).

### Codebase overview (what lives where)
- 3D Gaussians + renderer:
  - `phys_dreamer/gaussian_3d/` is a light copy of the official 3D Gaussian Splatting project, with minor additions for force rendering (README notes the upstream):
    ```1:3:._fork/physdreamer/gaussian_3d/README.md
    This folder is mainly a copy paste from https://github.com/graphdeco-inria/gaussian-splatting

    We add some function to render the applied external force.
    ```
  - Cameras and scene loading for COLMAP/Blender formats live in `gaussian_3d/scene/`, where COLMAP `sparse/0` or `transforms_train.json` is parsed and a `cameras.json` is emitted for the model path. It can also export `point_cloud.ply` per iteration for reuse:
    ```36:76:./gaussian_3d/scene/__init__.py
    class Scene:
        ...
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        ...
        if not self.loaded_iter:
            ...
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)
        ...
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud","iteration_"+str(self.loaded_iter),"point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
    ```
- Data and camera utilities:
  - `datasets/cameras.py` defines a `Camera` with FoVx/FoVy, world-view/projection transforms, and interpolation; this encodes the camera math used across the project:
    ```12:76:./datasets/cameras.py
    class Camera(nn.Module):
        def __init__( self, R: np.ndarray, T: np.ndarray, FoVx, FoVy, img_path, trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0, data_device="cuda", img_hw: Tuple[int, int] = (800, 800), timestamp: float = 0.0, ):
            ...
            self.world_view_transform = (torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device))
            self.projection_matrix = (getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).to(self.data_device))
            ...
            self.cam_plane_2_img = torch.tensor([
                [0.5 * self.image_width / math.tan(self.FoVx / 2.0), 0.0],
                [0.0, 0.5 * self.image_height / math.tan(self.FoVy / 2.0)],
            ]).to(self.data_device)
    ```
  - Dataset (images + videos) loader: `datasets/multiview_video_dataset.py` supports COLMAP (`sparse/0`) or Blender-style `transforms_train.json`. It expects a `videos/` (or configurable) folder containing either mp4s (named with a prefix matching image names) or a folder of PNG frames:
    ```114:132:./datasets/multiview_video_dataset.py
    class MultiviewVideoDataset(Dataset):
        def __init__( self, data_dir, use_white_background=True, resolution=[576, 1024], scale_x_angle=1.0, use_index=None, video_dir_name="videos",):
            self.data_dir = data_dir
            self.video_dir = os.path.join(data_dir, video_dir_name)
            self._parse_dataset(data_dir)
    ```
    ```220:264:./datasets/multiview_video_dataset.py
    def _parse_video_names(self, camera_list, video_dir):
        video_names = [_ for _ in os.listdir(video_dir) if _.endswith(".mp4")]
        if len(video_names) > 0:
            # match mp4s by image name prefix
            ...
            return video_name_list_list, ret_camera_list, "FormatInVideo"
        else:
            # or stitch PNG frames into a video tensor
            img_names = [_ for _ in os.listdir(video_dir) if _.endswith(".png")]
            assert len(img_names) > 0, "no images or videos found in video_dir!"
            ...
            return ret_video_list, ret_camera_list, "FormatInImage"
    ```
- Diffusion video priors:
  - Stable Video Diffusion integration is under `diffusion/` with `sv_diffusion_engine.py` and helpers in `utils/svd_helpper.py` (loads `sgm` configs/ckpts). You don’t have to use these to generate videos; any SVD pipeline that outputs mp4s is fine, as long as you place them where the dataset loader expects.
- MPM and training:
  - Velocity pretraining: `motion_exp/fast_train_velocity.py` (uses 3DGS + k-means downsampling + Warp MPM; learns an initial velocity field) with CLI in the file:
    ```1040:1123:./motion_exp/fast_train_velocity.py
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_dir", type=str, default=".../alocasia_nerfstudio")
        parser.add_argument("--num_frames", type=str, default=14)
        parser.add_argument("--grid_size", type=int, default=32)
        parser.add_argument("--substep", type=int, default=96)
        parser.add_argument("--downsample_scale", type=float, default=0.1)
        ...
        parser.add_argument("--output_dir", type=str, default="../../output/inverse_sim")
        parser.add_argument("--wandb_name", type=str, required=True)
    ```
    It expects a 3DGS point cloud at `dataset_dir/point_cloud.ply` and (optionally) additional PLYs that define constraints (see Data format below).
  - Material training: `motion_exp/train_material.py` learns a spatial material field, conditioning on short videos in `dataset_dir/<video_dir_name>`. It also expects `dataset_dir/point_cloud.ply` and can load the velocity pretrain checkpoint:
    ```1320:1409:./motion_exp/train_material.py
    def parse_args():
        parser.add_argument("--dataset_dir", type=str, default=".../hat_nerfstudio/")
        parser.add_argument("--video_dir_name", type=str, default="videos")
        parser.add_argument("--checkpoint_path", type=str, default=".../checkpoint_model_000299", help="path to load velocity pretrain checkpoint from")
        parser.add_argument("--grid_size", type=int, default=64)
        parser.add_argument("--substep", type=int, default=768)
        parser.add_argument("--downsample_scale", type=float, default=0.04)
        ...
    ```

### Data format and camera conventions
- Dataset directory (`dataset_dir`):
  - Either COLMAP layout:
    - `images/` (all RGB images used by 3DGS)
    - `sparse/0/{images.(bin|txt), cameras.(bin|txt), points3D.(bin|txt)}`
  - Or Blender layout:
    - `transforms_train.json` (+ optional `transforms_test.json`) with fields `camera_angle_x` and `frames[i].transform_matrix` (camera-to-world).
  - Videos for dynamics supervision:
    - `dataset_dir/videos/` (default) containing either:
      - mp4s with names starting with the corresponding image name prefix (the loader matches by prefix, first 4 chars must also match), or
      - a list of PNG frames; in that case the folder name must start with the image name of the intended viewpoint.
- Cameras:
  - Intrinsics: FoVx/FoVy computed from focal and image resolution (`fov2focal` / `focal2fov`).
  - Extrinsics: OpenGL/Blender c2w are converted to COLMAP convention by flipping Y/Z, then inverted to w2c; rotation is stored transposed for the CUDA renderer.
  - See the exact conversions:
    ```172:192:./datasets/cameras.py
    def interpolate_cameras(R1, t1, R2, t2, steps=10):
        # quaternions slerp + linear T
    ```
    ```235:287:./gaussian_3d/scene/dataset_readers.py
    # Blender transforms JSON → R, T, FoVx, FoVy, image path
    c2w = np.array(frame["transform_matrix"])
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]
    ```
- Additional PLYs for simulation (all looked for under `dataset_dir`):
  - `point_cloud.ply`: 3DGS point cloud used by MPM training.
  - `moving_part_points.ply` (required): defines the “allowed-to-move” region; everything far from these points is frozen as boundary condition.
  - `stem_points.ply` (optional): marks denser subregions (e.g., a stem).
  - `internal_filled_points.ply` (optional): additional interior points to fill the object for simulation stability.
  - Where these are used:
    ```331:356:./motion_exp/fast_train_velocity.py
    filled_in_points_path = os.path.join(dataset_dir, "internal_filled_points.ply")
    ...
    moving_pts_path = os.path.join(dataset_dir, "moving_part_points.ply")
    if os.path.exists(moving_pts_path):
        moving_pts = pcu.load_mesh_v(moving_pts_path)
        ...
        freeze_mask = find_far_points(sim_xyzs, moving_pts, thres=0.25 / grid_size).bool()
    else:
        raise NotImplementedError
    ...
    stem_pts_path = os.path.join(dataset_dir, "stem_points.ply")
    if os.path.exists(stem_pts_path):
        ...
        density[stem_mask] = 2000
    ```

### Inputs/outputs of the training scripts
- Inputs:
  - `--dataset_dir`: the dataset folder described above.
  - `--video_dir_name`: subfolder containing mp4s/pngs (“videos”, “videos_2”, etc.).
  - `--checkpoint_path` (material training): velocity pretrain checkpoint (it will look for `model.pt` in that folder).
  - Optional knobs: `--grid_size`, `--substep`, `--downsample_scale`, `--num_frames`, etc.
- Outputs:
  - Checkpoints in `--output_dir/<exp>/<seed>/`: `model.pt` per component (velocity/material fields).
  - GIFs/videos for qualitative inspection.
  - You can point `demo.py` to a config scene and a checkpoint to render novel-view interactive sequences.

### Known issues in the original repo (typical pitfalls)
- Dependencies/build:
  - Building `diff-gaussian-rasterization` and matching CUDA/PyTorch versions is crucial; mismatches lead to compile/load errors.
  - NVIDIA Warp (used for MPM) depends on GPU driver/compute capability; ensure a compatible Warp version and CUDA toolchain.
- OOM during KMeans downsampling on GPU:
  - Switch to the CPU implementation as advised in `README_infer.md` if you hit OOM; flip to `downsample_with_kmeans` (CPU) instead of the GPU variant:
    ```11:21:./README_infer.md
    # uncomment CPU version in inference.py to avoid GPU OOM
    ```
- SVD weights/paths:
  - Make sure the SVD config and checkpoints exist and paths are correct if you use the internal `diffusion/` engines; otherwise use Diffusers’ SVD pipelines to generate mp4s.
- Path assumptions:
  - The trainers expect `dataset_dir/point_cloud.ply` and the `videos/` folder naming conventions. Symlink/copy accordingly.
- Camera mismatches:
  - Ensure all images are same resolution or set `resolution=[H,W]` consistently in loaders; incorrect FOV or aspect assumptions can blur/offset supervision.

You can browse the issue tracker for more context and setup pitfalls: `https://github.com/a1600012888/PhysDreamer/issues`

### Step-by-step: train on your images (from static 3D)
Below are minimal, runnable scripts. Adjust absolute paths to your environment.

1) Create environment and install deps
- Requires: CUDA-compatible PyTorch, `diff-gaussian-rasterization`, NVIDIA Warp, decord (for mp4), diffusers if using external SVD.

```bash
# 00_setup_env.sh
#!/usr/bin/env bash
set -euo pipefail

# Create env
conda create -y -n physdreamer python=3.10
conda activate physdreamer

# PyTorch (example; match your CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core requirements
pip install -r ./requirements.txt

# Build diff-gaussian-rasterization (follow their readme)
# https://github.com/graphdeco-inria/diff-gaussian-rasterization
# Example:
# git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
# cd diff-gaussian-rasterization && pip install .

# NVIDIA Warp (match CUDA/driver)
pip install warp-lang

# Video IO + optional SVD (diffusers route)
pip install imageio[ffmpeg] decord diffusers transformers accelerate safetensors
```

2) Prepare dataset_dir with cameras
- If you already have COLMAP outputs for your renders, place them as:
  - `dataset_dir/images/*.png`
  - `dataset_dir/sparse/0/{cameras.bin|txt, images.bin|txt, points3D.bin|txt}`
- If you have known camera-to-worlds, create `dataset_dir/transforms_train.json` (Blender format). The loader handles both.

3) Train 3D Gaussian Splatting (3DGS)
- Use the official GS repo to train, then symlink the resulting `point_cloud.ply` into your dataset_dir.

```bash
# 02_train_3dgs.sh
#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="/abs/path/to/your_scene"     # has images/ + sparse/0 or transforms_train.json
MODEL_DIR="/abs/path/to/gs_models/your_scene"

mkdir -p "$MODEL_DIR"

# Example running upstream GS training (adjust to your runner)
# python train.py --source_path "$DATASET_DIR" --model_path "$MODEL_DIR" --iterations 30000

# After training, link point cloud into dataset_dir
# Pick your desired iteration
ITER="30000"
ln -sf "$MODEL_DIR/point_cloud/iteration_${ITER}/point_cloud.ply" "$DATASET_DIR/point_cloud.ply"
```

4) Generate videos with Stable Video Diffusion (external, via diffusers)
- Create short, neutral videos from one or a few reference images. Place mp4s into `dataset_dir/videos/` with filenames starting with the source image name (e.g., `frame_00037_*.mp4`).

```bash
# 04_generate_svd_videos.py
#!/usr/bin/env python
import os, argparse, glob
from PIL import Image
import torch
from diffusers import StableVideoDiffusionPipeline
import imageio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--image_glob", default="images/frame_*.png")
    ap.add_argument("--out_dirname", default="videos")
    ap.add_argument("--num_frames", type=int, default=25)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    device = "cuda"
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16
    ).to(device)
    pipe.enable_model_cpu_offload()

    img_paths = sorted(glob.glob(os.path.join(args.dataset_dir, args.image_glob)))
    out_dir = os.path.join(args.dataset_dir, args.out_dirname)
    os.makedirs(out_dir, exist_ok=True)

    for ip in img_paths:
        base = os.path.splitext(os.path.basename(ip))[0]
        img = Image.open(ip).convert("RGB")
        video_frames = pipe(image=img, num_frames=args.num_frames).frames[0]  # list of PIL
        mp4_path = os.path.join(out_dir, f"{base}_svd.mp4")
        imageio.mimwrite(mp4_path, [f for f in video_frames], fps=args.fps)
        print("Wrote", mp4_path)

if __name__ == "__main__":
    main()
```

```bash
# 04_run_generate_videos.sh
#!/usr/bin/env bash
set -euo pipefail
DATASET_DIR="/abs/path/to/your_scene"
python 04_generate_svd_videos.py --dataset_dir "$DATASET_DIR" --image_glob "images/frame_00037.png"
```

5) Create simulation helper PLYs (moving/stem/internal)
- Minimal generator that creates a small ball of points around an interactable location (moving region). Adjust center and radius as desired.

```bash
# 05_make_sim_points.py
#!/usr/bin/env python
import os, argparse
import numpy as np
import point_cloud_utils as pcu

def sphere_points(center, radius, num=5000):
    pts = np.random.normal(size=(num,3))
    pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True)+1e-8)
    r = np.random.uniform(0, radius, size=(num,1))
    return center + pts * r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--move_center", nargs=3, type=float, required=True)  # approx in world coords
    ap.add_argument("--move_radius", type=float, default=0.05)
    ap.add_argument("--stem_center", nargs=3, type=float)
    ap.add_argument("--stem_radius", type=float, default=0.03)
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--fill_radius", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.dataset_dir, exist_ok=True)

    moving = sphere_points(np.array(args.move_center), args.move_radius, num=8000)
    pcu.save_mesh_v(os.path.join(args.dataset_dir,"moving_part_points.ply"), moving)

    if args.stem_center is not None:
        stem = sphere_points(np.array(args.stem_center), args.stem_radius, num=5000)
        pcu.save_mesh_v(os.path.join(args.dataset_dir,"stem_points.ply"), stem)

    if args.fill:
        fill = sphere_points(np.array(args.move_center), args.fill_radius, num=10000)
        pcu.save_mesh_v(os.path.join(args.dataset_dir,"internal_filled_points.ply"), fill)

if __name__ == "__main__":
    main()
```

```bash
# 05_run_make_sim_points.sh
#!/usr/bin/env bash
set -euo pipefail
DATASET_DIR="/abs/path/to/your_scene"

# Approximate a point on the object that should remain free (others frozen)
python 05_make_sim_points.py --dataset_dir "$DATASET_DIR" \
  --move_center -0.39 0.14 -0.18 \
  --move_radius 0.06 \
  --fill
```

6) Velocity pretrain
- Trains initial velocity field against short videos.

```bash
# 06_pretrain_velocity.sh
#!/usr/bin/env bash
set -euo pipefail

PY=./motion_exp/fast_train_velocity.py
DATASET_DIR="/abs/path/to/your_scene"
OUT_DIR="/abs/path/to/output/inverse_sim"
WANDB_NAME="your_scene_velo_pretrain"

python "$PY" \
  --dataset_dir "$DATASET_DIR" \
  --video_dir_name "videos" \
  --num_frames 14 \
  --grid_size 32 \
  --substep 96 \
  --downsample_scale 0.1 \
  --output_dir "$OUT_DIR" \
  --wandb_name "$WANDB_NAME"
```

7) Material training (MPM) with videos
- Uses the videos and (optionally) the velocity checkpoint above.

```bash
# 07_train_material.sh
#!/usr/bin/env bash
set -euo pipefail

PY=./motion_exp/train_material.py
DATASET_DIR="/abs/path/to/your_scene"
OUT_DIR="/abs/path/to/output/inverse_sim"

# Point to velocity checkpoint folder (containing model.pt)
VELO_CKPT="/abs/path/to/output/inverse_sim/<exp>/seed0/checkpoint_model_000199"

python "$PY" \
  --dataset_dir "$DATASET_DIR" \
  --video_dir_name "videos" \
  --checkpoint_path "$VELO_CKPT" \
  --num_frames 14 \
  --grid_size 64 \
  --substep 384 \
  --downsample_scale 0.06 \
  --output_dir "$OUT_DIR" \
  --wandb_name "your_scene_material"
```

8) Demos and rendering
- Use `configs/<scene>.py`-style params or set via `config_demo.py`; then call `demo.py`.

```bash
# 08_demo.sh
#!/usr/bin/env bash
set -euo pipefail

PY=./demo.py
SCENE_CFG="hat"  # or "alocasia", "carnation", etc. (adjust your dataset paths in the config file accordingly)
OUT_DIR="/abs/path/to/output/inverse_sim"

python "$PY" \
  --scene_name "$SCENE_CFG" \
  --dataset_dir "/abs/path/to/your_scene" \
  --output_dir "$OUT_DIR" \
  --run_eval
```

Notes and tips:
- If you hit GPU OOM at k-means downsampling, switch to the CPU downsampler as advised in `README_infer.md`.
- Make sure `dataset_dir/point_cloud.ply` exists (symlink to your trained 3DGS iteration).
- If you prefer COLMAP-less setup and you have camera-to-worlds, you can generate `transforms_train.json` (the loader handles that path, as shown above).

### References
- Paper: [PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation](https://arxiv.org/pdf/2404.13026)
- Issues tracker: `https://github.com/a1600012888/PhysDreamer/issues`

I’ve mapped the loaders, the training entry points, and provided scripts you can run. If you tell me your dataset path and image naming, I can pre-fill the scripts with exact values and suggest the moving region center for `moving_part_points.ply`.

- I inspected `datasets/cameras.py`, `gaussian_3d/scene/*`, `datasets/multiview_video_dataset.py`, and the training CLIs in `motion_exp/fast_train_velocity.py` and `motion_exp/train_material.py`.
- Next, you can execute the provided scripts in order: environment, 3DGS training + symlink, SVD video generation, simulation PLYs, velocity pretrain, material training, then demos.
