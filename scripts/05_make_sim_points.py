#!/usr/bin/env python
import os
import argparse
import numpy as np

try:
    import point_cloud_utils as pcu
except Exception as e:
    raise RuntimeError("Please install point_cloud_utils: pip install point-cloud-utils") from e


def sphere_points(center, radius, num=5000):
    pts = np.random.normal(size=(num, 3))
    pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8)
    r = np.random.uniform(0, radius, size=(num, 1))
    return center + pts * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--move_center", nargs=3, type=float, required=True)
    ap.add_argument("--move_radius", type=float, default=0.05)
    ap.add_argument("--stem_center", nargs=3, type=float)
    ap.add_argument("--stem_radius", type=float, default=0.03)
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--fill_radius", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.dataset_dir, exist_ok=True)

    moving = sphere_points(np.array(args.move_center), args.move_radius, num=8000)
    pcu.save_mesh_v(os.path.join(args.dataset_dir, "moving_part_points.ply"), moving)
    print("Wrote moving_part_points.ply")

    if args.stem_center is not None:
        stem = sphere_points(np.array(args.stem_center), args.stem_radius, num=5000)
        pcu.save_mesh_v(os.path.join(args.dataset_dir, "stem_points.ply"), stem)
        print("Wrote stem_points.ply")

    if args.fill:
        fill = sphere_points(np.array(args.move_center), args.fill_radius, num=10000)
        pcu.save_mesh_v(os.path.join(args.dataset_dir, "internal_filled_points.ply"), fill)
        print("Wrote internal_filled_points.ply")


if __name__ == "__main__":
    main()
