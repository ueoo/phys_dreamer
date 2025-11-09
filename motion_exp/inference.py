import argparse
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_exp.trainer_inference import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="se3_field")
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_hidden_size", type=int, default=64)
    # resolution of velocity fields
    parser.add_argument("--spatial_res", type=int, default=32)
    parser.add_argument("--zero_init", type=bool, default=True)

    parser.add_argument("--num_frames", type=str, default=14)

    # resolution of material fields
    parser.add_argument("--sim_res", type=int, default=8)
    parser.add_argument("--sim_output_dim", type=int, default=1)

    parser.add_argument("--downsample_scale", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=8)

    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, default="output/inverse_sim")
    parser.add_argument("--seed", type=int, default=0)

    # demo parameters. related to parameters specified in configs/{scene_name}.py
    parser.add_argument("--scene_name", type=str, default="carnation")
    parser.add_argument("--demo_name", type=str, default="inference_demo")
    parser.add_argument("--model_id", type=int, default=0)

    # if eval_ys > 10. Then all the youngs modulus is set to eval_ys homogeneously
    parser.add_argument("--eval_ys", type=float, default=1.0)
    parser.add_argument("--force_id", type=int, default=1)
    parser.add_argument("--force_mag", type=float, default=1.0)
    parser.add_argument("--velo_scaling", type=float, default=5.0)
    parser.add_argument("--point_id", type=int, default=0)
    parser.add_argument("--apply_force", action="store_true", default=False)
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--static_camera", action="store_true", default=False)

    args, extra_args = parser.parse_known_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)

    trainer.demo(
        velo_scaling=args.velo_scaling,
        eval_ys=args.eval_ys,
        static_camera=args.static_camera,
        apply_force=args.apply_force,
        save_name=args.demo_name,
    )
