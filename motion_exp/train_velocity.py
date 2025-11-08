import argparse
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_exp.trainer_velocity import Trainer
from utils.config import create_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./motion_exp/config.yml")

    # dataset params
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--video_dir", type=str, default="videos")

    parser.add_argument(
        "--dataset_res",
        type=str,
        default="large",  # ["middle", "small", "large"]
    )

    parser.add_argument("--model", type=str, default="se3_field")
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_hidden_size", type=int, default=64)
    parser.add_argument("--spatial_res", type=int, default=32)
    parser.add_argument("--zero_init", type=bool, default=True)
    parser.add_argument("--entropy_cls", type=int, default=0)

    parser.add_argument("--num_frames", type=str, default=14)

    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--sim_res", type=int, default=24)
    parser.add_argument("--sim_output_dim", type=int, default=1)
    parser.add_argument("--substep", type=int, default=96)
    parser.add_argument("--loss_decay", type=float, default=1.0)
    parser.add_argument("--start_window_size", type=int, default=2)
    parser.add_argument("--compute_window", type=int, default=2)
    parser.add_argument("--grad_window", type=int, default=14)

    parser.add_argument("--downsample_scale", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=8)

    # loss parameters
    parser.add_argument("--tv_loss_weight", type=float, default=1e-2)
    parser.add_argument("--ssim", type=float, default=0.5)

    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, default="./output/inverse_sim")
    parser.add_argument("--log_iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to load checkpoint from")
    # training parameters
    parser.add_argument("--train_iters", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # wandb parameters
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_entity", type=str, default="mit-cv")
    parser.add_argument("--wandb_project", type=str, default="inverse_sim")
    parser.add_argument("--wandb_iters", type=int, default=20)
    parser.add_argument("--wandb_name", type=str, default="flower_velo_pretrain")
    parser.add_argument("--run_eval", action="store_true", default=False)

    # distributed training args
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    args, extra_args = parser.parse_known_args()
    cfg = create_config(args.config, args, extra_args)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    print(args.local_rank, "local rank")

    return cfg


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)

    if args.run_eval:
        trainer.eval()
    else:
        # trainer.debug()
        trainer.train()
