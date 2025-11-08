from datasets.multiview_dataset import MultiviewImageDataset
from datasets.multiview_video_dataset import MultiviewVideoDataset


def create_dataset(args):
    print(f"Creating dataset with args: {args.dataset_dir}, {args.video_dir}, {args.dataset_res}")
    assert args.dataset_res in ["middle", "small", "large"]
    if args.dataset_res == "middle":
        res = [320, 576]
    elif args.dataset_res == "small":
        res = [192, 320]
    elif args.dataset_res == "large":
        res = [576, 1024]
    else:
        raise NotImplementedError

    # scaled the raw images 512x512 to 1024x576
    dataset = MultiviewVideoDataset(
        args.dataset_dir,
        use_white_background=False,
        resolution=res,
        scale_x_angle=1.0,
        video_dir=args.video_dir,
    )

    test_dataset = MultiviewImageDataset(
        args.dataset_dir,
        use_white_background=False,
        resolution=res,
        # use_index=list(range(0, 30, 4)),
        # use_index=[0],
        scale_x_angle=1.0,
        fitler_with_renderd=False,
        load_imgs=False,
    )
    print("len of test dataset", len(test_dataset))
    return dataset, test_dataset



# def create_dataset(args):
#     assert args.dataset_res in ["middle", "small", "large"]
#     if args.dataset_res == "middle":
#         res = [320, 576]
#     elif args.dataset_res == "small":
#         res = [192, 320]
#     elif args.dataset_res == "large":
#         res = [576, 1024]
#     else:
#         raise NotImplementedError

#     video_dir_name = "videos"
#     video_dir_name = args.video_dir_name

#     if args.test_convergence:
#         video_dir_name = "simulated_videos"
#     dataset = MultiviewVideoDataset(
#         args.dataset_dir,
#         use_white_background=False,
#         resolution=res,
#         scale_x_angle=1.0,
#         video_dir_name=video_dir_name,
#     )

#     test_dataset = MultiviewImageDataset(
#         args.dataset_dir,
#         use_white_background=False,
#         resolution=res,
#         # use_index=list(range(0, 30, 4)),
#         # use_index=[0],
#         scale_x_angle=1.0,
#         fitler_with_renderd=False,
#         load_imgs=False,
#     )
#     print("len of test dataset", len(test_dataset))
#     return dataset, test_dataset
