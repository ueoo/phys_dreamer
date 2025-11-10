import numpy as np


dataset_dir = "data_NeuROK_sim/box_images_infer"
result_dir = "./output/box_results"
exp_name = "box"

model_list = [
    "./output/box_inverse_sim_material/box_material_traindecay_1.0_substep_768_se3_field_lr_0.001_tv_0.0001_iters_200_sw_6_cw_1/seed0/checkpoint_model_000099"
]

focus_point_list = [
    np.array([0.0, 0.0, 0.0]),  # botton of the background
]

camera_cfg_list = [
    {
        "type": "spiral",
        "focus_point": focus_point_list[0],
        "radius": 1.0,
        "up": np.array([0, 0, 1]),
    },
    {
        "type": "interpolation",
        "start_frame": "065.png",
        "end_frame": "097.png",
    },
    # real captured viewpoint
    {
        "type": "interpolation",
        "start_frame": "065.png",
    },
    # another viewpoint
    {
        "type": "interpolation",
        "start_frame": "097.png",
    },
    {
        "type": "interpolation",
        "start_frame": "000.png",
    },
]

simulate_cfg = {
    "substep": 768,
    "grid_size": 64,
    "init_young": 1e6,
    "downsample_scale": 0.1,  # downsample the points to speed up the simulation
}


points_list = [
    np.array([0.000, 0.204, 0.281]),
    np.array([0.484, 0.205, 0.281]),
    np.array([-0.485, 0.205, 0.284]),
    np.array([0.486, 0.002, 0.177]),
    np.array([-0.486, 0.003, 0.191]),
]

force_directions = [
    np.array([1.0, 0.0, 0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([1.0, 0.0, 1.0]),
    np.array([1.0, 1.0, 0.0]),
    np.array([1.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 1.0]),
    np.array([1.0, 1.0, 1.0]),
]

force_directions = np.array(force_directions)
force_directions = force_directions / np.linalg.norm(force_directions, axis=1)[:, None]
