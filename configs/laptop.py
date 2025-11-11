import numpy as np


dataset_dir = "data_NeuROK_sim/laptop_images"
result_dir = "./output/laptop_results"
exp_name = "laptop"

model_list = [
    "./output/laptop_inverse_sim_material/laptop_material_traindecay_1.0_substep_768_se3_field_lr_0.001_tv_0.0001_iters_200_sw_6_cw_1/seed0/checkpoint_model_000059"
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
        "start_frame": "056.png",
        "end_frame": "051.png",
    },
    # real captured viewpoint
    {
        "type": "interpolation",
        "start_frame": "056.png",
    },
    # another viewpoint
    {
        "type": "interpolation",
        "start_frame": "051.png",
    },
]

simulate_cfg = {
    "substep": 768,
    "grid_size": 64,
    "init_young": 1e6,
    "downsample_scale": 0.1,  # downsample the points to speed up the simulation
}


points_list = [
    np.array([0.001, 0.329, 0.333]),
    np.array([0.488, 0.325, 0.047]),
    np.array([-0.483, 0.322, 0.051]),
]

force_directions = [
    np.array([1.0, 0.0, 0]),
    np.array([-1.0, 0.0, 0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([1.0, 0.0, 1.0]),
    np.array([1.0, 1.0, 0.0]),
    np.array([1.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 1.0]),
    np.array([1.0, 1.0, 1.0]),
]

force_directions = np.array(force_directions)
force_directions = force_directions / np.linalg.norm(force_directions, axis=1)[:, None]
