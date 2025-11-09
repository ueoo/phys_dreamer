import numpy as np


dataset_dir = "data_NeuROK_sim/flower_images"
result_dir = "./output/flower_results"
exp_name = "flower"

model_list = [
    "./output/flower_inverse_sim_whitebg_material/flower_inverse_sim_materialdecay_1.0_substep_768_se3_field_lr_0.001_tv_0.0001_iters_200_sw_6_cw_1/seed0/checkpoint_model_000059"
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
        "start_frame": "039.png",
        "end_frame": "101.png",
    },
    # real captured viewpoint
    {
        "type": "interpolation",
        "start_frame": "039.png",
    },
    # another viewpoint
    {
        "type": "interpolation",
        "start_frame": "101.png",
    },
]

simulate_cfg = {
    "substep": 768,
    "grid_size": 64,
    "init_young": 1e6,
    "downsample_scale": 0.1,  # downsample the points to speed up the simulation
}


points_list = [
    np.array([-0.003, -0.208, 0.276]),
    np.array([0.016, -0.001, 0.000]),
    np.array([-0.187, -0.055, 0.099]),
    np.array([-0.162, 0.238, 0.389]),
    np.array([0.246, 0.067, 0.246]),
    np.array([0.117, 0.367, 0.076]),
    np.array([-0.246, 0.407, -0.004]),
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
