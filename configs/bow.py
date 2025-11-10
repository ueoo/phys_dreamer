import numpy as np


dataset_dir = "data_NeuROK_sim/bow_images"
result_dir = "./output/bow_results"
exp_name = "bow"

model_list = [
    "./output/bow_inverse_sim_material/bow_material_traindecay_1.0_substep_768_se3_field_lr_0.001_tv_0.0001_iters_200_sw_6_cw_1/seed0/checkpoint_model_000099"
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
        "start_frame": "041.png",
        "end_frame": "144.png",
    },
    # real captured viewpoint
    {
        "type": "interpolation",
        "start_frame": "041.png",
    },
    # another viewpoint
    {
        "type": "interpolation",
        "start_frame": "144.png",
    },
]

simulate_cfg = {
    "substep": 768,
    "grid_size": 64,
    "init_young": 1e6,
    "downsample_scale": 0.1,  # downsample the points to speed up the simulation
}


points_list = [
    np.array([-0.315, -0.004, 0.001]),
    np.array([-0.001, -0.003, 0.492]),
    np.array([0.011, 0.000, -0.489]),
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
