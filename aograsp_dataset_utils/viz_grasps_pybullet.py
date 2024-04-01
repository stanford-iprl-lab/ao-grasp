import argparse
import os
import sys
from os.path import join as pjoin
import numpy as np
import pybullet
from pybullet_utils import bullet_client

from aograsp.data_utils.viz_env import VizEnv
import aograsp.rotation_utils as r_utils


def viz_grasps(
    bc,
    state_path,
    vhacd=False,
    use_ag_ori=False,
    max_to_draw=np.inf,
):
    init_states = np.load(pjoin(state_path, "init_state.npz"), allow_pickle=True)[
        "data"
    ].item()
    env = VizEnv(bc, init_states["object"], draw_gripper=True, draw_point=True)

    grasp_data_dict = {}

    # Iterate through state_path/raw/pos directory, add each grasp to grasp_data_dict
    raw_data_dir = os.path.join(state_path, "raw/pos")

    i = 0
    for item in sorted(os.listdir(raw_data_dir)):
        if os.path.splitext(item)[1] != ".npz":
            continue

        # Load data.npz file and extract grasp data
        data_path = os.path.join(raw_data_dir, item)
        grasp_dict = np.load(data_path, allow_pickle=True)["data"].item()

        if use_ag_ori:
            target_ori = grasp_dict["after_grasp_quat_wf"]
        else:
            target_ori = grasp_dict["quat_wf"]

        data = {
            "target_pos": grasp_dict["pos_wf"],
            "start_ori": target_ori,
            "prob": 1.0,
            "n_proposal": i,
        }
        i += 1
        grasp_data_dict[i] = data
        if i >= max_to_draw:
            break

    env.reset(grasp_data_dict, state_path)
    while True:
        env.step()


def main(args):
    connection_mode = pybullet.GUI
    bc = bullet_client.BulletClient(
        connection_mode=connection_mode,
        options="--background_color_red=1 --background_color_blue=1 --background_color_green=1",
    )

    if args.max_pts_to_draw is None:
        max_to_draw = np.inf
    else:
        max_to_draw = args.max_pts_to_draw

    viz_grasps(
        bc,
        args.state_path,
        use_ag_ori=args.use_ag_ori,
        max_to_draw=max_to_draw,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_path", type=str, help="""path to state dir""")
    parser.add_argument(
        "--use_ag_ori", "-ag", action="store_true", help="Viz after-grasp ori"
    )
    parser.add_argument(
        "--max_pts_to_draw", "-m", type=int, help="""Max number of grasps to viz"""
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
