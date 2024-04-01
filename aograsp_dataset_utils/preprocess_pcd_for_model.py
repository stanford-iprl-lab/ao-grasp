"""
Script for pre-processing the AO-Grasp dataset partial point clouds for the AO-Grasp model
- Downsample point cloud to 4096 points
- Transform from world frame to camera frame (z-front frame)
"""

import argparse
import numpy as np
import os
import open3d as o3d

import aograsp.viz_utils as v_utils
import aograsp.rotation_utils as r_utils
import aograsp.data_utils.dataset_utils as d_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "pc_file",
    type=str,
    help="Path to point_cloud_seg.npz",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="test_data/synthetic",
    help="Path to directory to write output files",
)
parser.add_argument(
    "--visualize",
    "-v",
    action="store_true",
    help="Use this flag to visualize point cloud in o3d"
)
args = parser.parse_args()

pcd_dict = np.load(args.pc_file, allow_pickle=True)["data"].item()

# Get points in camera frame, with z-front
pts_cf = d_utils.get_aograsp_pts_in_cam_frame_z_front(args.pc_file)

# Visualize heatmap in o3d
if args.visualize:
    print("Visualizing point cloud with Open3D...")
    v_utils.viz_heatmap(
        pts_cf, pcd_dict["grasp_likelihood_labels"],
        draw_frame=True,
        frame="camera"
    )

# Downsample to 4096 points
pcd = v_utils.get_o3d_pts(pts_cf)

# Save as .ply 
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
info = d_utils.get_info_from_path(os.path.dirname(args.pc_file))
cat = info["cat"]
ins = info["ins"]
state = info["state"]
render_num = info["render_num"]
save_str = f"{cat}_{ins}_state-{state}_render-{render_num}.ply"
save_path = os.path.join(args.output_dir, save_str)
o3d.io.write_point_cloud(
    save_path, pcd
)