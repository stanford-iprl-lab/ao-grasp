"""
Visualize point clouds in the AO-Grasp dataset
"""

import open3d as o3d
import argparse
import numpy as np
import os

import aograsp.viz_utils as v_utils

def main(args):

    pc_dict = np.load(args.pc_file, allow_pickle=True)["data"].item()
    pts = pc_dict["pts"]

    if args.seg_mask:
        actionable_part_labels = pc_dict["actionable_part_labels"]
        seg_mask_labels = pc_dict["seg_mask_labels"]
        labels = np.isin(seg_mask_labels, actionable_part_labels).astype(float)
    else:
        if "grasp_likelihood_labels" in pc_dict:
            labels = pc_dict["grasp_likelihood_labels"]
        else:
            raise ValueError("File does not contain grasp-likelihood labels. Try running script with the --seg_mask flag, or running it on a render/00**/point_cloud_seg.npz file.")
    
    # Visualize ground truth grasps by appending points to "pts"
    if args.gt_grasps:
        if os.path.basename(args.pc_file) != "point_cloud_seg.npz":
            raise ValueError("Can only visualize ground truth grasps on render/00**/point_cloud_seg.npz files")

        # Load ids of gt grasps from info.npz file
        info_path = os.path.join(os.path.dirname(args.pc_file), "info.npz")
        info_dict = np.load(info_path, allow_pickle=True)["data"].item()

        state_path = os.path.dirname(os.path.dirname(os.path.dirname(args.pc_file)))
        data_dir = os.path.join(state_path, "raw")
        eef_pos_list = []
        eef_quat_list = []
        for grasp_label in ["pos"]:
            data_path = os.path.join(data_dir, grasp_label)
            ids = info_dict[f"{grasp_label}_grasp_ids"]
            for i in ids:
                grasp_path = os.path.join(data_path, f"{i:04}.npz")
                grasp_dict = np.load(grasp_path, allow_pickle=True)["data"].item()
                grasp_pt = grasp_dict["pos_wf"]
                grasp_quat = grasp_dict["after_grasp_quat_wf"]

                eef_pos_list.append(grasp_pt)
                eef_quat_list.append(grasp_quat)

    if args.gt_grasps:
        v_utils.viz_pts_and_eef_o3d(
            pts,
            eef_pos_list,
            eef_quat_list,
            heatmap_labels=labels
        )
    else:
        v_utils.viz_heatmap(pts, labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pc_file", type=str, help="Path to .npz with point cloud or heatmap to visualize"
    )
    parser.add_argument(
        "--seg_mask",
        action="store_true",
        help="Visualize segmentation mask",
    )
    parser.add_argument(
        "--gt_grasps",
        action="store_true",
        help="Visualize ground truth grasps on partial point cloud. ONLY for render/00**/point_cloud_seg.npz files"
    )
    args = parser.parse_args()
    main(args)
