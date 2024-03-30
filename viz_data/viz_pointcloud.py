import open3d as o3d
import argparse
import numpy as np
import os

import aograsp.viz_utils as v_utils

def main(pc_path, seg_mask=False):

    pc_dict = np.load(pc_path, allow_pickle=True)["data"].item()
    pts = pc_dict["pts"]

    if seg_mask:
        actionable_part_labels = pc_dict["actionable_part_labels"]
        seg_mask_labels = pc_dict["seg_mask_labels"]
        labels = np.isin(seg_mask_labels, actionable_part_labels).astype(float)
    else:
        if "labels" in pc_dict:
            labels = pc_dict["labels"]
        else:
            raise ValueError("File does not contain grasp-likelihood labels")

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
    args = parser.parse_args()
    main(args.pc_file, seg_mask=args.seg_mask)
