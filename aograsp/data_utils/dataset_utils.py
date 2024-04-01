"""
Dataset functions
"""

import numpy as np
import os

import aograsp.rotation_utils as r_utils


def get_dir_order_list(dir_order):
    """Get dir order list from path"""

    # Iterate from child directory in path
    dir_order_list = dir_order.split("/")
    if dir_order_list[-1] == "":
        dir_order_list.pop()
    return dir_order_list


def get_info_from_path(path, dir_order="cat/ins/state/render_dir/render_num"):
    """
    Get dataset info from directory path, given dir_order
    """

    info = {}
    # Iterate from child directory in path
    dirs = path.split("/")
    if dirs[-1] == "":
        dirs.pop()

    dir_order_list = get_dir_order_list(dir_order)

    for i, key in enumerate(reversed(dir_order_list)):
        info[key] = dirs[-(i + 1)]
    return info


def get_aograsp_pts_in_cam_frame_z_front(pc_path):
    """
    Given a path to a partial point cloud render/000*/point_cloud_seg.npz
    from the AO-Grasp dataset, get points in camera frame with z-front, y-down
    """

    # Load pts in world frame
    pcd_dict = np.load(pc_path, allow_pickle=True)["data"].item()
    pts_wf = pcd_dict["pts"]

    # Load camera pose information
    render_dir = os.path.dirname(pc_path)
    info_path = os.path.join(render_dir, "info.npz")
    if not os.path.exists(info_path):
        raise ValueError(
            "info.npz cannot be found. Please ensure that you are passing in a path to a point_cloud_seg.npz file"
        )
    info_dict = np.load(info_path, allow_pickle=True)["data"].item()
    cam_pos = info_dict["camera_config"]["trans"]
    cam_quat = info_dict["camera_config"]["quat"]

    # Transform points from world to camera frame
    H_cam2world_xfront = r_utils.get_H(r_utils.get_matrix_from_quat(cam_quat), cam_pos)
    H_world2cam_xfront = r_utils.get_H_inv(H_cam2world_xfront)
    H_world2cam_zfront = get_H_world_to_cam_z_front_from_x_front(H_world2cam_xfront)
    pts_cf = r_utils.transform_pts(pts_wf, H_world2cam_zfront)

    return pts_cf


def get_H_world_to_cam_z_front_from_x_front(H_world_to_cam_x_front):
    """
    Given world to camera (where cam frame +x axis pointing in viewing direction)
    get world to camera (where +z axis pointing in viewing direction)
    """

    # In the camera space, x axis points right, y axis points down and z axis points
    # front. In the world space, x is front, y is left, and z is up. The following
    # transformation rotate the camera coordinate to the world coordinate.
    _CAMERA_Z_TO_WORLD_Z = (0.5, -0.5, 0.5, -0.5)

    H_cam_to_world_x_front = r_utils.get_H_inv(H_world_to_cam_x_front)

    R_c2w_x_front = H_cam_to_world_x_front[:3, :3]
    t = H_cam_to_world_x_front[:3, 3]

    R_z = r_utils.get_matrix_from_quat(_CAMERA_Z_TO_WORLD_Z)
    R_c2w_z_front = R_c2w_x_front @ R_z
    H_cam_to_world_z_front = r_utils.get_H(R_c2w_z_front, t)
    H_world_to_cam_z_front = r_utils.get_H_inv(H_cam_to_world_z_front)

    return H_world_to_cam_z_front
