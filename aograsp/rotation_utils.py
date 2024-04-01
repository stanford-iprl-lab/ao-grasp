import numpy as np
from scipy.spatial.transform import Rotation
import os


def get_H(R, p):
    """Construct homogenous transformation matrix H"""
    H = np.zeros((4, 4))
    H[0:3, 0:3] = R
    H[0:3, 3] = p
    H[3, 3] = 1
    return H


def get_H_inv(H):
    """Get inverse of homogenous transformation matrix H"""

    H_inv = np.zeros(H.shape)
    H_inv[3, 3] = 1
    R = H[:3, :3]
    P = H[:3, 3]

    H_inv[:3, :3] = R.T
    H_inv[:3, 3] = -R.T @ P

    return H_inv


def get_matrix_from_ori(ori):
    return Rotation.from_euler("xyz", ori).as_matrix()


def get_matrix_from_quat(quat):
    return Rotation.from_quat(quat).as_matrix()


def transform_pts(pts_in, H):
    """Transform pts ([N, 3] or [3,]) by H"""
    # Check pts dim and reshape
    if pts_in.ndim == 1:
        pts = np.expand_dims(pts_in, axis=0)
    else:
        pts = pts_in

    pts_new = np.concatenate([pts, np.ones_like(pts[:, :1])], axis=-1)  # [N, 4]
    pts_new = np.matmul(H, pts_new.T).T[
        :, :3
    ]  # [4, 4], [4, N] -> [4, N] -> [N, 4] -> [N, 3]

    if pts_in.ndim == 1:
        return np.squeeze(pts_new)
    else:
        return pts_new
