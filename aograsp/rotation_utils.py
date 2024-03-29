import numpy as np
from scipy.spatial.transform import Rotation


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
