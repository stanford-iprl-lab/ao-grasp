""" From Contact-GraspNet """

import argparse
import json
import os
import numpy as np
import pickle
import trimesh
import trimesh.transformations as tra


class PandaGripper(object):
    """An object representing a Franka Panda gripper."""

    def __init__(
        self,
        q=None,
        num_contact_points_per_finger=10,
        root_folder="aograsp/data_utils/",
    ):
        """Create a Franka Panda parallel-yaw gripper object.

        Keyword Arguments:
            q {list of int} -- opening configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
        """
        self.joint_limits = [0.0, 0.04]
        self.root_folder = root_folder

        self.default_pregrasp_configuration = 0.04
        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q
        fn_base = os.path.join(root_folder, "gripper_models/panda_gripper/hand.stl")
        fn_finger = os.path.join(root_folder, "gripper_models/panda_gripper/finger.stl")

        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        # transform fingers relative to the base
        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.contact_ray_origins = []
        self.contact_ray_directions = []

        with open(
            os.path.join(
                root_folder, "gripper_control_points/panda_gripper_coords.pickle"
            ),
            "rb",
        ) as f:
            self.finger_coords = pickle.load(f, encoding="latin1")
        finger_direction = (
            self.finger_coords["gripper_right_center_flat"]
            - self.finger_coords["gripper_left_center_flat"]
        )
        self.contact_ray_origins.append(
            np.r_[self.finger_coords["gripper_left_center_flat"], 1]
        )
        self.contact_ray_origins.append(
            np.r_[self.finger_coords["gripper_right_center_flat"], 1]
        )
        self.contact_ray_directions.append(
            finger_direction / np.linalg.norm(finger_direction)
        )
        self.contact_ray_directions.append(
            -finger_direction / np.linalg.norm(finger_direction)
        )

        self.contact_ray_origins = np.array(self.contact_ray_origins)
        self.contact_ray_directions = np.array(self.contact_ray_directions)

    def get_control_point_tensor(
        self, batch_size, use_tf=False, symmetric=False, convex_hull=True
    ):
        """
        Outputs a 5 point gripper representation of shape (batch_size x 5 x 3).

        Arguments:
            batch_size {int} -- batch size

        Keyword Arguments:
            use_tf {bool} -- outputing a tf tensor instead of a numpy array (default: {True})
            symmetric {bool} -- Output the symmetric control point configuration of the gripper (default: {False})
            convex_hull {bool} -- Return control points according to the convex hull panda gripper model (default: {True})

        Returns:
            np.ndarray -- control points of the panda gripper
        """

        control_points = np.load(
            os.path.join(self.root_folder, "gripper_control_points/panda.npy")
        )[:, :3]
        if symmetric:
            control_points = [
                [0, 0, 0],
                control_points[1, :],
                control_points[0, :],
                control_points[-1, :],
                control_points[-2, :],
            ]
        else:
            control_points = [
                [0, 0, 0],
                control_points[0, :],
                control_points[1, :],
                control_points[-2, :],
                control_points[-1, :],
            ]

        control_points = np.asarray(control_points, dtype=np.float32)
        if not convex_hull:
            # actual depth of the gripper different from convex collision model
            control_points[1:3, 2] = 0.0584
        control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])

        if use_tf:
            return tf.convert_to_tensor(control_points)

        return control_points


def create_gripper(
    name,
    configuration=None,
    root_folder="aograsp/data_utils/",
):
    """Create a gripper object.

    Arguments:
        name {str} -- name of the gripper

    Keyword Arguments:
        configuration {list of float} -- configuration (default: {None})
        root_folder {str} -- base folder for model files (default: {''})

    Raises:
        Exception: If the gripper name is unknown.

    Returns:
        [type] -- gripper object
    """
    if name.lower() == "panda":
        return PandaGripper(q=configuration, root_folder=root_folder)
    else:
        raise Exception("Unknown gripper: {}".format(name))
