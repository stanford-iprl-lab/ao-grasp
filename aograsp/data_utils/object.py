import numpy as np
import os
import sys
from os.path import join as pjoin
from typing import Dict
import json
from scipy.spatial.transform import Rotation
import pybullet as p
from pybullet_utils import bullet_client

import aograsp.rotation_utils as r_utils


class Object:
    """
    Load partnet-mobility object
    """

    def __init__(
        self,
        pybullet_api,
        sapien_path,
        init_pos=[0, 0, 0],
        init_quat=[0, 0, 0, 1],
        scaling=1.0,
        filter_cat=False,
        vhacd_root=None,
    ):
        self._p = pybullet_api

        self.scaling = scaling
        self.init_pos = init_pos
        self.init_quat = init_quat

        original_urdf_path = os.path.join(sapien_path, "mobility.urdf")
        if vhacd_root is not None:
            urdf_path = pjoin(vhacd_root, sapien_path.split("/")[-1], "mobility.urdf")
            if not os.path.exists(urdf_path):
                # No VHACD meshes found; use original meshes
                urdf_path = original_urdf_path
        else:
            urdf_path = original_urdf_path

        if urdf_path == original_urdf_path:
            print("Using original meshes")
        else:
            print(("Using VHACD meshes"))

        try:
            self.id = self._p.loadURDF(
                urdf_path,
                globalScaling=self.scaling,
                useFixedBase=True,
                flags=self._p.URDF_USE_INERTIA_FROM_FILE,
            )
        except:
            raise ValueError(
                "Failed to load object URDF from specified path {}".format(urdf_path)
            )

        mobility_path = os.path.join(sapien_path, "mobility_v2.json")
        with open(mobility_path, "r") as f:
            self.mobility_v2 = json.load(f)

        self.base_link_id = -1  # This is convention of how URDFs are loaded

        # Change color of links to all gray
        for j in range(self._p.getNumJoints(self.id)):
            if vhacd_root is not None:
                # If loading vhacd meshes, make object all-gray
                # because obj files do not have textures
                self._p.changeVisualShape(
                    self.id,
                    j,
                    rgbaColor=np.array([227, 238, 247, 255]) / 255.0,
                    # rgbaColor=[0.8, 0.8, 0.8, 1],
                )

    def reset(
        self,
        qpos,
        trans=None,
        quat=None,
    ):
        """
        Reset object:
        - set joint positions to qpos

        args:
            qpos: list of joint angles for each joint in object
        """

        for joint_id in range(self._p.getNumJoints(self.id)):
            self._p.resetJointState(
                self.id,
                joint_id,
                targetValue=qpos[joint_id],
                targetVelocity=0.0,
            )

        # Restore object base pose
        if trans is None:
            trans = self.init_trans
        if quat is None:
            quat = self.init_quat
        self._p.resetBasePositionAndOrientation(self.id, trans, quat)

    def H_link_to_world(self, link_id):
        if link_id == -1:
            # Base (root) link
            link_pos_wf, link_quat_wf = self._p.getBasePositionAndOrientation(self.id)
        else:
            link_pos_wf, link_quat_wf = self._p.getLinkState(self.id, link_id)[0:2]

        R_link_to_world = np.array(
            self._p.getMatrixFromQuaternion(link_quat_wf)
        ).reshape(3, 3)

        H_link_to_world = r_utils.get_H(R_link_to_world, link_pos_wf)

        return H_link_to_world
