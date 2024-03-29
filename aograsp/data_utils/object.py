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
    Loaded from urdf files

    Args:
      urdf_path (str) -- Path to urdf file
      name (str)      -- Name of object
      init_pos (list) -- Initial position of object
      scaling (float) -- Scale to scale object by
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
            self.valid = False
            raise ValueError(
                "Failed to load object URDF from specified path {}".format(urdf_path)
            )
        self.valid = True

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
                    rgbaColor=np.array([227, 238, 247, 255])/255.,
                    #rgbaColor=[0.8, 0.8, 0.8, 1],
                )

        self.config = {
            "trans": init_pos,
            "quat": init_quat,
            "path": sapien_path,
            "scaling": scaling,
        }

        self.disable_motors()

    def reset(
        self,
        random_joint=False,
        qpos=None,
        target_joint_list=None,
        trans=None,
        quat=None,
    ):
        """
        Reset object:
        - set joint positions to qpos
        - set target_joint_ids and target_link_ids based on target_joint_list

        args:
            random_joint: ??
            qpos: joint configuration
            target_joint_list: list of target joint names (string) [NEW]
        """

        if not self.valid:
            return
        if qpos is not None:
            self.config["qpos"] = qpos

        cur_qpos = self.config["qpos"]
        for joint_id in range(self._p.getNumJoints(self.id)):
            self._p.resetJointState(
                self.id,
                joint_id,
                targetValue=cur_qpos[joint_id],
                targetVelocity=0.0,
            )

        # Restore object base pose
        if trans is not None:
            self.config["trans"] = trans
        if quat is not None:
            self.config["quat"] = quat
        self._p.resetBasePositionAndOrientation(
            self.id, self.config["trans"], self.config["quat"]
        )


    def disable_motors(self):
        for j in range(self._p.getNumJoints(self.id)):
            self._p.setJointMotorControlArray(
                self.id,
                jointIndices=[j],
                controlMode=self._p.VELOCITY_CONTROL,
                forces=[0.0],
            )

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
