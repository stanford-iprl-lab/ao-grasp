import pybullet as p
import os
import sys
import numpy as np
import pandas
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.transform import Rotation as R

from aograsp.data_utils.object import Object
import aograsp.rotation_utils as r_utils
from aograsp.mesh_utils import create_gripper

# Load dataset paths
import aograsp.dataset_paths as dataset_paths
PARTNET_MOBILITY_PATH = dataset_paths.PARTNET_MOBILITY_PATH
VHACD_PATH = dataset_paths.VHACD_PATH


class VizEnv:
    def __init__(
        self,
        _p,
        init_state,
        save_dir=None,
        draw_gripper=True,
        draw_point=False,
    ):
        # Set sim parameters
        self._p = _p
        self.save_dir = save_dir
        self._time_step = 0.01
        self._set_pybullet_sim_params()

        self.object = self.init_obj(init_state)

        self.draw_gripper = draw_gripper
        self.draw_point = draw_point

        # IDs for debug lines
        self.debug = True
        self._debug_dict = {
            "fail_to_grasp": {"id": -1, "color": [1, 0, 0]},  # RED
            "fail_to_reach_start_pos": {"id": -1, "color": [1, 1, 1]},  # WHITE
            "success_good_action": {"id": -1, "color": [0, 1, 0]},  # GREEN
            "success_bad_action": {"id": -1, "color": [1, 1, 0]},  # YELLOW
            "fail_touching_obj": {"id": -1, "color": [0, 0, 1]},  # BLUE
            "fail_to_reach_target_pos_collision": {
                "id": -1,
                "color": [1, 1, 1],
            },  # WHITE
            "fail_unknown": {"id": -1, "color": [1, 0, 1]},  # MAGENTA
            "fail_penetrating_obj": {"id": -1, "color": [0.5, 0, 1]},  # PURPLE
            "not_evaluated": {"id": -1, "color": [0, 0, 0]},  # BLACK
            "text": {"id": -1, "color": [0, 0, 0]},  # BLACK
        }
        self.cmap = LinearSegmentedColormap.from_list(
            "", ["crimson", "gold", "limegreen"]
        )

        # Gripper viz init
        g_opening = 0.07
        gripper = create_gripper("panda")
        gripper_control_points = gripper.get_control_point_tensor(
            1, False, convex_hull=False
        ).squeeze()
        mid_point = 0.5 * (gripper_control_points[1, :] + gripper_control_points[2, :])
        grasp_line_plot = np.array(
            [
                np.zeros((3,)),
                mid_point,
                gripper_control_points[1],
                gripper_control_points[3],
                gripper_control_points[1],
                gripper_control_points[2],
                gripper_control_points[4],
            ]
        )
        self.gripper_control_points_closed = grasp_line_plot.copy()
        self.gripper_control_points_closed[2:, 0] = (
            np.sign(grasp_line_plot[2:, 0]) * g_opening / 2
        )

    def init_obj(self, object_state):
        """Add object from grasp dataset given init_state dict"""

        self.scaling = object_state["scaling"]
        object_path = os.path.join(
            PARTNET_MOBILITY_PATH,
            os.path.basename(os.path.normpath(object_state["path"])),
        )

        # init_pos, init_quat = sample_object_pose(self._p, range=self.scaling * 0.25)
        obj = Object(
            self._p,
            object_path,
            scaling=self.scaling,
            vhacd_root=VHACD_PATH,
        )
        if "qpos" in object_state:
            obj.config["qpos"] = object_state["qpos"]  # add qpos to object config
        obj.config["trans"] = object_state["trans"]
        obj.config["quat"] = object_state["quat"]
        obj.reset()  # reset object, sets qpos to config['qpos']

        # transformation from object to world for object pose in dataset
        self.data_H_obj_to_world = obj.H_link_to_world(obj.base_link_id)

        return obj

    def reset(self, grasp_data_dict, grasp_results_df, data_name=""):
        """Environment reset called at the beginning of an episode."""

        self.object.reset()  # reset object, sets qpos to config['qpos']

        # Visualize grasps
        if grasp_data_dict is not None:
            self.viz_grasps(grasp_data_dict, grasp_results_df)

            self._debug_dict["text"]["id"] = self._p.addUserDebugText(
                data_name,
                [-0.5, 0, 0.5],
                textColorRGB=[0, 0, 0],
                replaceItemUniqueId=self._debug_dict["text"]["id"],
                textSize=1.5,
            )

        self._p.stepSimulation()

    def viz_grasps(self, grasp_data_dict, grasp_results_df):
        for i, grasp_data in grasp_data_dict.items():
            if grasp_results_df is None or i >= len(grasp_results_df):
                eval_str = "not_evaluated"
            else:
                eval_str = grasp_results_df.loc[grasp_results_df["grasp_id"] == i][
                    "eval_str"
                ].item()
            self.viz_grasp_from_data(grasp_data, eval_str)

    def viz_grasp_from_data(
        self,
        grasp_data,
        eval_str,
    ):
        """Get grasp info in object frame"""

        data_H_world_to_obj = r_utils.get_H_inv(self.data_H_obj_to_world)

        target_pos_wf = (
            self.data_H_obj_to_world @ np.append(grasp_data["target_pos"], 1)
        )[:3]

        print("Evaluated", grasp_data["n_proposal"], eval_str)
        if eval_str != "not_evaluated":
            if eval_str not in self._debug_dict:
                color = [0, 0, 0]
            else:
                color = self._debug_dict[eval_str]["color"]
            pt_color = self.cmap(grasp_data["prob"])[:3]
        else:
            color = self.cmap(grasp_data["prob"])[:3]
            pt_color = self.cmap(grasp_data["prob"])[:3]

        if self.draw_point:
            # Draw point in world frame
            self._p.addUserDebugPoints(
                [target_pos_wf],
                pointColorsRGB=[pt_color],
                pointSize=10.0,
            )

        if self.draw_gripper:
            # Compute gripper viz pts in object frame
            g = np.zeros((4, 4))
            rot = r_utils.get_matrix_from_ori(grasp_data["start_ori"])
            g[:3, :3] = rot
            g[:3, 3] = grasp_data["target_pos"].T
            g[3, 3] = 1
            r = R.from_euler("z", 90, degrees=True)
            gripper_control_points = self.gripper_control_points_closed.copy()
            gripper_control_points = r.apply(gripper_control_points)
            z = gripper_control_points[-1, -1]
            gripper_control_points[:, -1] -= z
            gripper_control_points[[1], -1] -= 0.02
            pts = np.matmul(gripper_control_points, g[:3, :3].T)
            pts += np.expand_dims(g[:3, 3], 0)

            lines = [[0, 1], [2, 3], [1, 4], [1, 5], [5, 6]]

            # pts, lines = grasp_data["grasp"]
            for line in lines:
                o, d = line
                # Compute gripper viz pts in world frame
                o_wf = (self.data_H_obj_to_world @ np.append(pts[o], 1))[:3]
                e_wf = (self.data_H_obj_to_world @ np.append(pts[d], 1))[:3]
                # o_of = (data_H_world_to_obj @ np.append(pts[o], 1))[:3]
                # e_of = (data_H_world_to_obj @ np.append(pts[d], 1))[:3]
                # o_of, e_of = pts[o], pts[d]
                self._p.addUserDebugLine(
                    o_wf,
                    e_wf,
                    lineColorRGB=color,
                    lineWidth=3.0,
                )
                s = str(grasp_data["n_proposal"])
                if "start_pos" in grasp_data:
                    text_pos = grasp_data["start_pos"]
                else:
                    text_pos = grasp_data["target_pos"]

                # self._p.addUserDebugText(
                #    text=s,
                #    textPosition=text_pos,
                # )

                # if eval_str == "success_good_action":
                #    self._p.addUserDebugText(
                #        str(grasp_data["n_proposal"]),
                #        grasp_data["start_pos"],
                #        textColorRGB=[0, 0, 0],
                #        textSize=1,
                #    )

    def step(self):
        self._p.stepSimulation()

    def _set_pybullet_sim_params(self):
        """Set pybullet simulation parameters"""

        self._p.resetSimulation()
        self._p.setTimeStep(self._time_step)
        self._p.setPhysicsEngineParameter(enableConeFriction=1)
        self._p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self._p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)
        self._p.setPhysicsEngineParameter(numSolverIterations=200)
        self._p.setPhysicsEngineParameter(numSubSteps=5)
        self._p.setPhysicsEngineParameter(
            constraintSolverType=self._p.CONSTRAINT_SOLVER_LCP_DANTZIG,
            globalCFM=0.000001,
        )
        self._p.setPhysicsEngineParameter(enableFileCaching=0)
        self._p.setGravity(0, 0, 0)  # Set 0 gravity to prevent joints from falling
        self._p.setRealTimeSimulation(0)

        # Debug visualizer camera params
        # camParams = self._p.getDebugVisualizerCamera()
        # print("cameraDistance={}, cameraYaw={}, cameraPitch={}, cameraTargetPosition={}".\
        #        format(camParams[-2], camParams[-4], camParams[-3], camParams[-1]))
        # self._p.resetDebugVisualizerCamera(cameraDistance=1.492018699645996, cameraYaw=20.76853370666504, cameraPitch=-45.29729080200195, cameraTargetPosition=(-0.026409128680825233, 0.2726774513721466, -0.05787532031536102))
        self._p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=-40,
            cameraPitch=-30,
            cameraTargetPosition=(0, 0, 0),
        )
        # Turn off extra menus for a cleaner pybullet UI.
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_SHADOWS, 0)
