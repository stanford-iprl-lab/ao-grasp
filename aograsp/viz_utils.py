"""
Visualization functions
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import open3d as o3d


def get_o3d_pts(pts):
    """
    Get open3d pcd from pts np.array
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def viz_heatmap(
    all_pts,
    heatmap_labels,
    save_path=None,
    frame="world",
    draw_frame=False,
    scale_cmap_to_heatmap_range=False,
):
    pcd = get_o3d_pts(all_pts)
    cmap = matplotlib.cm.get_cmap("RdYlGn")
    if scale_cmap_to_heatmap_range:
        # Scale heatmap labels to [0,1] to index into cmap
        heatmap_labels = scale_to_0_1(heatmap_labels)
    colors = cmap(np.squeeze(heatmap_labels))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Plot and save without opening a window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)

    # Draw ref frame
    if draw_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis.add_geometry(mesh_frame)

    if frame == "camera":
        # If visualizing in camera frame, view pcd from scene view
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        fov = ctr.get_field_of_view()
        H = np.eye(4)
        H[2, 3] = 0.2  # Move camera back by 20cm
        param.extrinsic = H
        ctr.convert_from_pinhole_camera_parameters(param)
    else:
        # If world frame, place camera accordingly to face object front
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        fov = ctr.get_field_of_view()
        H = np.eye(4)
        H[2, -1] = 1
        R = Rotation.from_euler("XYZ", [90, 0, 90], degrees=True).as_matrix()
        H[:3, :3] = R
        param.extrinsic = H
        ctr.convert_from_pinhole_camera_parameters(param)

    vis.poll_events()
    vis.update_renderer()

    if save_path is None:
        vis.run()
    else:
        vis.capture_screen_image(
            save_path,
            do_render=True,
        )
    vis.destroy_window()


def scale_to_0_1(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def viz_histogram(
    labels,
    save_path=None,
    scale_cmap_to_heatmap_range=False,
):
    # Plot histgram with y axis log scale
    if scale_cmap_to_heatmap_range:
        # Scale histogram min, max to labels range
        n, bins, patches = plt.hist(labels, log=True, range=(min(labels), max(labels)))
    else:
        # Use histogram range [0,1]
        n, bins, patches = plt.hist(labels, log=True, range=(0, 1))

    # Set each bar according to color map
    cmap = matplotlib.cm.get_cmap("RdYlGn")
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # Scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    if scale_cmap_to_heatmap_range:
        # Scale heatmap labels to [0,1] to index into cmap
        col = scale_to_0_1(col)
    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cmap(c))

    # Label each bar with count
    plt.bar_label(patches)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()
