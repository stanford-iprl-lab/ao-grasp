"""
Run pointscore inference on partial pointcloud
"""

import os
from argparse import ArgumentParser
import numpy as np
import torch
import open3d as o3d

#from train_model.test import load_conf
import aograsp.viz_utils as v_utils
import aograsp.model_utils as m_utils


def get_heatmap(args):

    # Load model and weights
    model = m_utils.load_model()
    model.to(args.device)
    print(str(model))

    # Directory to save output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"Writing output to: {args.output_dir}")

    # TODO get some test data (synthetic and real)
    # Test inference on this data

    point_score_dir = os.path.join(args.output_dir, "point_score")
    point_score_img_dir = os.path.join(args.output_dir, "point_score_img")
    os.makedirs(point_score_dir, exist_ok=True)
    os.makedirs(point_score_img_dir, exist_ok=True)

    # Read from args.pcd_path and put into input_dict for model
    if args.pcd_path is None:
        raise ValueError("Missing path to input point cloud (--pcd_path)")

    pcd_ext = os.path.splitext(args.pcd_path)[1]
    data_name = os.path.splitext(os.path.basename(args.pcd_path))[0]
    if pcd_ext == ".ply":
        # Open3d pointcloud
        pcd = o3d.io.read_point_cloud(args.pcd_path)
        pts_arr = np.array(pcd.points)
    else:
        raise ValueError(f"{pcd_ext} filetype not supported")

    # Recenter pcd to origin
    mean = np.mean(pts_arr, axis=0)
    pts_arr -= mean

    # Get pts as tensor and create input dict for model
    pts = torch.from_numpy(pts_arr).float().to(args.device)
    pts = torch.unsqueeze(pts, dim=0)
    input_dict = {"pcs": pts}

    # Run inference
    model.eval()
    with torch.no_grad():
        test_dict = model.test(input_dict, None)

    # Save heatmap point cloud
    scores = test_dict["point_score_heatmap"][0].cpu().numpy()

    pcd_path = os.path.join(point_score_dir, f"{data_name}.npz")
    heatmap_dict = {
        "pts": pts_arr + mean,  # Save original un-centered data
        "labels": scores,
    }
    np.savez_compressed(pcd_path, data=heatmap_dict)

    # Save image of heatmap
    fig_path = os.path.join(point_score_img_dir, f"heatmap_{data_name}.png")
    hist_path = os.path.join(point_score_img_dir, f"heatmap_{data_name}_hist.png")
    try:
        v_utils.viz_heatmap(
            heatmap_dict["pts"],
            scores,
            save_path=fig_path,
            frame="camera",
            scale_cmap_to_heatmap_range=True,
        )
    except Exception as e:
        print(e)

    v_utils.viz_histogram(
        scores,
        save_path=hist_path,
        scale_cmap_to_heatmap_range=True,
    )


def parse_args():

    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to directory to write output files",
    )
    parser.add_argument("--pcd_path", type=str, help="Path to seg_pcd_clean.ply")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cpu or cuda:x for using cuda on GPU x",
    )

    parser.add_argument(
        "--display",
        type=str,
        default=":3",
        choices=[":1", ":2", ":3"],
        help="Display number",
    )

    # parse args
    conf = parser.parse_args()

    conf.data_path = None

    return conf


def main(args):
    os.environ["DISPLAY"] = args.display

    ### prepare before training
    # make exp_name

    # Load and test single trained model
    # Load conf params
    #load_conf_path = os.path.join(conf.exp_dir, "conf.pth")
    #conf = load_conf(conf, load_conf_path)

    get_heatmap(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
