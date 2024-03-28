# AO-Grasp: Articulated Object Grasp Generation

Get actionable grasps for interacting with articulated objects from partial point clouds.

## Installation

AO-Grasp requires two conda environments, one for running inference to predict heatmaps, and one for running Contact-GraspNet. Follow the instructions below to set up both environments.

This code has been tested with Ubuntu 20.04 and CUDA 11.0

### Part 1: Setting up the `ao-grasp` conda environment

1. From within the `ao-grasp/` directory, create a conda env named `ao-grasp` with the provided environment yaml file:

```
conda env create --name ao-grasp --file aograsp-environment.yml
```

2. Activate the new conda env you just created:
```
conda activate ao-grasp
```

3. Install PyTorch:
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

4. Install the `aograsp` package as an editable package
```
pip install -e .
```

5. Install PointNet++. In the `ao-grasp` conda env, install PointNet2_PyTorch from the directory contained within this repo by running the following commands:
```
cd aograsp/models/Pointnet2_PyTorch/
pip install -r requirements.txt
pip install -e .
```

6. Test the installation by predicting the per-point grasp likelihood scores on a provided test point cloud:

```
python run_pointscore_inference.py --pcd_path '/juno/u/clairech/ao-grasp/test_data/real/microwave_closed.ply'
```

This will save the predicted scores in `output/point_score/microwave_closed.npz` and a visualization of the scores in `output/point_score/microwave_closed.npz`.

### Part 2: Setting up the `cgn` conda environment

1. From within the `ao-grasp/contact_graspnet` directory, create a conda env named `cgn` with the provided environment yaml file.
```
cd contact_graspnet
conda env create --name cgn --file aograsp_cgn_environment.yml 
```

2. Download CGN checkpoints

Download trained models from [here](https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl?usp=sharing) and copy them into the `checkpoints/` folder.

3. Test the installation from the `ao-grasp` directory
```
cd ..
python contact_graspnet/contact_graspnet/run_cgn_on_heatmap_file.py '/juno/u/clairech/ao-grasp/output/point_score/microwave_closed.npz' --viz_top_k 1
```
