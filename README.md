# AO-Grasp: Articulated Object Grasp Generation

## Installation

AO-Grasp requires two conda environments, one for running inference to predict heatmaps, and one for running Contact-GraspNet. Follow the instructions below to set up both environments.

### Setting up the `ao-grasp` conda environment

1. From within the `ao-grasp/` directory, create a conda env named `ao-grasp` with the provided environment yaml file.

```
conda env create --name ao-grasp --file aograsp-environment.yml
```

2. Activate the new conda env you just created.
```
conda activate ao-grasp
```

3. Install PyTorch 
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

4. Install PointNet++. In the `ao-grasp` conda env, install PointNet2_PyTorch from the directory contained within this repo by running the following commands:
```
cd aograsp/models/Pointnet2_PyTorch/
pip install -r requirements.txt
pip install -e .
```

5. Install the `aograsp` package as an editable package
```
pip install -e .
```

6. Test the installation

### Setting up the `cgn` conda environment

1. From within the `ao-grasp/contact_graspnet` directory, create a conda env named `cgn` with the provided environment yaml file.
```
cd contact_graspnet
conda env create --name cgn --file aograsp_cgn_environment.yml 
```

2. Download CGN checkpoints

Download trained models from [here](https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl?usp=sharing) and copy them into the `checkpoints/` folder.

3. Test the installation
