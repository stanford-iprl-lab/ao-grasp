# AO-Grasp: Articulated Object Grasp Generation

## Installation

1. Create a conda environment:
```
conda create -n <ENV_NAME> python==3.7
```

2. Install PyTorch:

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

3. Install PointNet++. In the `ao-grasp` conda env, install PointNet2_PyTorch from the directory contained within this repo by running the following commands:
```
cd ao-grasp/ao-grasp/Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e .
```