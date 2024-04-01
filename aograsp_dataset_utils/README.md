# AO-Grasp dataset

The AO-Grasp dataset contains 78,000 6 DoF parallel jaw grasps on 84 articulated object instances across 7 categires from the PartNet-Mobility dataset. It contains grasps for each object in 10 joint states: 1 closed state and 9 randomly-sampled open states. For each object state, we provide the full point clouds and partial point clouds captured from 20 randomly-sampled camera viewpoints, as well as part segmentation masks. Additionally, we include the pre-processed, PartNet-Mobility objects we used to generated data. We have pre-processed these instances by running V-HACD on their meshes to obtain convex meshes, which we find result in better collision geometries in PyBullet.

## Downloading AO-Grasp dataset

Fill out [this form](https://forms.gle/EVZbZGMYRiyKpo6GA) to download the AO-Grasp dataset and pre-processed object meshes. Note that the form will require you to sign into a Google account. 

## Contents of the AO-Grasp dataset

### AO-Grasp dataset

*Directory structure*

```
aograsp_dataset_2024
└── Box                                      # category
    └── 47645                                # instance
        └── 0                                # state, 10 per instance
            ├── init_state.npz               # information for loading object into PyBullet
            ├── point_cloud_info.npz         # full point cloud and segmentation mask
            └── raw/                         # grasp data files
                └── pos/                     # positive grasps
                    ├── 0000.npz             # positive grasp 0
                    ├── ...                  # positive grasps 1, 2, 3...
                └── neg/                     # negative grasps
                    ├── 0000.npz             # negative grasp 0
                    ├── ...                  # negative grasps 1, 2, 3...    
            └── raw_ref_img/                 # grasp data episode rollout images
                └── pos/                     # positive grasp images
                    ├── 0000_0.png           # positive grasp 0, image 0- gripper at start pose; pre collision check
                    ├── 0000_1.png           # positive grasp 0, image 1- gripper at start pose; after collision check
                    ├── 0000_2.png           # positive grasp 0, image 2- gripper after grasp
                    ├── 0000_3.png           # positive grasp 0, image 3- gripper after action
                    ├── ...                  # images for positive grasps 1, 2, 3...
                └── neg/                     # negative grasps
                    ├── ...                  # images for negative grasps 1, 2, 3...             
            └── render/                      # partial point clouds, 20 per state
                ├── all_renders.png          # RGB images of all 20 camera viewpoints
                └── 0000/                    # Camera viewpoint 0
                    ├── info.npz             # Camera pose, IDs of ground truth grasps in partial point cloud  
                    ├── point_cloud_seg.npz  # Partial point cloud from viewpoint 0, segmentation mask, ground truth grasp-likelihood labels 
                    ├── rgb.png              # RGB image of object from viewpoint 0
                    ├── seg.png              # Segmentation mask visualization from viewpoint 0
                    ├── depth.png            # Depth image from viewpoint 0
                └── ..../                    # Camera viewpoints 1, 2,... 19

```

### Pre-processed object meshes

Note: The instances we include here are only a subset of the PartNet-Mobility dataset. To download the full PartNet-Mobility dataset, visit their [webpage](https://sapien.ucsd.edu/downloads).

## Visualizing the AO-Grasp dataset

After you have downloaded the dataset and object meshes, we provide two scripts to visualize the data.

**Visualizing grasps in PyBullet on object instance**

To run this script, make sure the `aograsp_instances` directory is contained in the top-level `ao-grasp` directory.

**Visualizing point clouds**
