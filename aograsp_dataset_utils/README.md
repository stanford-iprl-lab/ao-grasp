# AO-Grasp dataset

The AO-Grasp dataset contains 78,000 6 DoF parallel jaw grasps on 84 articulated object instances across 7 categires from the PartNet-Mobility dataset. It contains grasps for each object in 10 joint states: 1 closed state and 9 randomly-sampled open states. For each object state, we provide the full point clouds and partial point clouds captured from 20 randomly-sampled camera viewpoints, as well as part segmentation masks. Additionally, we include the pre-processed, PartNet-Mobility objects we used to generated data. We have pre-processed these instances by running V-HACD on their meshes to obtain convex meshes, which we find result in better collision geometries in PyBullet.

## Downloading AO-Grasp dataset

Fill out [this form](https://forms.gle/EVZbZGMYRiyKpo6GA) to download the AO-Grasp dataset and pre-processed object meshes. Note that the form will require you to sign into a Google account. 

## Contents of the AO-Grasp dataset

### AO-Grasp dataset

### Pre-processed object meshes

Note: The instances we include here are only a subset of the PartNet-Mobility dataset. To download the full PartNet-Mobility dataset, visit their [webpage](https://sapien.ucsd.edu/downloads).

## Visualizing the AO-Grasp dataset

After you have downloaded the dataset and object meshes, we provide two scripts to visualize the data.
