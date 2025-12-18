#!/bin/bash

model=$1
pretrained=$2

sbatch launch_monodepth.sh monodepth/configs/nyuv2_dpt.yaml $model $pretrained
sbatch launch_monodepth.sh monodepth/configs/kitti_dpt.yaml $model $pretrained
sbatch launch_monodepth.sh monodepth/configs/nyuv2_linear.yaml $model $pretrained
sbatch launch_monodepth.sh monodepth/configs/kitti_linear.yaml $model $pretrained

sbatch launch_semseg.sh semseg/configs/ade20k_linear.yaml $model $pretrained
sbatch launch_semseg.sh semseg/configs/pascal_linear.yaml $model $pretrained
sbatch launch_semseg.sh semseg/configs/loveda_linear.yaml $model $pretrained
sbatch launch_semseg.sh semseg/configs/ade20k_dpt.yaml $model $pretrained
sbatch launch_semseg.sh semseg/configs/pascal_dpt.yaml $model $pretrained
sbatch launch_semseg.sh semseg/configs/loveda_dpt.yaml $model $pretrained

sbatch launch_knn.sh $model $pretrained

# Examples:
# bash run_all.sh pixio_vith16 pixio_vith16.pth
