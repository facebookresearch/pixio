#!/bin/bash

model=$1
pretrained=$2
outdir=$3

sbatch launch_monodepth_lingua.sh monodepth/configs/nyuv2_dpt.yaml $model $pretrained $outdir
sbatch launch_monodepth_lingua.sh monodepth/configs/kitti_dpt.yaml $model $pretrained $outdir
sbatch launch_monodepth_lingua.sh monodepth/configs/nyuv2_linear.yaml $model $pretrained $outdir
sbatch launch_monodepth_lingua.sh monodepth/configs/kitti_linear.yaml $model $pretrained $outdir

sbatch launch_semseg_lingua.sh semseg/configs/ade20k_linear.yaml $model $pretrained $outdir
sbatch launch_semseg_lingua.sh semseg/configs/pascal_linear.yaml $model $pretrained $outdir
sbatch launch_semseg_lingua.sh semseg/configs/loveda_linear.yaml $model $pretrained $outdir
sbatch launch_semseg_lingua.sh semseg/configs/ade20k_dpt.yaml $model $pretrained $outdir
sbatch launch_semseg_lingua.sh semseg/configs/pascal_dpt.yaml $model $pretrained $outdir
sbatch launch_semseg_lingua.sh semseg/configs/loveda_dpt.yaml $model $pretrained $outdir

sbatch launch_knn_lingua.sh $model $pretrained $outdir

# Examples:
# bash run_all.sh pixo_1b pixio_vit1b16.pth
# bash run_all.sh pixo_huge pixio_vith16.pth
# bash run_all.sh pixo_large pixio_vitl16.pth
