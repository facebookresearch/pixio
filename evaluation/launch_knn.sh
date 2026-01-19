#!/bin/bash

#SBATCH --job-name=evalin                      # Job name
#SBATCH -N 2                                   # Number of nodes
#SBATCH --gres=gpu:4                           # Number of GPUs per node
#SBATCH --ntasks-per-node=4                    # Number of tasks per node
#SBATCH --cpus-per-task=10
#SBATCH --mem=1200G
#SBATCH -A <your_account>
#SBATCH --qos=<your_qos>
#SBATCH --time=07-00:00:00
#SBATCH --output=slurm/%j.out

ckp_path=$2
dir_name=$(basename $(dirname $ckp_path))
epoch_file=$(basename $ckp_path .pth)
mkdir -p $3/knn
exec > $3/knn/${dir_name}_${epoch_file}_${SLURM_JOB_ID}.log 2>&1

srun python -u eval_knn_lingua.py \
    --data_path /datasets/imagenet_fullsize/061417 \
    --outdir $3 \
    --model $1 \
    --pretrained_ckp $2
