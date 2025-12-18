#!/bin/bash

#SBATCH --job-name=evalin                      # Job name
#SBATCH -N 1                                   # Number of nodes
#SBATCH --gres=gpu:8                           # Number of GPUs per node
#SBATCH --ntasks-per-node=8                    # Number of tasks per node
#SBATCH --cpus-per-task=10
#SBATCH --mem=1200G
#SBATCH -A <your_account>
#SBATCH --qos=<your_qos>
#SBATCH --time=07-00:00:00
#SBATCH --output=slurm/%j.out

mkdir -p exp/knn
exec > exp/knn/${SLURM_JOB_ID}.log 2>&1

srun python -u eval_knn.py \
    --data_path /datasets/imagenet_fullsize/061417 \
    --model $1 \
    --pretrained_ckp $2
