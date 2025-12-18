#!/bin/bash

#SBATCH --job-name=evalseg                     # Job name
#SBATCH -N 1                                   # Number of nodes
#SBATCH --gres=gpu:8                           # Number of GPUs per node
#SBATCH --ntasks-per-node=8                    # Number of tasks per node
#SBATCH --cpus-per-task=10
#SBATCH --mem=1200G
#SBATCH -A <your_account>
#SBATCH --qos=<your_qos>
#SBATCH --time=07-00:00:00
#SBATCH --output=slurm/%j.out

config_name=$(basename $1 .yaml)
mkdir -p exp/semseg/${config_name}
exec > exp/semseg/${config_name}/${SLURM_JOB_ID}.log 2>&1

srun python -u eval_semseg.py \
    --config $1 \
    --encoder $2 \
    --pretrained_ckp $3
