#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p opengpu.p 
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_tinyimagenet_usresnet56.out 
#SBATCH -e slurm_logs/err_tinyimagenet_usresnet56.out 
python train_cifar.py --model tinyimagenet_usresnet56 --dataset tinyimagenet --inplace-distill