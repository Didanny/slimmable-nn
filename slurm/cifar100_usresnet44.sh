#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p opengpu.p 
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_cifar100_usresnet44.out 
#SBATCH -e slurm_logs/err_cifar100_usresnet44.out 
python train_cifar.py --model cifar100_usresnet44 --dataset cifar100 --inplace-distill