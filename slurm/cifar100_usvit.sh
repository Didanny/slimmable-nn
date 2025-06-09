#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p opengpu.p 
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_cifar100_usvit.out 
#SBATCH -e slurm_logs/err_cifar100_usvit.out 
python train_cifar.py --model cifar100_usvit --dataset cifar100 --optimizer adamw --vit-scheduler --inplace-distill --label-smoothing 0.1