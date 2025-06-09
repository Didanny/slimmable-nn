#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p opengpu.p 
#SBATCH -w korn
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_tinyimagenet_usvit.out 
#SBATCH -e slurm_logs/err_tinyimagenet_usvit.out 
python train_cifar.py --model tinyimagenet_usvit --dataset tinyimagenet --optimizer adamw --vit-scheduler --inplace-distill --label-smoothing 0.1