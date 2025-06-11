#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p opengpu.p 
#SBATCH -w korn
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_tinyimagenet_usresnet56_x25.out 
#SBATCH -e slurm_logs/err_tinyimagenet_usresnet56_x25.out 
python train_static.py --model tinyimagenet_usresnet56_x25 --width 0.25 --dataset tinyimagenet