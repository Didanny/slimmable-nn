#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p opengpu.p 
#SBATCH -w korn
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_tinyimagenet_usvit_x100.out 
#SBATCH -e slurm_logs/err_tinyimagenet_usvit_x100.out 
python train_static.py --model tinyimagenet_usvit_x100 --width 1.0 --dataset tinyimagenet --optimizer adamw --vit-scheduler --label-smoothing 0.1