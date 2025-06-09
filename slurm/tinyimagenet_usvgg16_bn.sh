#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p opengpu.p 
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_tinyimagenet_usvgg16_bn.out 
#SBATCH -e slurm_logs/err_tinyimagenet_usvgg16_bn.out 
python train_cifar.py --model tinyimagenet_usvgg16_bn --dataset tinyimagenet --inplace-distill