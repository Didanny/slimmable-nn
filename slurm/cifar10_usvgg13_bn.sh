#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p opengpu.p 
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_cifar10_usvgg13_bn.out 
#SBATCH -e slurm_logs/err_cifar10_usvgg13_bn.out 
python train_cifar.py --model cifar10_usvgg13_bn --dataset cifar10 --inplace-distill