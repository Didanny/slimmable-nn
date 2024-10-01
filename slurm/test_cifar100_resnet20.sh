#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p opengpu.p 
#SBATCH --gres=gpu:1 
#SBATCH -o slurm_logs/log_test_cifar100_resnet20.out
#SBATCH -e slurm_logs/err_test_cifar100_resnet20.out
python test_cifar.py --model cifar100_usresnet20 --dataset cifar100 --best
