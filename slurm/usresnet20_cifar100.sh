#!/bin/sh 
#SBATCH -N 1 
#SBATCH -n 1 
#SBATCH -p opengpu.p
#SBATCH --gres=gpu:1
#SBATCH -e slurm_logs/err_usvgg11_cifar100.out 
#SBATCH -o slurm_logs/log_usvgg11_cifar100.out

# for i in $(seq 1 10); do
python train_cifar.py --model cifar100_usresnet20 --dataset cifar100; 
# done