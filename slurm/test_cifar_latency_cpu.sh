#!/bin/sh 
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -p openlab.p
#SBATCH -o slurm_logs/log_test_cifar.out
#SBATCH -e slurm_logs/err_test_cifar.out

python measure_latency_cifar.py --best --batch-size 1