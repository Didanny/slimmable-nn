#!/bin/sh 
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -p openlab.p
#SBATCH -e slurm_logs/err_test_resize_time_1m.out 
#SBATCH -o slurm_logs/log_test_resize_time_1m.out

python test_yolo_runtime.py --iters 1000000