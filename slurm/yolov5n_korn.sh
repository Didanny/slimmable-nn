#!/bin/sh 
#SBATCH -N 1 
#SBATCH -n 20 
#SBATCH -p opengpu.p
#SBATCH -w korn
#SBATCH --gres=gpu:4
#SBATCH -e slurm_logs/err_yolov5n_korn.out 
#SBATCH -o slurm_logs/log_yolov5n_korn.out

python train_yolo.py --weights '' --cfg us_yolov5n.yaml --data yolov5/data/coco.yaml --batch-size 64 --epochs 300 --dp --workers 19
