#!/bin/sh 
#SBATCH -N 1 
#SBATCH -n 8 
#SBATCH -p opengpu.p
#SBATCH -w poison
#SBATCH --gres=gpu:1
#SBATCH -e slurm_logs/err_yolov5n.out 
#SBATCH -o slurm_logs/log_yolov5n.out

python train_yolo.py --weights '' --cfg us_yolov5n.yaml --data yolov5/data/coco.yaml --batch-size -1 --epochs 300
