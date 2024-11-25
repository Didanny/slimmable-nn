#!/bin/sh 
#SBATCH -N 1 
#SBATCH -n 20 
#SBATCH -p opengpu.p
#SBATCH -w poison
#SBATCH --gres=gpu:4
#SBATCH -e slurm_logs/err_yolov5n.out 
#SBATCH -o slurm_logs/log_yolov5n.out

python train_yolo_ddp.py --weights '' --cfg us_yolov5n.yaml --data yolov5/data/coco.yaml --batch-size 128 --epochs 300 --workers 19
