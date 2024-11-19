#!/bin/sh 
#SBATCH -N 1 
#SBATCH -n 9
#SBATCH -p opengpu.p
#SBATCH -w korn
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -e slurm_logs/err_test_yolov5n.out 
#SBATCH -o slurm_logs/log_tetst_yolov5n.out

python test_yolo.py --weights '' --cfg us_yolov5n.yaml --data yolov5/data/coco.yaml --batch-size 32
