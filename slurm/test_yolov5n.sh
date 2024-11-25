#!/bin/sh 
#SBATCH -N 1 
#SBATCH -n 4
#SBATCH -p openlab.p
#SBATCH --mem=32G
#SBATCH -e slurm_logs/err_test_yolov5n.out 
#SBATCH -o slurm_logs/log_test_yolov5n.out

python test_yolo.py --weights '' --cfg us_yolov5n.yaml --data yolov5/data/coco.yaml --batch-size 1 --no-cal --best
