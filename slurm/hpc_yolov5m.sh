#!/bin/sh 
#SBATCH -n 9
#SBATCH -p free-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -A DUTT_LAB
#SBATCH -e slurm_logs/err_yolov5s.out 
#SBATCH -o slurm_logs/log_yolov5s.out

python train_yolo.py --weights '' --cfg us_yolov5m.yaml --data yolov5/data/coco.yaml --batch-size -1 --epochs 300 --hyp data/hyps/hyp.scratch-med.yaml
