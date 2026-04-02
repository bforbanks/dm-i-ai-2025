#!/bin/sh
### Job: model_c  —  Hybrid GRU + inter-lane transformer
#BSUB -q gpuv100
#BSUB -J wm_model_c
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o gpu_%J_model_c.out
#BSUB -e gpu_%J_model_c.err
#BSUB -B
#BSUB -N

module load cuda/11.6
source ~/Desktop/dm-i-ai-2025/venv/bin/activate

cd ~/Desktop/dm-i-ai-2025
[ -f race-car/WorldModel/train_run_1/wandb.env ] && . race-car/WorldModel/train_run_1/wandb.env

python race-car/WorldModel/train.py \
    --model     model_c \
    --data      laneshift_dataset.npz \
    --run-name  model_c \
    --project   laneshift-worldmodel \
    --out-dir   race-car/WorldModel/checkpoints \
    --epochs    5000 \
    --batch-size 1024 \
    --T-seg     120 \
    --lr        3e-4 \
    --lambda-vel 0.1 \
    --patience  10 \
    --num-workers 4
