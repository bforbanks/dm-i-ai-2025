#!/bin/sh
### Job: model_a_lg  —  GRU belief propagator (gru=256, large)
#BSUB -q gpuv100
#BSUB -J wm_a_lg
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o gpu_%J_model_a_lg.out
#BSUB -e gpu_%J_model_a_lg.err
#BSUB -B
#BSUB -N

module load cuda/11.6
source ~/Desktop/dm-i-ai-2025/venv/bin/activate

cd ~/Desktop/dm-i-ai-2025
[ -f race-car/WorldModel/train_run_1/wandb.env ] && . race-car/WorldModel/train_run_1/wandb.env

python race-car/WorldModel/train.py \
    --model     model_a_lg \
    --data      laneshift_dataset.npz \
    --run-name  model_a_lg \
    --project   laneshift-worldmodel \
    --out-dir   race-car/WorldModel/checkpoints \
    --epochs    5000 \
    --batch-size 1024 \
    --T-seg     120 \
    --lr        3e-4 \
    --lambda-vel 0.1 \
    --patience  10 \
    --num-workers 4
