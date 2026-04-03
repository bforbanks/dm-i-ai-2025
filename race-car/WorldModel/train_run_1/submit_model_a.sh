#!/bin/sh
### Job: model_a  —  GRU belief propagator (gru=128)
#BSUB -q gpuv100
#BSUB -J wm_model_a
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o race-car/WorldModel/gpu_logs/gpu_%J_model_a.out
#BSUB -e race-car/WorldModel/gpu_logs/gpu_%J_model_a.err
#BSUB -B
#BSUB -N

module load cuda/11.6
source ~/Desktop/dm-i-ai-2025/venv/bin/activate

cd ~/Desktop/dm-i-ai-2025
[ -f race-car/WorldModel/train_run_1/wandb.env ] && . race-car/WorldModel/train_run_1/wandb.env

python race-car/WorldModel/train.py \
    --model     model_a \
    --data      laneshift_dataset.npz \
    --run-name  model_a \
    --project   laneshift-worldmodel \
    --out-dir   race-car/WorldModel/checkpoints \
    --epochs    5000 \
    --batch-size 1024 \
    --T-seg     120 \
    --lr        3e-4 \
    --lambda-vel 0.1 \
    --patience  10 \
    --num-workers 4
