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
source ~/dm-i-ai-2025/venv/bin/activate

# export WANDB_API_KEY="your_key_here"

cd ~/dm-i-ai-2025

python race-car/WorldModel/train.py \
    --model     model_a_lg \
    --data      laneshift_dataset.npz \
    --run-name  model_a_lg \
    --project   laneshift-worldmodel \
    --out-dir   race-car/WorldModel/checkpoints \
    --epochs    50 \
    --batch-size 32 \
    --T-seg     120 \
    --lr        3e-4 \
    --lambda-vel 0.1 \
    --patience  5 \
    --num-workers 4
