#!/bin/sh
### Experiment 7 — full-game training (see train_run_1/README.md §7)
#BSUB -q gpua100
#BSUB -J wm_exp7_fg
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o gpu_%J_exp7_fullgame.out
#BSUB -e gpu_%J_exp7_fullgame.err
#BSUB -B
#BSUB -N

module load cuda/11.6
source ~/Desktop/dm-i-ai-2025/venv/bin/activate

cd ~/Desktop/dm-i-ai-2025
[ -f race-car/WorldModel/train_run_1/wandb.env ] && . race-car/WorldModel/train_run_1/wandb.env

python race-car/WorldModel/train_run_1/train_fullgame.py \
    --model     model_a_lg \
    --data      laneshift_dataset.npz \
    --run-name  model_a_fullgame \
    --project   laneshift-worldmodel \
    --out-dir   race-car/WorldModel/checkpoints \
    --epochs    5000 \
    --batch-size 64 \
    --lr        1e-3 \
    --lambda-vel 0.1 \
    --patience  10 \
    --tbptt-chunk 25 \
    --num-workers 0
