#!/bin/sh
### Experiment 8 — model_a_lg + SensorParser features (not in submit_all.sh)
#BSUB -q gpua100
#BSUB -J wm_exp8_parser
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -o gpu_%J_exp8_parser.out
#BSUB -e gpu_%J_exp8_parser.err
#BSUB -B
#BSUB -N

module load cuda/11.6
source ~/Desktop/dm-i-ai-2025/venv/bin/activate

cd ~/Desktop/dm-i-ai-2025
[ -f race-car/WorldModel/train_run_1/wandb.env ] && . race-car/WorldModel/train_run_1/wandb.env

python race-car/WorldModel/train_run_1/train_exp8.py \
    --data      laneshift_dataset.npz \
    --run-name  model_a_lg_parser \
    --project   laneshift-worldmodel \
    --out-dir   race-car/WorldModel/checkpoints \
    --epochs    5000 \
    --batch-size 1024 \
    --T-seg     120 \
    --lr        1e-3 \
    --lambda-vel 0.1 \
    --patience  10 \
    --num-workers 4
