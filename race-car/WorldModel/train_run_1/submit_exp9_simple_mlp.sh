#!/bin/sh
### Experiment 9 — one hidden-layer MLP on full games (not in submit_all.sh)
#BSUB -q gpuv100
#BSUB -J wm_exp9_mlp
#BSUB -n 2
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -o gpu_%J_exp9_simple_mlp.out
#BSUB -e gpu_%J_exp9_simple_mlp.err
#BSUB -B
#BSUB -N

module load cuda/11.6
source ~/Desktop/dm-i-ai-2025/venv/bin/activate

cd ~/Desktop/dm-i-ai-2025
[ -f race-car/WorldModel/train_run_1/wandb.env ] && . race-car/WorldModel/train_run_1/wandb.env

python race-car/WorldModel/train_run_1/simple_mlp_fullgame.py \
    --data       laneshift_dataset.npz \
    --run-name   simple_mlp_fullgame_parser \
    --project    laneshift-worldmodel \
    --out-dir    race-car/WorldModel/checkpoints \
    --hidden     128 \
    --epochs     10000 \
    --batch-size 32 \
    --lr         1e-3 \
    --lambda-vel 0.1 \
    --patience   100 \
    --tbptt-chunk 200 \
    --num-workers 0
