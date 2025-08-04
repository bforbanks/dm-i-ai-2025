#!/bin/sh
#BSUB -J dmiai
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -q gpuv100
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options

# module load python3
module load python3/3.13.5
source venv/bin/activate
python3 -m pip install -r requirements.txt
# python3 -m pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 trainer.py fit --config models/config_base.yaml --config models/StaticNNUN/config.yaml --config models/StaticNNUN/wandb.yaml
