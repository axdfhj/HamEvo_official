#!/bin/bash

# remove debug logs
rm -rf ./logs/debug/*

# activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate HamEvo

# set python path
export PYTHONPATH=$PYTHONPATH:.
export PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=0,1

# set wandb
export WANDB_BASE_URL=https://api.bandw.top
# wandb login <your-wandb-key>

# run training
python ./src/train.py