#!/bin/bash

# remove debug logs
rm -rf ./logs/debug/*

# activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate HamEvo

# set python path
export PYTHONPATH=$PYTHONPATH:.
export PROJECT_ROOT=./

# set cuda visible devices
export CUDA_VISIBLE_DEVICES=0

# run debug
HYDRA_FULL_ERROR=1 python ./src/train.py debug=default