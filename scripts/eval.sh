#!/bin/bash

# remove debug logs
rm -rf ./logs/debug/*

# activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate matv2

# set python path
export PYTHONPATH=$PYTHONPATH:.
export PROJECT_ROOT=./

# run evaluation
export CUDA_VISIBLE_DEVICES=0
python ./src/eval.py \
--config-name eval.yaml

echo "${RESULTS_NAME} evaluation finished!"