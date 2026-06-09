#!/bin/bash

# activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate matv2

# set python path
export PYTHONPATH=$PYTHONPATH:.
export PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# set cuda visible devices
export CUDA_VISIBLE_DEVICES=3

# run inference
HYDRA_FULL_ERROR=1 python ./src/infer.py \
  ckpt_path=<path_to_checkpoint> \
  inference.name=<dataset_name>

echo "Inference finished!"
