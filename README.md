# HamEvo

This repository contains training, evaluation, and inference code for Hamiltonian prediction models.

## Environment Reference

Python 3.10 or newer is recommended for this release.

| Package | Reference version |
| --- | --- |
| Python | 3.10+ recommended |
| torch | 2.4.1+cu121 |
| pytorch-lightning | 2.4.0 |
| pyscf | 2.7.0 |
| gpu4pyscf-cuda11x | 1.2.0 |
| lmdb | 1.5.1 |
| apsw | 3.46.1.0 |
| hydra-core | 1.3.2 |
| omegaconf | 2.3.0 |

## Data and Checkpoints

Datasets and trained checkpoints are not included in this release. They will be provided in a later update.

Coming soon:

- Dataset download and preprocessing instructions
- Trained checkpoints
- Example commands for evaluation and inference

Training, evaluation, and inference entry points are:

- `src/train.py`
- `src/eval.py`
- `src/infer.py`
