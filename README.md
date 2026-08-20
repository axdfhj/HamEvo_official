# HamEvo

HamEvo is a fixed-point neural operator for molecular Hamiltonian prediction. This repository contains the source code and configurations used for training, evaluation, and inference.

## Released contents

| Content | Location |
| --- | --- |
| Training, evaluation, and inference source code | `src/` |
| Hydra configurations | `configs/` |
| Small inference input | `notebooks/Adamantane.xyz` |
| Executable inference notebook | `notebooks/inference_demo.ipynb` |
| Research datasets and trained checkpoints | [HamEvo data repository](https://huggingface.co/datasets/ZJUSCL/hamevo-data) |

The short inference demo requires only this GitHub repository, one approximately 561 MB checkpoint, and the included XYZ file. It does not require downloading the complete research dataset, which is approximately 1.15 TB.

## System requirements

### Tested reference environment

The public release environment has been tested on Linux x86_64 with an NVIDIA CUDA GPU. The following versions provide the reference environment; nearby compatible versions may also work.

| Component | Tested version |
| --- | --- |
| Operating system | Linux x86_64, kernel 5.15 |
| Python | 3.10 |
| PyTorch | 2.4.1+cu121 |
| PyTorch Geometric | 2.6.1 |
| e3nn | 0.5.6 |
| PyTorch Lightning | 2.4.0 |
| PySCF | 2.7.0 |
| ASE | 3.23.0 |
| CuPy | 12.3.0 (`cupy-cuda12x`) |
| GPU4PySCF | 1.2.0 (`gpu4pyscf-cuda11x`) |

The remaining direct dependencies and tested versions are listed in the installation command below. These versions are based on the `atomos12` experiment environment; nearby compatible versions may also work. The public source requires Python 3.10 syntax. The CUDA 12 CuPy package and CUDA 11 GPU4PySCF package shown above were used together during the study and were import-tested on an H100 GPU.

### Hardware

- A CUDA-capable NVIDIA GPU is the reference platform for training and the inference demo.
- The release has been exercised on NVIDIA H100 80 GB GPUs and on other NVIDIA GPUs used during the study.
- CPU-only execution has not been validated for the demo.
- Allow at least 10 GB of free disk space for the software environment and one checkpoint. Full-data reproduction requires substantially more storage.

## Installation

Create a new environment rather than upgrading an existing environment in place:

```bash
conda create -n hamevo python=3.10 -y
conda activate hamevo
```

Install the tested PyTorch and PyG wheels:

```bash
python -m pip install \
  torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  pyg-lib==0.4.0+pt24cu121 \
  torch-scatter==2.1.2+pt24cu121 \
  torch-sparse==0.6.18+pt24cu121 \
  torch-cluster==1.6.3+pt24cu121 \
  torch-spline-conv==1.2.2+pt24cu121 \
  torch-geometric==2.6.1 \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

Install the remaining tested dependencies:

```bash
python -m pip install \
  numpy==1.24.4 scipy==1.10.1 \
  pytorch-lightning==2.4.0 torchmetrics==1.4.3 torch-ema==0.3 \
  e3nn==0.5.6 \
  hydra-core==1.3.2 hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0 omegaconf==2.3.0 rootutils==1.0.7 \
  lmdb==1.5.1 apsw==3.46.1.0 ase==3.23.0 pyscf==2.7.0 \
  geometric==1.1 pyscf-dispersion==1.3.0 cupy-cuda12x==12.3.0 cutensor-cu11==2.0.2 \
  rich==13.9.4 termcolor==2.4.0 tqdm==4.66.5 wandb==0.18.3 \
  huggingface-hub==0.25.2 ipykernel==5.5.5

# Preserve the tested CUDA 12 CuPy installation when adding GPU4PySCF.
python -m pip install --no-deps \
  gpu4pyscf-libxc-cuda11x==0.5 gpu4pyscf-cuda11x==1.2.0

python -m ipykernel install --user --name hamevo --display-name "Python (hamevo)"
```

Allow approximately 15–30 minutes for installation on a Linux workstation with a broadband connection. Most of this time is used to download the CUDA, PyTorch, and PyG packages; network speed and package caches affect the total time.

Check the installation:

```bash
python - <<'PY'
import cupy
import e3nn
import gpu4pyscf
import torch
import torch_geometric

print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("torch_geometric", torch_geometric.__version__)
print("e3nn", e3nn.__version__)
print("cupy", cupy.__version__, "devices", cupy.cuda.runtime.getDeviceCount())
print("gpu4pyscf", gpu4pyscf.__version__)
PY
```

## Demonstration

### 1. Download one trained checkpoint

The fixed-point fine-tuned checkpoint is used for the default demo:

```bash
huggingface-cli download \
  ZJUSCL/hamevo-data \
  ckpt/stage2-dm-ft.ckpt \
  --repo-type dataset \
  --local-dir .
```

The resulting path is `ckpt/stage2-dm-ft.ckpt`. The repository also provides the pretraining checkpoint `ckpt/stage1-pt.ckpt`.

### 2. Run the notebook

Open `notebooks/inference_demo.ipynb` in VS Code or another Jupyter frontend and select the `hamevo` kernel. In the user-input cell, set the existing notebook variables as follows, and then run all cells:

```python
CKPT_PATH = Path("ckpt/stage2-dm-ft.ckpt")
XYZ_PATH = Path("notebooks/Adamantane.xyz")
DEVICE = "cuda"
```

The GDB17 part of the notebook is optional. It is skipped when `GDB17_ROOT` does not point to a downloaded GDB17 dataset.

### Expected output

The model-loading cell prints the checkpoint path and CUDA device. The final XYZ result is a dictionary with the following fields:

- `ham_pred`: predicted Hamiltonian matrix;
- `homo_ev` and `lumo_ev`: predicted frontier orbital energies in eV;
- `dipole_debye`: predicted dipole magnitude in Debye;
- `nstep` and `gx`: fixed-point solver diagnostics.

Numerical values depend on the selected checkpoint, GPU, and numerical libraries. Successful completion with finite values and the expected output fields is the demo acceptance criterion.

### Expected run time

After the environment and checkpoint are available, allow several minutes for the single-structure Adamantane demo on a recent CUDA-capable NVIDIA GPU. Checkpoint download time is excluded. CPU-only execution has not been benchmarked.

## Using HamEvo with another XYZ file

1. Prepare an XYZ or multi-frame XYZ trajectory with coordinates in angstrom.
2. Use elements and chemical domains supported by the selected checkpoint.
3. Set `XYZ_PATH` in `notebooks/inference_demo.ipynb` to the new file.
4. Run the model-loading and XYZ inference cells.

The notebook calculates the overlap matrix with PySCF using the def2-SVP basis and returns a predicted Hamiltonian and derived orbital properties for each structure.

## Training and evaluation

The entry points are:

- `src/train.py` for training and fine-tuning;
- `src/eval.py` for evaluation on a configured dataset;
- `src/infer.py` for configuration-driven trajectory inference.

Set the repository and data paths before running an entry point:

```bash
export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Select a dataset configuration from `configs/data/`, place the corresponding public data under `scf_dataset/` or override `paths.data_dir`, and run, for example:

```bash
python src/eval.py \
  --config-name eval.yaml \
  ckpt_path=/absolute/path/to/checkpoint.ckpt \
  data=gdb17 \
  paths.data_dir=/absolute/path/to/data
```

Hydra configuration files record the model, solver, dataset, batch-size, optimizer, and trainer settings. Full-data evaluation requires the relevant dataset from the HamEvo data repository.

## Algorithm overview

```text
Input: molecular species and coordinates, overlap matrix S, trained parameters theta
Construct the molecular graph and orbital masks
Initialize the latent Hamiltonian blocks z_0
For k = 0, ..., maximum fixed-point iterations:
    z_(k+1) = f_theta(z_k, molecular graph)
    stop when the fixed-point residual satisfies the configured tolerance
Assemble the full Hamiltonian H from the converged blocks
Solve H C = S C epsilon
Return H, orbital energies, orbital coefficients, and derived properties
```

Implementation details are in `src/models/deqham_module.py`, `src/models/components/equiformer_function.py`, and `src/utils/deq_lib/solvers.py`.

## Data and checkpoints

- Data and checkpoints: <https://huggingface.co/datasets/ZJUSCL/hamevo-data>
- Pretraining checkpoint: `ckpt/stage1-pt.ckpt`
- Fixed-point fine-tuned checkpoint: `ckpt/stage2-dm-ft.ckpt`

The data repository contains the processed datasets, split files, and checkpoints used by the released workflows. Users reproducing manuscript results should download only the datasets required for the relevant experiment.

## License

HamEvo is released under the MIT License. See `LICENSE` for the license text.
