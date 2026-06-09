# HamEvo

This repository contains training, evaluation, and inference code for Hamiltonian prediction models.

## Environment

The release environment has been validated with Python 3.10, PyTorch 2.4.1/cu121, PyG 2.6.1, and e3nn 0.5.6. Use a new conda environment instead of upgrading an existing one in place.

```bash
conda create -n hamevo python=3.10 -y
conda activate hamevo

# Required by cupy-cuda11x / gpu4pyscf-cuda11x at runtime.
conda install -c conda-forge cudatoolkit=11.8 -y
```

Install PyTorch:

```bash
python -m pip install \
  torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install PyG wheels matching PyTorch 2.4/cu121:

```bash
python -m pip install \
  pyg-lib==0.4.0+pt24cu121 \
  torch-scatter==2.1.2+pt24cu121 \
  torch-sparse==0.6.18+pt24cu121 \
  torch-cluster==1.6.3+pt24cu121 \
  torch-spline-conv==1.2.2+pt24cu121 \
  torch-geometric==2.6.1 \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

Install the remaining runtime dependencies:

```bash
python -m pip install \
  numpy==1.24.4 scipy==1.10.1 pandas==2.0.3 \
  pytorch-lightning==2.4.0 torchmetrics==1.4.3 torch-ema==0.3 \
  e3nn==0.5.6 \
  hydra-core==1.3.2 hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0 omegaconf==2.3.0 rootutils==1.0.7 \
  lmdb==1.5.1 apsw==3.46.1.0 ase==3.23.0 pyscf==2.7.0 \
  cupy-cuda11x==12.3.0 cutensor-cu11==2.0.2 gpu4pyscf-cuda11x==1.2.0 gpu4pyscf-libxc-cuda11x==0.5 \
  rich==13.9.4 matplotlib==3.7.5 seaborn==0.13.2 scikit-learn==1.3.2 h5py==3.11.0 \
  transformers==4.45.2 tokenizers==0.20.1 safetensors==0.4.5 huggingface-hub==0.25.2 regex==2024.9.11 \
  termcolor==2.4.0 ipykernel notebook nbformat tqdm==4.66.5
```

Register the notebook kernel:

```bash
python -m ipykernel install --user --name hamevo --display-name "Python (hamevo)"
```

Quick validation:

```bash
python -m pip check
python - <<'PY'
import torch, e3nn, cupy, gpu4pyscf, torch_geometric
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
print("e3nn", e3nn.__version__)
print("cupy", cupy.__version__, "devices", cupy.cuda.runtime.getDeviceCount())
print("gpu4pyscf", gpu4pyscf.__version__)
print("torch_geometric", torch_geometric.__version__)
PY
```

Notes:

- `e3nn==0.5.4` is not available from PyPI for Python 3.10; the validated Python 3.10 environment uses `e3nn==0.5.6`.
- `cudatoolkit=11.8` is needed even though PyTorch uses cu121, because `cupy-cuda11x` and `gpu4pyscf-cuda11x` need CUDA 11 runtime libraries such as `libcudart.so.11.0`.

## Data and Checkpoints

Datasets and trained checkpoints are hosted on Hugging Face:

[https://huggingface.co/datasets/ZJUSCL/hamevo-data](https://huggingface.co/datasets/ZJUSCL/hamevo-data)

Some datasets and checkpoints are still being organized for release. Updates will be published under the same Hugging Face link.

Training, evaluation, and inference entry points are:

- `src/train.py`
- `src/eval.py`
- `src/infer.py`
- `notebooks/inference_demo.ipynb`
