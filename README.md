<think>The user is asking me to write a more professional-looking GitHub README.md for their project called "iFlame: Interleaving Full and Linear Attention for Efficient Mesh Generation". The README should include:

1. Project title and brief description
2. Dependency information (tested on PyTorch 2.5.1 with CUDA 12.1)
3. Dataset instructions (using ShapeNet dataset processed by MeshGPT)
4. Training instructions (single-GPU vs multi-GPU)
5. Testing instructions

Let me create a clean, professional README.md with proper formatting, sections, and badges where appropriate.</think># iFlame: Interleaving Full and Linear Attention for Efficient Mesh Generation

A single-GPU trainable unconditional mesh generative model that efficiently generates 3D meshes using interleaved full and linear attention mechanisms.

## Overview

iFlame is designed for efficient mesh generation with optimized attention mechanisms, making it feasible to train on a single GPU while producing high-quality 3D meshes.

## Requirements

This project has been tested on:
- PyTorch 2.5.1
- CUDA 12.1

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/iFlame.git
cd iFlame

# Install dependencies
pip install -r requirements.txt
```

### Dependencies (requirements.txt)

```
torch==2.5.1
numpy
tqdm
networkx
trimesh
einops
timm
flash-attn
lightning-attn
wandb
pathlib
```

## Dataset

This project uses the ShapeNet dataset processed by [MeshGPT](https://github.com/audi/MeshGPT).

1. Download the processed dataset from MeshGPT repository
2. Place the dataset in the same directory level as the iFlame project

## Usage

### Training

For single-GPU training:
```bash
python iFlame.py 1
```

For multi-GPU training (e.g., 4 GPUs):
```bash
python iFlame.py 4
```

### Testing/Inference

To generate meshes using a trained checkpoint:
```bash
python test.py "path/to/checkpoint"
```

## Citation

If you use this code in your research, please cite:
```
@article{iflame2025,
  title={iFlame: Interleaving Full and Linear Attention for Efficient Mesh Generation},
  author={Your Name},
  journal={Wang, Hanxiao and Zhang, Biao and Quan, Weize and Yan, Dong-Ming and Wonka, Peter},
  year={2025}
}
```

## License

[MIT License](LICENSE)
