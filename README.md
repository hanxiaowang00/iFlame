# iFlame🔥: Interleaving Full and Linear Attention for Efficient Mesh Generation ⚡

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/yourusername/iFlame?style=social)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A single-GPU trainable unconditional mesh generative model** 🚀

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=iFlame+Mesh+Generation" alt="iFlame Demo" width="600"/>
</p>

</div>

---

## ✨ Overview

iFlame is a cutting-edge 3D mesh generation framework that employs a novel approach by interleaving full and linear attention mechanisms. This approach significantly reduces computational requirements while maintaining high-quality outputs, making it feasible to train on a single GPU.

## 🛠️ Requirements

This project has been tested on:
- 🔥 PyTorch 2.5.1
- 🖥️ CUDA 12.1

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/iFlame.git
cd iFlame

# Install dependencies
pip install -r requirements.txt
```

### 📋 Dependencies (requirements.txt)

```
torch==2.5.1
numpy>=1.20.0
tqdm>=4.62.0
networkx>=2.6.3
trimesh>=3.9.0
einops>=0.4.1
timm>=0.6.0
flash-attn>=2.0.0
lightning-attn>=1.0.0
wandb>=0.13.0
pathlib>=1.0.1
```

## 📊 Dataset

This project uses the ShapeNet dataset processed by [MeshGPT](https://github.com/audi/MeshGPT) 🏆

<details>
<summary>📥 Dataset Preparation</summary>

1. Download the processed dataset from the MeshGPT repository
2. Place the dataset in the same directory level as the iFlame project
3. The model expects the data to be in the format processed by MeshGPT

</details>

## 🚀 Usage

### 🏋️‍♂️ Training

<table>
<tr>
<td>Single-GPU Training:</td>
<td>

```bash
python iFlame.py 1
```

</td>
</tr>
<tr>
<td>Multi-GPU Training (e.g., 4 GPUs):</td>
<td>

```bash
python iFlame.py 4
```

</td>
</tr>
</table>

### 🔍 Testing/Inference

To generate meshes using a trained checkpoint:

```bash
python test.py "path/to/checkpoint"
```

## 🧠 Model Architecture

<div align="center">
  <img src="https://via.placeholder.com/800x300?text=iFlame+Architecture" alt="Model Architecture" width="700"/>
</div>

iFlame uses an innovative attention mechanism that interleaves full attention with linear attention to achieve efficient mesh generation while maintaining high quality outputs:

- 🔄 Alternating layers of full and linear attention
- ⚡ Optimized memory usage and computational efficiency
- 🔍 Fine-grained detail preservation despite reduced complexity

## 📝 Citation

If you use this code in your research, please cite:
```bibtex
@article{iflame2023,
  title={iFlame: Interleaving Full and Linear Attention for Efficient Mesh Generation},
  author={Wang, Hanxiao and Zhang, Biao and Quan, Weize and Yan, Dong-Ming and Wonka, Peter},
  journal={arXiv preprint},
  year={2023}
}
```

## 📜 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

- Thanks to the authors of [MeshGPT](https://github.com/audi/MeshGPT) for the dataset preprocessing
- the  code is based on is shape2vecset, lightning attention, flash attention 
- Developed with support from the research community

---

<div align="center">
  <b>✨ Star this repository if you find it useful! ✨</b>
</div>

