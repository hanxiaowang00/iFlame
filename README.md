# iFlame🔥: Interleaving Full and Linear Attention for Efficient Mesh Generation ⚡

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/hanxiaowang00/iFlame?style=social)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)



**A single-GPU trainable unconditional mesh generative model** 🚀

For more information, visit the project page: [iFlame Project Page](https://hanxiaowang00.github.io/iFlame/)
arXiv: [https://arxiv.org/abs/2503.16653](https://arxiv.org/abs/2503.16653)
</div>

---

## ✨ Overview

iFlame is a 3D mesh generation framework that introduces a novel approach by strategically interleaving full and linear attention mechanisms. This innovative technique dramatically reduces computational requirements while preserving exceptional output quality, enabling effective training on just a single GPU.

## 🛠️ Requirements

This project has been thoroughly tested on:
- 🔥 PyTorch 2.5.1
- 🖥️ CUDA 12.1

### 📦 Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/hanxiaowang00/iFlame.git

# Step 2: Navigate into the project directory
cd iFlame

# Step 3: Create a new Conda environment with Python 3.10
conda create -n iFlame python=3.10

# Step 4: Activate the Conda environment
conda activate iFlame

# Step 5: Install the CUDA toolkit (version 12.1)
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

# Step 6: Install PyTorch and its related packages
pip install torch==2.5.1 torchvision torchaudio

# Step 7: Install flash-attn without build isolation
pip install flash-attn --no-build-isolation

# Step 8: Install additional dependencies from requirements.txt
pip install -r requirements.txt
```




## 📊 Dataset

This project leverages the ShapeNet dataset processed by [MeshGPT](https://github.com/audi/MeshGPT) 

<details>
<summary>📥 Dataset Preparation</summary>

1. Download the processed dataset from the MeshGPT repository
2. Place the dataset in the same directory level as the iFlame project
3. The model expects the data to be in the format processed by MeshGPT

</details>
If you want to train on Objaverse, we also provide an ID list (`objaverse_list.txt`) for your convenience.


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

We provide the ShapeNet checkpoint, which can be found at the following link:

[ShapeNet Checkpoint](https://huggingface.co/spaces/hxddd/iFlame/blob/main/padded_datashapenet.npy40unet24mix0.001muonsmall.pth)


## 📝 Citation

If you use this code in your research, please cite:
```bibtex
@article{wang2025iflameinterleavinglinearattention,
      title={{iFlame}: Interleaving Full and Linear Attention for Efficient Mesh Generation}, 
      author={Hanxiao Wang and Biao Zhang and Weize Quan and Dong-Ming Yan and Peter Wonka},
      year={2025},
      eprint={2503.16653},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.16653}, 
}
```

## 📜 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

- Thanks to the authors of [MeshGPT](https://github.com/audi/MeshGPT) for the dataset preprocessing
- Implementation built upon foundations from Shape2VecSet, Lightning Attention, and Flash Attention
- Developed with support from the research community

---

<div align="center">
  <b>✨ Star this repository if you find it useful! ✨</b>
</div>
