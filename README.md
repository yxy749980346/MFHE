# MFHE: Multi-Feature Fusion and Hyperbolic Embedding for Liver Registration

This repository contains the **official PyTorch implementation** of:

**[Liver point cloud registration via multi-feature fusion and hyperbolic embedding for augmented reality surgical navigation](https://www.sciencedirect.com/science/article/abs/pii/S0169260726002051?via%3Dihub)**  
Xiangyue Yang, Baochun He, Yue Dai, Huoling Luo, Lei Wang, and Fucang Jia  
*Computer Methods and Programs in Biomedicine*, Volume 284, Article 109451, 2026.  

## Installation

The code has been **tested successfully on Ubuntu 24.04 with PyTorch 2.0.0 and CUDA 11.8**.

### 1. Create environment and install dependencies

```bash
conda create -n mfhe python=3.10 -y
conda activate mfhe
pip install "setuptools<70" wheel ninja
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Configure CUDA

```bash
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
```

### 3. Build extensions

```bash
python setup.py build develop
cd pareconv/extensions/pointops
python setup.py install
cd ../../..
```

## Data Preparation

Configure the dataset in [`experiments/Liver/config.py`](experiments/Liver/config.py). Replace each `your-path` placeholder with the corresponding local path.

### 1. 3Dircadb

```python
_C.data.dataset = '3Dircadb'
_C.data.dataset_root = 'your-path'
```

Organize the data by patient and sample:

```text
your-path
├── 03
│   ├── 0001
│   │   ├── surface.stl
│   │   ├── partialSurface.stl
│   │   └── transform.npy
│   └── ...
├── 05
│   └── ...
└── ...
```

- `surface.stl`: source, preoperative liver surface.
- `partialSurface.stl`: target, partial intraoperative liver surface.
- `transform.npy`: a 4 x 4 rigid transformation mapping the source to the target.
- Point coordinates and translation values are expressed in **meters**.
- Patients **03 and 05** form the test set. Samples from the remaining patients are split into training and validation sets at **80:20**, using `split_seed`.

### 2. P2I-LReg

P2I-LReg was introduced in **[Landmark-Free Preoperative-to-Intraoperative Registration in Laparoscopic Liver Resection](https://doi.org/10.1109/TMI.2025.3574198)** (IEEE Transactions on Medical Imaging, 2025). Please refer to the authors' official [Self-P2IR repository](https://github.com/junzastar/Self-P2IR) for dataset access and preparation instructions, and cite their paper when using this dataset.

We converted the dataset's point clouds to `.pth` files for multi-worker loading with PyTorch DataLoader (`num_workers`).

```python
_C.data.dataset = 'P2I-LReg'
_C.data.dataset_root = 'your-path'
```

## Training

Select the dataset and set its root in `experiments/Liver/config.py`, then run:

```bash
cd experiments/Liver
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

To resume an existing training run:

```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py --resume
```

For multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 trainval.py
```

## Register Your Own Point Clouds

Run from the repository root with a checkpoint trained using this code:

```bash
python experiments/Liver/register_pair_demo.py \
    --src your-path \
    --ref your-path \
    --checkpoint your-path \
    --input-unit m \
    --transform-source model \
    --output-transform transform.npy
```

`--src` and `--ref` specify the preoperative and intraoperative surfaces. The output maps source to target, with translation in the selected input unit (meters for `--input-unit m`, millimeters for `--input-unit mm`). Use `--visualize` to view the result, or `--help` for more options.

## Real-Data Projection Evaluation

[`eval_real_rigid_projection_dice.py`](experiments/Liver/eval_real_rigid_projection_dice.py) computes projection Dice scores for the real P2I-LReg data. It requires the external [Self-P2IR](https://github.com/junzastar/Self-P2IR) code, rendering dependencies, and real-data assets. Set `SELF_P2IR_ROOT` before use.

## Citation

If you find this work useful, please cite:

```bibtex
@article{yang2026liver,
  title   = {Liver point cloud registration via multi-feature fusion and hyperbolic embedding for augmented reality surgical navigation},
  author  = {Yang, Xiangyue and He, Baochun and Dai, Yue and Luo, Huoling and Wang, Lei and Jia, Fucang},
  journal = {Computer Methods and Programs in Biomedicine},
  volume  = {284},
  pages   = {109451},
  year    = {2026},
  doi     = {10.1016/j.cmpb.2026.109451}
}
```

## Acknowledgements

Our code is built upon the implementations of the following projects:

- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [PARE-Net](https://github.com/yaorz97/PARENet)
- [HECPG](https://github.com/IvanXie416/HECPG)

We sincerely thank their authors for sharing their code and for their valuable contributions to point cloud registration.

We also thank the authors of **[Self-P2IR](https://github.com/junzastar/Self-P2IR)**, *[Landmark-Free Preoperative-to-Intraoperative Registration in Laparoscopic Liver Resection](https://doi.org/10.1109/TMI.2025.3574198)*, for providing the P2I-LReg dataset and releasing their implementation. We further thank **[LiverMatch](https://github.com/zixinyang9109/LiverMatch)** for sharing its code and advancing preoperative-to-intraoperative liver point cloud matching.
