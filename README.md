# PDG-based Image Representation

This repository provides a **PDG-based CUDA library**, as well as code for **image representation experiments** and **auto-encoder/tokenizer experiments**.

---

## Quick Start

### Environment

- Ubuntu 24.04
- CUDA 12.8
- Python 3.10
- PyTorch 2.7.1+cu128

---

## Installation

### 1) Install `pdg_gsplat`

```bash
pip install pdg_gsplat-0.0.0-cp310-cp310-linux_x86_64.whl
```

### 2) Install dependencies for Gaussianimage_plus

```bash
cd gaussianimage_plus_ours

cd gsplat
pip install .[dev]
cd ../
pip install -r requirements.txt
```

### 3) Install dependencies for Downstream task: GaussianToken

```bash
cd GaussianToken-master
```

#### 3.1 Install Python dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-extra.txt
```

#### 3.2 Install `gsplat` & `deformable attn` modules

```bash
cd gstk/modules/gsplat && python setup.py build install
cd ../gaussianembed/ops && python setup.py build install
```

------

### 4) Notes

- `gsplat` and our `pdg_gsplat` require strict compatibility between **CUDA** and **PyTorch built-in CUDA**.
- Installing `pdg_gsplat` **only supports CUDA 12.8**.
- **GaussianToken** and **GaussianImage++** require different `gsplat` versions, so they must be installed independently (recommended: separate conda environments).

------

## Pretrained Models

- We provide checkpoints for **Table 1** (Gaussian / PDG / PDG+STE for GaussianImage++).
   Link: https://zenodo.org/records/21523140.
   Please download and place them under `gaussianimage_plus_ours/`.
- We provide checkpoints for **Table 4** (VQGAN / GaussianToken / GaussianToken+PDG).
   Link: https://zenodo.org/records/21614182.
   Please place them under:
  - `GaussianToken-master/`
  - `GaussianToken-master-PDG/`

------

## Data

### Image representation experiments

Please download:

- [DIV2K test set](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- [Kodak dataset](https://r0k.us/graphics/kodak/)
- [ImageTexture dataset](https://github.com/NYU-ICL/image-gs)

### GaussianToken experiments

Follow the GaussianToken instructions to download:

- [CIFAR-100](https://cave.cs.toronto.edu/kriz/cifar.html)
- [Mini-ImageNet](https://www.kaggle.com/datasets/jiajundong/mini-imagenet/)

Then add the following environment variables to your `.bashrc`:

```bash
# dataset env
export DATASET_ROOT="<path-to-dataset>"
export MINI_IMAGENET_ROOT="${DATASET_ROOT}/mini-imagenet"
export CIFAR100_ROOT="${DATASET_ROOT}"
export IMAGENET_ROOT="${DATASET_ROOT}/imagenet"
```

Apply the changes:

```bash
source ~/.bashrc
```

------

## Commands

### 1) Image representation

- Test (to reproduce Table 1 results for `gaussianimage_plus`):

```bash
python test.py
```

- Train:

```bash
python train.py --num_points 2500 --max_num_points 5000 --data_name kodak -d ./dataset/kodak/
```

You can modify the dataset path and the number of primitives to obtain different results.

### 2) Auto-encoder / tokenizer experiments (GaussianToken)

Checkpoint mapping:
- `logs_paper/gqgan` → GaussianToken
- `logs_paper/gqgan_PDG` → GaussianToken + PDG
- `logs_paper/vqgan` → VQGAN

Evaluate (to reproduce Table 4 results for `GaussianToken`):

```bash
bash ./scripts/val-1.sh
```

Train:

```bash
bash ./scripts/cifar-gqgan-1.sh
bash ./scripts/mini-gqgan-1.sh
```
------

## Acknowledgments

We thank the following projects for their open-source contributions:

- GaussianImage (gsplat support):
   https://github.com/Xinjie-Q/GaussianImage
- GaussianImage++:
   https://github.com/Sweethyh/GaussianImage_plus
- LIG:
   https://github.com/HKU-MedAI/LIG
- Image-GS:
   https://github.com/NYU-ICL/image-gs
- SmartSplat:
   https://github.com/lif314/SmartSplat
- GaussianToken:
   https://github.com/ChrisDong-THU/GaussianToken
- GSASR:
   https://github.com/ChrisDud0257/GSASR
- 2DGS_inpaint:
   https://github.com/lihy715/2DGS_inpaint
- GaussianVision:
   https://github.com/Tambe-Lab/GaussianVision
