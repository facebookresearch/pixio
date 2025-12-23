<h1 align="center">
Pixio
</h1>

<h3 align="center">
A capable vision encoder dedicated to dense tasks, simply by pixel reconstruction
</h3>

<div align="center">

  [![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2512.15715)
  [![Hugging Model Card](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/facebook/pixio)

</div>

---

Official implementation of **Pixio** from the paper [In Pursuit of Pixel Supervision for Visual Pre-training](https://arxiv.org/abs/2512.15715).

[Lihe Yang](https://liheyoung.github.io), [Shang-Wen Li](https://swdanielli.github.io), [Yang Li](https://openreview.net/profile?id=~Yang_Li112), [Xinjie Lei](https://scholar.google.com/citations?user=nIQqtuAAAAAJ), [Dong Wang](https://scholar.google.com/citations?user=ioHfZecAAAAJ), [Abdelrahman Mohamed](https://www.cs.toronto.edu/~asamir), [Hengshuang Zhao](https://hszhao.github.io), [Hu Xu](https://howardhsu.github.io)

[[`BibTeX`](#citation)]

Pixio is largely built on [MAE](https://github.com/facebookresearch/mae), with three minimal yet critical algorithm updates:
- deeper decoder
- larger masking granularity
- more class tokens

Pixio also updates MAE's pre-training data from ImageNet-1K to [MetaCLIP-2B](https://github.com/facebookresearch/MetaCLIP) with a simple self-curation strategy.

<p align="left">
<img src="./assets/pixio.png" width=90% height=90% 
class="center">
</p>

## Performance

**Monocular depth estimation ($\delta_1 \uparrow$, frozen encoder)**

| Method    | ViT   | #Params | NYUv2 (DPT head) | KITTI (DPT head) | NYUv2 (linear head) | KITTI (linear head) |
| :-------- | ----: | ------: | :--------------: | :--------------: | :-----------------: | :-----------------: |
| MAE       | H/14  | 631M    | 80.8             | 90.9             | 70.3                | 79.4                |
| DINOv2    | g/14  | 1137M   | 90.1             | 94.6             | 75.3                | 78.1                |
| DINOv3    | H+/16 | 841M    | 93.2             | 95.6             | 76.3                | 73.2                |
| **Pixio** | H/16  | 631M    | **95.5**         | **96.7**         | **90.8**            | **90.3**            |


**Feed-forward 3D reconstruction ([MapAnything](https://github.com/facebookresearch/map-anything), ScanNet++ v2)**

| Method    | ViT   | #Params | Scale (rel $\downarrow$) | Points (rel $\downarrow$) | Points ($\tau \uparrow$) | Pose (auc@5 $\uparrow$) | Depth (rel $\downarrow$) | Depth ($\tau \uparrow$) |
| :-------- | ----: | ------: | :----------------------: | :-----------------------: | :----------------------: | :--------------------: | :-----------------------: | :-------------------------: |
| MAE       | H/14  | 631M    | 0.050                    | 0.057                     | 63.3                     | 65.6                   | 0.058                     | 55.4                        | 
| DINOv2    | L/14  | 304M    | 0.041                    | 0.052                     | 67.6                     | 73.2                   | 0.052                     | 60.6                        |
| DINOv3    | H+/16 | 841M    | 0.035                    | 0.051                     | 69.0                     | 68.5                   | 0.051                     | 61.2                        |
| **Pixio** | H/16  | 631M    | **0.029**                | **0.041**                 | **78.8**                 | **80.5**               | **0.042**                 | **72.4**                    |

**Semantic segmentation (mIoU $\uparrow$, frozen encoder)**

| Method    | ViT   | #Params | ADE20K (DPT) | VOC (DPT) | LoveDA (DPT) | ADE20K (linear) | VOC (linear) | LoveDA (linear) |
| :-------- | ----: | ------: | :----------: | :-------: | :----------: | :-------------: | :----------: | :-------------: |
| MAE       | H/14  | 631M    | 37.6         | 76.0      | 50.2         | 35.2            | 70.8         | 47.6            | 
| DINOv2    | g/14  | 1137M   | 51.5         | 85.2      | 55.0         | 49.0            | 81.8         | 51.9            |
| DINOv3    | H+/16 | 841M    | 52.3         | 85.6      | **55.3**     | **50.3**        | 82.1         | 52.7            |
| **Pixio** | H/16  | 631M    | **53.6**     | **85.9**  | 54.7         | 50.2            | **82.2**     | **53.9**        |

## Installation

This codebase is developed with PyTorch 2.8.0 + CUDA 12.8.
```bash
conda create -n pixio python=3.10.18
conda activate pixio
pip install -r requirements.txt
```

## Inference (may need Huggingface login)

You can either use *source code* from this repo or call *Transformers APIs*.

### Source Code

Pixio ViT models pre-trained on web-scale dataset (MetaCLIP-2B):

<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Pre-training Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pixio-B/16</td>
      <td align="right">86M</td>
      <td align="center">MetaCLIP-2B</td>
      <td align="center"><a href="https://huggingface.co/facebook/pixio-vitb16/resolve/main/pixio_vitb16.pth">[link]</a></td>
    </tr>
    <tr>
      <td>Pixio-L/16</td>
      <td align="right">303M</td>
      <td align="center">MetaCLIP-2B</td>
      <td align="center"><a href="https://huggingface.co/facebook/pixio-vitl16/resolve/main/pixio_vitl16.pth">[link]</a></td>
    </tr>
    <tr>
      <td>Pixio-H/16</td>
      <td align="right">631M</td>
      <td align="center">MetaCLIP-2B</td>
      <td align="center"><a href="https://huggingface.co/facebook/pixio-vith16/resolve/main/pixio_vith16.pth">[link]</a></td>
    </tr>
    <tr>
      <td>Pixio-1B/16</td>
      <td align="right">1362M</td>
      <td align="center">MetaCLIP-2B</td>
      <td align="center"><a href="https://huggingface.co/facebook/pixio-vit1b16/resolve/main/pixio_vit1b16.pth">[link]</a></td>
    </tr>
    <tr>
      <td>Pixio-5B/16</td>
      <td align="right">5441M</td>
      <td align="center">MetaCLIP-2B</td>
      <td align="center"><a href="https://huggingface.co/facebook/pixio-vit5b16/resolve/main/pixio_vit5b16.pth">[link]</a></td>
    </tr>
  </tbody>
</table>

```bash
cd pixio
```

Then testing as follows:

```python
from PIL import Image
from torchvision import transforms

from pixio import pixio_vith16

model = pixio_vith16(pretrained="your/checkpoint/path")

# you can try larger resolution, but ensure both sides are divisible by 16
transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=3), # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

img = Image.open("your/image/path").convert("RGB")
img = transform(img)

# block-wise features containing class tokens and patch tokens
features = model(img.unsqueeze(0))
```

### Transformers (may need Huggingface login)

You can find all HuggingFace paths under this [collection](https://huggingface.co/collections/facebook/pixio).

```python
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

img = Image.open("your/image/path")

processor = AutoImageProcessor.from_pretrained("facebook/pixio-vith16")
model = AutoModel.from_pretrained("facebook/pixio-vith16")

inputs = processor(images=img, return_tensors="pt")
outputs = model(**inputs)
features_norm = outputs.last_hidden_state # class tokens + patch tokens after last LayerNorm
features = outputs.hidden_states[-1] # class tokens + patch tokens before last LayerNorm
```

## Pre-training

### Data Preparation

We provide examples using ImageNet-1K and ImageNet-21K. We use ImageNet datasets organized as tar files from HuggingFace:

- ImageNet-1K: [download](https://huggingface.co/datasets/timm/imagenet-1k-wds) 
- ImageNet-21K: [download](https://huggingface.co/datasets/timm/imagenet-w21-wds)

### Launch Pre-training

```bash
cd pretraining

# specify your data path in the script
bash scripts/pretrain_pixio_vith16_imagenet.sh
```

## Evaluation

We provide the evaluation code for monocular depth estimation (NYUv2, KITTI), semantic segmentation (ADE20K, Pascal VOC, LoveDA), and *k*-NN classification (ImageNet-1K).

### Data Preparation

<details>
<summary>Click here for details</summary>

#### Monocular Depth Estimation

We follow [ZoeDepth](https://github.com/isl-org/ZoeDepth) and [BTS](https://github.com/cleinc/bts), preparing the data as follows:

- NYUv2: [training set](https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view) | [validation set](https://github.com/cleinc/bts#prepare-nyu-depth-v2-test-set)
- KITTI: [images](https://github.com/cleinc/bts/tree/master/pytorch#kitti) | [annotations](https://github.com/cleinc/bts#prepare-kitti-official-ground-truth-depth-maps)

Please organize the data as follows:
```
├── [Your NYUv2 Path]
    ├── sync
    │   ├── basement_0001a
    │   ├── bathroom_0001
    │   └── ...    
    └── official_splits
        └── test
            ├── bathroom
            ├── bedroom
            └── ...

├── [Your KITTI Path]
    ├── images
    │   ├── 2011_09_26
    │   ├── 2011_09_28
    │   └── ...    
    └── annotations # extracted from data_depth_annotated.zip
        ├── 2011_09_26_drive_0001_sync
        ├── 2011_09_26_drive_0002_sync
        └── ...
```

#### Semantic Segmentation

We mainly follow [UniMatch V2](https://github.com/LiheYoung/UniMatch-V2), preparing the data as follows:

- ADE20K: [images](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) | [annotations](https://drive.google.com/file/d/1f2a4d_mycaI4JCqz-EAVLXVwb6s5EsWa/view?usp=sharing)
- Pascal: [images](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [annotations](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)
- LoveDA: [data](https://www.kaggle.com/datasets/mohammedjaveed/loveda-dataset) (run `evaluation/semseg/util/process_loveda.py` to convert masks)

Please organize the data as follows:
```
├── [Your ADE20K Path]
    ├── images
    │   ├── training
    │   └── validation
    └── annotations
        ├── training
        └── validation

├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass

├── [Your LoveDA Path]
    ├── Train/Train
    └── Val/Val
```

#### *k*-NN Classification

Following [this script](https://gist.github.com/bonlime/4e0d236cf98cd5b15d977dfa03a63643) to prepare ImageNet-1K.

</details>

### Launch Evaluation

```bash
cd evaluation

model="pixio_vith16"
pretrained="your/checkpoint/path"

# specify the data path in config files or script
sbatch launch_monodepth.sh monodepth/configs/nyuv2_dpt.yaml $model $pretrained
sbatch launch_semseg.sh semseg/configs/ade20k_linear.yaml $model $pretrained
sbatch launch_knn.sh $model $pretrained

# or run all evaluations together
bash run_all.sh $model $pretrained
```

## Distillation

### Launch Distillation

```bash
cd distillation

# specify your data path and teacher checkpoint path in the scripts
bash scripts/distill_pixio_vit5b16_to_vit1b16+vith16_imagenet.sh
```

## License
Pixio is licensed under [Facebook license](LICENSE).

## Acknowledgement

We sincerely thank the authors of [MAE](https://github.com/facebookresearch/mae), [DINO](https://github.com/facebookresearch/dino), [DINOv2](https://github.com/facebookresearch/dinov2), and [DINOv3](https://github.com/facebookresearch/dinov3) for open-sourcing their code and models.

## Citation
```bib
@article{pixio,
  title={In Pursuit of Pixel Supervision for Visual Pre-training},
  author={Yang, Lihe and Li, Shang-Wen and Li, Yang and Lei, Xinjie and Wang, Dong and Mohamed, Abdelrahman and Zhao, Hengshuang and Xu, Hu},
  journal={arXiv:2512.15715},
  year={2025}
}
```
