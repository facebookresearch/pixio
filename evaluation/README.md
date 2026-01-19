# Evaluation

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


### Evaluate 3rd-party Models

External models need to have implementations on forwarding calls (`rets = self.encoder(x)`) and return a list of (layers) of dict (wit keys `patch_tokens`, `cls_tokens`, `patch_tokens_norm` and `cls_tokens_norm`) similar to `pixio.py`.

You can wrap external models as 
```python
def setup_inference():
    import types
    model = your_model()
    model.load_state_dict(torch.load(pretrained))
    model.forward = types.MethodType(your_forward_func, model)
    return model
```

and pass the `encoder` argument as the format `<repo_path>:<module_name>:setup_inference`

`python eval_all.py` to print on multiple models or `python eval_all.py y` to launch jobs for evaluating on multiple models.


## Citation
```bib
@article{pixio,
  title={In Pursuit of Pixel Supervision for Visual Pre-training},
  author={Yang, Lihe and Li, Shang-Wen and Li, Yang and Lei, Xinjie and Wang, Dong and Mohamed, Abdelrahman and Zhao, Hengshuang and Xu, Hu},
  journal={arXiv:},
  year={2025}
}
```
