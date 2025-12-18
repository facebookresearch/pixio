# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# UniMatch V2: https://github.com/liheyoung/unimatch-v2
# --------------------------------------------------------

import numpy as np
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .transform import resize, crop, hflip, normalize


class SemSegDataset(Dataset):
    def __init__(self, dataset, data_root, mode, input_size=None):
        self.dataset = dataset
        self.data_root = data_root
        self.mode = mode
        self.input_size = input_size
        
        with open(f'semseg/splits/{dataset}/{mode}.txt', 'r') as f:
            self.ids = f.read().splitlines()
    
    def __getitem__(self, item):
        img_path, mask_path = self.ids[item].split(' ')
        img = Image.open(os.path.join(self.data_root, img_path)).convert('RGB')
        mask = Image.open(os.path.join(self.data_root, mask_path))
        
        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask
        
        img, mask = resize(img, mask, (0.5, 2.0))
        img, mask = crop(img, mask, self.input_size, ignore_value=255)
        img, mask = hflip(img, mask, p=0.5)
        
        return normalize(img, mask)

    def __len__(self):
        return len(self.ids)
