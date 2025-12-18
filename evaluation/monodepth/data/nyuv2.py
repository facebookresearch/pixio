# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ZoeDepth: https://github.com/isl-org/zoedepth
# --------------------------------------------------------

import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import Crop, HorizontalFlip, NormalizeImage, PrepareForNet


class NYUv2(Dataset):
    def __init__(self, data_root, mode, input_size=None):
        self.data_root = data_root
        self.mode = mode
        
        with open(f'monodepth/splits/nyuv2/{mode}.txt', 'r') as f:
            self.ids = f.read().splitlines()
        
        if mode == 'train':
            self.transform = Compose([
                Crop(input_size),
                HorizontalFlip(),
                NormalizeImage((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                PrepareForNet(),
            ])
        elif mode == 'val':
            self.transform = Compose([
                NormalizeImage((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                PrepareForNet(),
            ])
        else:
            raise NotImplementedError
    
    def __getitem__(self, item):
        img_path, depth_path = self.ids[item].split(' ')
        
        img_path = os.path.join(self.data_root, img_path)
        depth_path = os.path.join(self.data_root, depth_path)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')
        depth = depth / 1000.0
        
        # Eigen crop, follow ZoeDepth
        image = image[45:471, 41:601]
        depth = depth[45:471, 41:601]
        
        valid_mask = depth > 0
        
        sample = self.transform({'image': image, 'depth': depth, 'valid_mask': valid_mask})
        
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['valid_mask'] = torch.from_numpy(sample['valid_mask'])
        
        return sample

    def __len__(self):
        return len(self.ids)
