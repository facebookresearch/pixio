# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MiDaS: https://github.com/isl-org/midas
# --------------------------------------------------------

import numpy as np
import random


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "valid_mask" in sample:
            sample["valid_mask"] = sample["valid_mask"].astype(np.float32)
            sample["valid_mask"] = np.ascontiguousarray(sample["valid_mask"])
        
        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)
        
        return sample


class Crop(object):
    """Crop sample for batch-wise training. Image is of shape CxHxW
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample['image'].shape[:2]
        assert h >= self.size[0] and w >= self.size[1], 'Wrong size'
        
        h_start = np.random.randint(0, h - self.size[0] + 1)
        w_start = np.random.randint(0, w - self.size[1] + 1)
        h_end = h_start + self.size[0]
        w_end = w_start + self.size[1]
        
        sample['image'] = sample['image'][h_start: h_end, w_start: w_end, :]
        
        if "depth" in sample:
            sample["depth"] = sample["depth"][h_start: h_end, w_start: w_end]
        
        if "valid_mask" in sample:
            sample["valid_mask"] = sample["valid_mask"][h_start: h_end, w_start: w_end]
        
        return sample


class HorizontalFlip(object):
    """Randomly flip the image and depth map horizontally
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        if random.random() < 0.5:
            sample['image'] = np.flip(sample['image'], axis=1)
            
            if "depth" in sample:
                sample["depth"] = np.flip(sample["depth"], axis=1)
            
            if "valid_mask" in sample:
                sample["valid_mask"] = np.flip(sample["valid_mask"], axis=1)
        
        return sample
