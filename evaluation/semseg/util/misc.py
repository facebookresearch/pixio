# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# UniMatch V2: https://github.com/liheyoung/unimatch-v2
# --------------------------------------------------------

import logging
import os

import numpy as np


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


CLASSES = {
    'ade20k': 
        [
            'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
            'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
            'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
            'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
            'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
            'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
            'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
            'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
            'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
            'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
            'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
            'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
            'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
            'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
            'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
            'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
            'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
            'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
            'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
            'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
            'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 
            'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
            'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag'
        ],
        
    'pascal': 
        [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
            'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 
            'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
        ],
        
    'loveda': [
        'background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture'
    ]
}