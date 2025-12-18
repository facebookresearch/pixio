# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MetaCLIP: https://github.com/facebookresearch/metaclip
# --------------------------------------------------------

import glob
from io import BytesIO
import json
import os
from PIL import Image, ImageFile
import random
import tarfile

import torch
from torchvision import transforms

from util.misc import get_rank, get_world_size

ImageFile.LOAD_TRUNCATED_IMAGES = True


class WebDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        data_path, 
        json_path,
        resolution,
        crop_scale,
        crop_ratio,
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
        hist_thresh=None,
        imgres_thresh=None,
        soft_sampling=False
    ):
        
        self.data_path = data_path
        self.json_path = json_path
        
        self.shard_ids = glob.glob(os.path.join(data_path, '**/*.tar'), recursive=True)
        print(f'Found {len(self.shard_ids)} tars')
        
        self.shard_ids.sort()
        self.num_shards = len(self.shard_ids)
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(resolution, crop_scale, crop_ratio, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
        
        self.hist_thresh = hist_thresh
        self.imgres_thresh = imgres_thresh
        self.soft_sampling = soft_sampling
 
    def set_epoch(self, epoch):
        random.seed(epoch)
        random.shuffle(self.shard_ids)
    
    def _get_tarball_path(self, shard_id):
        return self.shard_ids[shard_id]

    def _get_next_shard_id(self, shard_id):
        next_shard_id = (shard_id + self.worker_size) % self.num_shards
        self.global_shard_id = next_shard_id
        return self.global_shard_id

    def _get_worker_info(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        global_rank = get_rank()
        world_size = get_world_size()
        
        self.worker_size = int(num_workers * world_size)
        
        return (global_rank, worker_id), num_workers

    def __iter__(self):
        (global_rank, worker_id), num_workers = self._get_worker_info()
        
        self.global_shard_id = global_rank * num_workers + worker_id
        shard_id = self.global_shard_id
        
        while True:
            tarball_path = self._get_tarball_path(shard_id)
            if not os.path.exists(tarball_path):
                shard_id = self._get_next_shard_id(shard_id)
                continue
            
            if self.json_path:
                json_path = os.path.join(
                    self.json_path, 
                    tarball_path.split('/')[-2], tarball_path.split('/')[-1].replace('.tar', '.json')
                )
                if not os.path.exists(json_path):
                    shard_id = self._get_next_shard_id(shard_id)
                    continue
            
                with open(json_path, 'r') as f:
                    img_infos = json.load(f)
            
            with tarfile.open(tarball_path) as tar:
                members = tar.getmembers()
                
                for member in members:
                    if not member.name.endswith('.jpeg'):
                        continue
                    
                    uuid = member.name
                    
                    if self.json_path and uuid not in img_infos:
                        continue
                    
                    if self.hist_thresh or self.imgres_thresh or self.soft_sampling:
                        img_info = img_infos[uuid]
                        
                        if self.hist_thresh:
                            hist = img_info['histogram_score']
                            if hist <= self.hist_thresh:
                                continue
                        
                        if self.imgres_thresh:
                            if img_info['height'] < self.imgres_thresh or img_info['width'] < self.imgres_thresh:
                                continue
                        
                        if self.soft_sampling:
                            hardness = img_info['mae_loss']
                            if random.random() > hardness:
                                continue
                    
                    with tar.extractfile(member) as f:
                        img = f.read()
                    
                    with Image.open(BytesIO(img)) as img:
                        img = img.convert('RGB')
                        img = self.transform(img)
                    yield img
            
            shard_id = self._get_next_shard_id(shard_id)


def make_webdata_loader(args):
    dataset = WebDataset(
        data_path=args.data_path,
        json_path=args.json_path,
        resolution=args.input_size,
        crop_scale=(args.crop_scale_min, 1.0),
        crop_ratio=(3/4, 4/3),
        hist_thresh=args.hist_thresh,
        imgres_thresh=args.imgres_thresh,
        soft_sampling=args.soft_sampling,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        sampler=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader
