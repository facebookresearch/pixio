# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob

from torch.utils.data import default_collate
from torchvision import transforms
import webdataset as wds


def make_imagenet_loader(args):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, (args.crop_scale_min, 1.0), (3/4, 4/3), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    def make_sample(sample):
        img = sample['jpg'].convert('RGB')
        return transform_train(img)
    
    filelist = sorted(glob.glob(f'{args.data_path}/*train*.tar'))
    assert len(filelist) != 0, 'No tar files found'
    print(f'Found {len(filelist)} tar files')
    
    dataset_train = wds.WebDataset(
        filelist,
        resampled=True,
        shardshuffle=True,
        workersplitter=wds.split_by_worker,
        nodesplitter=wds.split_by_node,
    )
    
    dataset_train = (
        dataset_train
        .select(lambda sample: 'jpg' in sample)
        .shuffle(10000)
        .decode('pil')
        .map(make_sample)
    )

    dataset_train = dataset_train.batched(args.batch_size, collation_fn=default_collate)
    data_loader_train = wds.WebLoader(dataset_train, batch_size=None, num_workers=args.num_workers)
    
    return data_loader_train
