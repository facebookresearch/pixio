# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DINO: https://github.com/facebookresearch/dino
# --------------------------------------------------------

import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
from torchvision import models as torchvision_models

import common.misc as misc
import common.pixio as pixio


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, 'train'), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, 'val'), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f'Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.')
    
    # ============ building network ... ============
    model = pixio.__dict__[args.model]()
    misc.load_pretrained_ckp(model, args.pretrained_ckp)
    model.cuda()
    model.eval()
    
    # ============ extract features ... ============
    print('Extracting features for train set...')
    train_features = extract_features(model, data_loader_train, args)
    print('Extracting features for val set...')
    test_features = extract_features(model, data_loader_val, args)
    
    if dist.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)
    
    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long().cuda()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long().cuda()
    
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, args):
    metric_logger = misc.MetricLogger(delimiter='  ')
    features = None
    
    dtype = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32
    }[args.dtype]
    print(f'Using {args.dtype} for feature extraction')
    
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=dtype), torch.no_grad():
            rets = model(samples)[-1]
            feats = rets['cls_tokens'].mean(dim=1)
        
        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1]).cuda()
            print(f'Storing features into tensor of shape {features.shape}')
        
        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)
        
        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            features.index_copy_(0, index_all, torch.cat(output_l))
    
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)
        
        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
        
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    
    return top1, top5


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pixio evaluation in ImageNet k-NN classification')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')
    
    parser.add_argument('--model', default='pixio_vith16', type=str, help='model name')
    parser.add_argument('--pretrained_ckp', default='', type=str, help='Path to pretrained checkpoint to evaluate.')
    parser.add_argument('--dtype', default='bf16', choices=['fp16', 'bf16', 'fp32'],
                        help='inference precision')
    
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--local_rank', default=0, type=int, help='Please ignore and do not set this argument.')
    parser.add_argument('--port', default=None, type=int, help='network port')
    args = parser.parse_args()
    
    misc.setup_distributed(port=args.port)
    cudnn.benchmark = True
    
    train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)
    
    if dist.get_rank() == 0:
        print('Features are ready!\nStart the k-NN classification.')
        for k in args.nb_knn:
            top1, top5 = knn_classifier(
                train_features,
                train_labels,
                test_features, 
                test_labels, 
                k, 
                args.temperature
            )
            print(f'{k}-NN classifier result: Top1: {top1}, Top5: {top5}')
    
    dist.barrier()
