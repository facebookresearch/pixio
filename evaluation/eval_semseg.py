# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# UniMatch V2: https://github.com/liheyoung/unimatch-v2
# --------------------------------------------------------

import argparse
import datetime
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml

from common.misc import setup_distributed, MetricLogger, SmoothedValue
from semseg.data.dataset import SemSegDataset
from semseg.models.dpt import DPTSeg
from semseg.models.linear import LinearSeg
from semseg.util.misc import intersectionAndUnion, CLASSES, AverageMeter


parser = argparse.ArgumentParser(description='Pixio evaluation in semantic segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--encoder', type=str, default='pixio_vith16')
parser.add_argument('--pretrained_ckp', type=str)
parser.add_argument('--dtype', default='bf16', choices=['fp16', 'bf16', 'fp32'])
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


@torch.no_grad()
def evaluate(model, loader, args):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    
    for img, mask in loader:
        img = img.cuda()
        
        crop_size = args.crop_size
        b, _, h, w = img.shape
        stride = int(crop_size * 2 / 3)
        final = torch.zeros(b, args.nclass, h, w).cuda()
        
        # sliding window inference
        row = 0
        while row < h:
            col = 0
            while col < w:
                end_row = min(row + crop_size, h)
                end_col = min(col + crop_size, w)
                
                crop_region = img[:, :, row:end_row, col:end_col]
                
                if crop_region.shape[2] != crop_size or crop_region.shape[3] != crop_size:
                    pad_region = torch.zeros((b, img.shape[1], crop_size, crop_size)).cuda()
                    pad_region[:, :, :crop_region.shape[2], :crop_region.shape[3]] = crop_region
                    crop_region = pad_region
                
                pred = model(crop_region)
                final[:, :, row:end_row, col:end_col] += pred.softmax(dim=1)[:, :, :end_row-row, :end_col-col]
                
                col += stride
            
            row += stride
        
        final = final.argmax(dim=1)
        
        intersection, union, target = intersectionAndUnion(
            final.cpu().numpy(), 
            mask.numpy(), 
            args.nclass, 
            255
        )
        
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-6) * 100.0
    mIOU = np.mean(iou_class)
    
    return mIOU, iou_class


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    for key, value in cfg.items():
        setattr(args, key, value)
    
    rank, world_size = setup_distributed(port=args.port)
    
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        print(pprint.pformat(all_args))
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    if args.head == 'dpt':
        model = DPTSeg(args.encoder, args.pretrained_ckp, args.nclass)
    elif args.head == 'linear':
        model = LinearSeg(args.encoder, args.pretrained_ckp, args.nclass)
    else:
        raise NotImplementedError
    
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad], 
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01
    )
    
    local_rank = int(os.environ['LOCAL_RANK'])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)
    
    trainset = SemSegDataset(args.dataset, args.data_root, 'train', args.crop_size)
    valset = SemSegDataset(args.dataset, args.data_root, 'val')
    
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset, args.batch_size, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler
    )
    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, 1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )
    
    total_iters = len(trainloader) * args.epochs
    previous_best = 0.0
    
    dtype = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32
    }[args.dtype]
    
    for epoch in range(args.epochs):
        model.train()
        
        trainsampler.set_epoch(epoch + 1)
        
        metric_logger = MetricLogger(delimiter='  ')
        metric_logger.add_meter('lr', SmoothedValue(fmt='{value:.6f}'))
        metric_logger.add_meter('loss', SmoothedValue(fmt='{value:.4f} ({avg:.4f})'))
        header = 'Epoch: [{}]'.format(epoch)
        
        for i, (img, mask) in enumerate(metric_logger.log_every(trainloader, 50, header)):
            img, mask = img.cuda(), mask.cuda()
            
            with torch.autocast('cuda', dtype=dtype):
                pred = model(img)
                loss = criterion(pred, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iters = epoch * len(trainloader) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=lr)
            
        with torch.autocast('cuda', dtype=dtype):
            mIoU, iou_class = evaluate(model, valloader, args)
        
        if rank == 0:
            cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for (cls_idx, iou) in enumerate(iou_class):
                classname = CLASSES[args.dataset][cls_idx]
                print(f'[{cur_time}]  Epoch: [{epoch}]  Evaluation >>>> Class [{cls_idx} {classname}] IoU: {iou:.2f}')
            
            print(f'[{cur_time}]  Epoch: [{epoch}]  Evaluation >>>> MeanIoU: {mIoU:.2f}')
            
            previous_best = max(mIoU, previous_best)
            print(f'[{cur_time}]  Epoch: [{epoch}]  Previous best >>>> MeanIoU: {previous_best:.2f}\n')
    
    dist.barrier()

if __name__ == '__main__':
    main()
