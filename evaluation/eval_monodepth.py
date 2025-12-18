# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import os
import pprint
import yaml

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW

from common.misc import setup_distributed, MetricLogger, SmoothedValue
from monodepth.data.kitti import KITTI
from monodepth.data.nyuv2 import NYUv2
from monodepth.models.dpt import DPTDepth
from monodepth.models.linear import LinearDepth
from monodepth.util.misc import SiLogLoss, eval_depth


parser = argparse.ArgumentParser(description='Pixio evaluation in monocular depth estimation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--encoder', type=str, default='pixio_vith16')
parser.add_argument('--pretrained_ckp', type=str)
parser.add_argument('--dtype', default='bf16', choices=['fp16', 'bf16', 'fp32'])
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


@torch.no_grad()
def infer_model(model, img, args):
    crop_size = args.crop_size
    b, _, h, w = img.shape
    stride = int(crop_size * 2 / 3)
    final = torch.zeros(b, h, w).cuda()
    weight = torch.zeros(b, h, w).cuda()
    
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
            final[:, row:end_row, col:end_col] += pred[:, :end_row-row, :end_col-col]
            weight[:, row:end_row, col:end_col] += 1
            
            col += stride
        row += stride
    
    pred = final / weight
    
    return pred * args.max_depth


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
        model = DPTDepth(args.encoder, args.pretrained_ckp)
    elif args.head == 'linear':
        model = LinearDepth(args.encoder, args.pretrained_ckp)
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
    
    criterion = SiLogLoss().cuda(local_rank)
    
    if args.dataset == 'nyuv2':
        trainset = NYUv2(args.data_root, 'train', args.crop_size)
    elif args.dataset == 'kitti':
        trainset = KITTI(args.data_root, 'train', args.crop_size)
    else:
        raise NotImplementedError
    
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset, args.batch_size, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler
    )
    
    if args.dataset == 'nyuv2':
        valset = NYUv2(args.data_root, 'val')
    elif args.dataset == 'kitti':
        valset = KITTI(args.data_root, 'val')
    else:
        raise NotImplementedError
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, 1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )
    
    total_iters = args.epochs * len(trainloader)
    
    previous_best = {
        'd1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100
    }
    
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
        
        warmup_iters = int(args.epochs * len(trainloader) * 0.1)
        
        for i, sample in enumerate(metric_logger.log_every(trainloader, 50, header)):
            img = sample['image'].cuda()
            depth = sample['depth'].cuda()
            valid_mask = sample['valid_mask'].cuda()
            
            with torch.autocast('cuda', dtype=dtype):
                pred = model(img) * args.max_depth
                
                loss = criterion(
                    pred.clamp(min=args.min_depth), 
                    depth, 
                    (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
                )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iters = epoch * len(trainloader) + i
            if iters < warmup_iters:
                lr = args.lr * (iters / warmup_iters)
            else:
                lr = args.lr * (1 - (iters - warmup_iters) / (total_iters - warmup_iters)) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=lr)
        
        model.eval()
        
        results = {
            'd1': torch.tensor(0.0).cuda(), 
            'd2': torch.tensor(0.0).cuda(),
            'd3': torch.tensor(0.0).cuda(), 
            'abs_rel': torch.tensor(0.0).cuda(), 
            'sq_rel': torch.tensor(0.0).cuda(), 
            'rmse': torch.tensor(0.0).cuda(), 
            'rmse_log': torch.tensor(0.0).cuda(),
            'log10': torch.tensor(0.0).cuda(),
            'silog': torch.tensor(0.0).cuda()
        }
        nsamples = torch.tensor(0.0).cuda()
        
        for i, sample in enumerate(valloader):
            
            img = sample['image'].cuda().float()
            depth = sample['depth'].cuda()[0]
            valid_mask = sample['valid_mask'].cuda()[0]
            
            with torch.autocast('cuda', dtype=dtype):
                pred = infer_model(model, img, args)[0]
            
            valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            
            if valid_mask.sum() < 10:
                continue
            
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1
        
        torch.distributed.barrier()
        
        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)
        
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if rank == 0:
            print(f'[{cur_time}]  Epoch: [{epoch}]  Evaluation:')
            print(f'[{cur_time}]  ==========================================================================================')
            print(f'[{cur_time}]  ' + '{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
            print(f'[{cur_time}]  ' + '{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, '
                  '{:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
            print(f'[{cur_time}]  ==========================================================================================')
        
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
            
        if rank == 0:
            print(f'[{cur_time}]  Epoch: [{epoch}]  Previous best:')
            print(f'[{cur_time}]  ==========================================================================================')
            print(f'[{cur_time}]  ' + '{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(previous_best.keys())))
            print(f'[{cur_time}]  ' + '{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, '
                  '{:8.3f}, {:8.3f}'.format(*tuple(previous_best.values())))
            print(f'[{cur_time}]  ==========================================================================================\n')
    
    dist.barrier()
    

if __name__ == '__main__':
    main()
