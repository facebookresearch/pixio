# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import os
import sys
from typing import Iterable

import torch
import wandb

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable, 
    optimizer: torch.optim.Optimizer,
    device: torch.device, 
    epoch: int, 
    loss_scaler: misc.NativeScalerWithGradNormCount,
    args
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    
    accum_iter = args.accum_iter

    optimizer.zero_grad()
    
    iters_per_epoch = args.iters_per_epoch
    
    dtype = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32
    }[args.dtype]
    
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header, iters_per_epoch)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / iters_per_epoch + epoch, args)
        
        samples = samples.to(device, non_blocking=True)
        
        with torch.autocast('cuda', dtype=dtype):
            loss = model(samples, args.mask_ratio)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print('Loss is {}'.format(loss_value))
            sys.exit(1)
        
        loss /= accum_iter
        
        loss_scaler(
            loss, 
            optimizer, 
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        
        metric_logger.update(loss=loss.item())
        
        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if misc.is_main_process() and (data_iter_step + 1) % accum_iter == 0:
            wandb.log(
                {
                    'train_loss': loss_value_reduce,
                    'lr': lr,
                    'epoch': (data_iter_step / iters_per_epoch) + epoch
                }
            )
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
