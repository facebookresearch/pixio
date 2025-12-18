# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Dict

import torch
import torch.nn.functional as F
import wandb

import util.misc as misc
from util.loss import feat_align_loss
import util.lr_sched as lr_sched


def train_one_epoch(
    student_models: Dict[str, torch.nn.Module],
    teacher_model: torch.nn.Module,
    data_loader: Iterable, 
    optimizers: Dict[str, torch.optim.Optimizer],
    device: torch.device, 
    epoch: int, 
    loss_scalers: Dict[str, misc.NativeScalerWithGradNormCount],
    args=None
):
    for student_model in student_models.values():
        student_model.train(True)
    
    student_names = list(student_models.keys())
    
    metric_logger = misc.MetricLogger(delimiter='  ')
    for student_name in student_names:
        metric_logger.add_meter(f'{student_name}_lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'{student_name}_loss_cls', misc.SmoothedValue())
        metric_logger.add_meter(f'{student_name}_loss_patch', misc.SmoothedValue())
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    accum_iter = args.accum_iter
    # Zero gradients for all optimizers
    for optimizer in optimizers.values():
        optimizer.zero_grad()
    iter_per_epoch = args.iter_per_epoch
    
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header, iter_per_epoch)):
        
        if data_iter_step % accum_iter == 0:
            for optimizer in optimizers.values():
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / iter_per_epoch + epoch, args)
        
        samples = samples.to(device, non_blocking=True)
        
        dtype = {
            'bf16': torch.bfloat16, 
            'fp16': torch.float16, 
            'fp32': torch.float32
        }[args.dtype]
        
        # Compute teacher features once (shared by all students)
        with torch.autocast('cuda', dtype=dtype):
            with torch.no_grad():
                tea_feat, _ = teacher_model(samples, norm=False)
        
        # Train each student model
        for student_name in student_names:
            student_model = student_models[student_name]
            optimizer = optimizers[student_name]
            loss_scaler = loss_scalers[student_name]
            
            with torch.autocast('cuda', dtype=dtype):
                stu_feat, ids_keep = student_model(samples, args.mask_ratio, norm=True)
                
                loss, loss_cls, loss_patch = feat_align_loss(stu_feat, tea_feat, ids_keep, args)
            
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value} for {student_name}, stopping training')
                sys.exit(1)
            
            loss /= accum_iter
            
            loss_scaler(
                loss, optimizer, 
                parameters=student_model.parameters(),
                update_grad=(data_iter_step + 1) % accum_iter == 0
            )
            
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
            
            torch.cuda.synchronize()
            
            metric_logger.update(**{f'{student_name}_loss_cls': loss_cls.item()})
            metric_logger.update(**{f'{student_name}_loss_patch': loss_patch.item()})
            
            lr = optimizer.param_groups[0]['lr']
            metric_logger.update(**{f'{student_name}_lr': lr})
            
            loss_cls_value_reduce = misc.all_reduce_mean(loss_cls.item())
            loss_patch_value_reduce = misc.all_reduce_mean(loss_patch.item())
            
            if misc.is_main_process() and (data_iter_step + 1) % accum_iter == 0:
                wandb.log({
                    f'{student_name}_train_loss_cls': loss_cls_value_reduce,
                    f'{student_name}_train_loss_patch': loss_patch_value_reduce,
                    f'{student_name}_lr': lr,
                    'epoch': (data_iter_step / iter_per_epoch) + epoch,
                })
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
