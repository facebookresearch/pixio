# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import glob
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import wandb

from datasets.webdataset import make_webdata_loader
from datasets.imagenet import make_imagenet_loader
from engine_distill import train_one_epoch
import models_distill
import util.misc as misc
import util.optim_factory as optim_factory


def get_args_parser():
    parser = argparse.ArgumentParser('Pixio model distillation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--iter_per_epoch', default=None, type=int)
    parser.add_argument('--save_freq', default=20, type=int)
    
    # Model parameters
    parser.add_argument('--teacher_model', default='pixio_vit5b16', type=str,
                        help='Name of teacher model to produce pre-trained features')
    parser.add_argument('--teacher_pretrained', required=True, type=str,
                        help='teacher model pre-trained path')
    parser.add_argument('--student_models', default='pixio_vith16', nargs='+', type=str,
                        help='Name of student models to learn from teacher model')
    
    parser.add_argument('--n_cls_tokens', default=8, type=int,
                        help='number of class (register) tokens')
    
    parser.add_argument('--drop_path', default=0, type=float,
                        help='drop path ratio for student models')
    
    # Distillation parameters
    parser.add_argument('--loss_fn_cls', type=str, default='cosine', choices=['none', 'cosine', 'l1', 'mse'],
                        help='distillation loss for class tokens')
    parser.add_argument('--loss_fn_patch', type=str, default='cosine', choices=['cosine', 'l1', 'mse'],
                        help='distillation loss for patch tokens')
    parser.add_argument('--loss_fuse', type=str, default='avg', choices=['sum', 'avg'],
                        help='strategy for fusing class token loss and patch token loss')
    
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_grid', default=4, type=int,
                        help='mask grid size (#patch x #patch)')
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    
    # Dataset parameters
    parser.add_argument('--dataset_type', default='imagenet', type=str, choices=['imagenet', 'webdataset'],
                        help='dataset type')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--crop_scale_min', default=0.2, type=float,
                        help='images input size')
    
    parser.add_argument('--json_path', default=None, type=str,
                        help='json root path of pre-computed image statistics for data curation')
    parser.add_argument('--soft_sampling', action='store_true',
                        help='sample training images based on their reconstruction loss')
    parser.set_defaults(soft_sampling=False)
    parser.add_argument('--hist_thresh', default=None, type=float,
                        help='image color histogram entropy threshold for data curation')
    parser.add_argument('--imgres_thresh', default=None, type=int,
                        help='image resolution threshold for data curation')
    
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to wandb log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--dtype', default='bf16',
                        help='training precision', choices=['fp16', 'bf16', 'fp32'])
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    rank = misc.get_rank()
    
    # fix the seed for reproducibility
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True

    if args.dataset_type == 'imagenet':
        data_loader_train = make_imagenet_loader(args)
    else:
        data_loader_train = make_webdata_loader(args)
    
    global_rank = misc.get_rank()
    
    if global_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        
        wandb_path = glob.glob(os.path.join(args.log_dir, 'wandb', 'run*'))
        wandb_run_id = os.path.basename(wandb_path[0]).split('-')[-1] if len(wandb_path) != 0 else None
        
        wandb.init(
            project='Pixio-distill',
            config=args,
            name=os.path.basename(args.log_dir),
            dir=args.log_dir,
            resume='allow',
            id=wandb_run_id
        )
    
    print(f'=== Initializing teacher model: {args.teacher_model} ===')
    teacher_model = models_distill.__dict__[args.teacher_model](
        img_size=args.input_size,
        n_cls_tokens=args.n_cls_tokens
    )
    print(teacher_model)
    
    state_dict = torch.load(args.teacher_pretrained, map_location='cpu', weights_only=False)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    teacher_model.load_state_dict(state_dict, strict=False)
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    teacher_model.to(device)
    
    student_models = {}
    student_models_without_ddp = {}
    optimizers = {}
    loss_scalers = {}
    
    for student_name in args.student_models:
        print(f'=== Initializing student model: {student_name} ===')
        
        # Create student model
        student_model = models_distill.__dict__[student_name](
            img_size=args.input_size,
            n_cls_tokens=args.n_cls_tokens,
            drop_path=args.drop_path,
            mask_grid=args.mask_grid,
            output_dim=teacher_model.embed_dim,
        )
        
        print(student_model)
        
        student_model.to(device)
        model_without_ddp = student_model
        n_parameters = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        print(f'Model = {student_name}')
        print(f'Number of params (M): {n_parameters / 1.e6:.2f}')
        if args.distributed:
            student_model = torch.nn.parallel.DistributedDataParallel(
                student_model, device_ids=[args.gpu], find_unused_parameters=True
            )
            model_without_ddp = student_model.module
        
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(f'Optimizer for {student_name}: {optimizer}')
        
        loss_scaler = misc.NativeScalerWithGradNormCount()
        
        misc.load_model(args=args, model=student_model, optimizer=optimizer, 
                       loss_scaler=loss_scaler, prefix=f'{student_name}_')
        
        student_models[student_name] = student_model
        student_models_without_ddp[student_name] = model_without_ddp
        optimizers[student_name] = optimizer
        loss_scalers[student_name] = loss_scaler
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print('base lr: %.2e' % (args.lr * 256 / eff_batch_size))
    print('actual lr: %.2e' % args.lr)
    
    print('accumulate grad iterations: %d' % args.accum_iter)
    print('effective batch size: %d' % eff_batch_size)
    
    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    
    if args.dataset_type == 'webdataset':
        # when resuming training, shuffle tars to avoid always the same order
        data_loader_train.dataset.set_epoch(args.start_epoch)
    data_loader_train = iter(data_loader_train)
    
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            student_models, teacher_model, data_loader_train,
            optimizers, device, epoch, loss_scalers, args
        )
        
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            for student_name in args.student_models:
                misc.save_model(
                    args=args, 
                    epoch=epoch,
                    model=student_models[student_name], 
                    model_without_ddp=student_models_without_ddp[student_name], 
                    optimizer=optimizers[student_name],
                    loss_scaler=loss_scalers[student_name],
                    prefix=f'{student_name}_'
                )
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
