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
import glob
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import wandb

from datasets.imagenet import make_imagenet_loader
from datasets.webdataset import make_webdata_loader
from engine_pretrain import train_one_epoch
import models_pixio
import util.misc as misc
import util.optim_factory as optim_factory


def get_args_parser():
    parser = argparse.ArgumentParser('Pixio pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--iters_per_epoch', required=True, type=int)
    parser.add_argument('--save_freq', default=20, type=int)
    
    # Model parameters
    parser.add_argument('--model', default='pixio_vit_1b_enc1536x48h24_dec512x48h16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--n_cls_tokens', default=8, type=int,
                        help='number of class tokens')
    
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='masking ratio (percentage of removed patches)')
    parser.add_argument('--mask_grid', default=4, type=int,
                        help='masking granularity (#patch x #patch)')
    
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    
    parser.add_argument('--drop_path', default=0, type=float,
                        help='drop path ratio')
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2.5e-5, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    
    # Dataset parameters
    parser.add_argument('--dataset_type', default='imagenet', type=str, choices=['imagenet', 'webdataset'],
                        help='dataset type')
    parser.add_argument('--data_path', default=None, type=str,
                        help='dataset root path, images should be stored in tars')
    
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
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    parser.add_argument('--dtype', default='bf16', choices=['fp16', 'bf16', 'fp32'],
                        help='training precision')
    
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

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
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
            project='Pixio',
            config=args,
            name=os.path.basename(args.log_dir),
            dir=args.log_dir,
            resume='allow',
            id=wandb_run_id
        )
    
    # define the model
    model = models_pixio.__dict__[args.model](
        n_cls_tokens=args.n_cls_tokens,
        mask_grid=args.mask_grid,
        drop_path=args.drop_path,
        norm_pix_loss=args.norm_pix_loss
    )
    
    model.to(device)

    model_without_ddp = model
    
    trainable_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    n_param = sum(p.numel() for _, p in trainable_params)
    n_param_encoder = sum(p.numel() for name, p in trainable_params if "decoder" not in name)
    n_param_decoder = sum(p.numel() for name, p in trainable_params if "decoder" in name)
    
    print('Model = %s' % str(model_without_ddp))
    print('number of total params (M): %.2f' % (n_param / 1e6))
    print('number of encoder params (M): %.2f' % (n_param_encoder / 1e6))
    print('number of decoder params (M): %.2f' % (n_param_decoder / 1e6))
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print('base lr: %.2e' % (args.lr * 256 / eff_batch_size))
    print('actual lr: %.2e' % args.lr)

    print('accumulate grad iterations: %d' % args.accum_iter)
    print('effective batch size: %d' % eff_batch_size)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = misc.NativeScalerWithGradNormCount()
    
    misc.load_model(args, model_without_ddp, optimizer, loss_scaler)
    
    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    
    if args.dataset_type == 'webdataset':
        # when resuming training, shuffle tars to avoid always the same order
        data_loader_train.dataset.set_epoch(args.start_epoch)
    data_loader_train = iter(data_loader_train)
    
    for epoch in range(args.start_epoch, args.epochs):
        
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, args
        )
        
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args, epoch, model, model_without_ddp, optimizer, loss_scaler
            )
    
    torch.distributed.barrier()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
