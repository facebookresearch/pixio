# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DINO: https://github.com/facebookresearch/dino
# distributed setup: https://github.com/BIGBALLON/distribuuuu
# --------------------------------------------------------

import argparse
from collections import defaultdict, deque
import datetime
import os
import random
import subprocess
import time

import torch
import torch.distributed as dist


def load_pretrained_ckp(model, pretrained_ckp):
    if not pretrained_ckp:
        return
    
    state_dict = torch.load(pretrained_ckp, map_location='cpu', weights_only=False)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    model.load_state_dict(state_dict, strict=False)


def setup_distributed(backend='nccl', port=None):
    num_gpus = torch.cuda.device_count()

    if 'SLURM_JOB_ID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        # specify master port
        if port is not None:
            os.environ['MASTER_PORT'] = str(port)
        elif 'MASTER_PORT' not in os.environ:
            job_id = int(os.environ['SLURM_JOB_ID'])
            random.seed(job_id)
            port = random.randint(20000, 30000)
            os.environ['MASTER_PORT'] = str(port)
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(rank % num_gpus)
        os.environ['RANK'] = str(rank)
    else:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f'cuda:{local_rank}')
    )
    return rank, world_size



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.6f} ({global_avg:.6f})'
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
        

class MetricLogger(object):
    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                '{}: {}'.format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                '[{current_time}]',
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                '[{current_time}]',
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if (i % print_freq == 0 or i == len(iterable) - 1) and dist.get_rank() == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), current_time=current_time,
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), current_time=current_time,
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if dist.get_rank() == 0:
            print('[{}]  {}  Total time: {} ({:.4f} s / it)'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), header, total_time_str, total_time / len(iterable)))
