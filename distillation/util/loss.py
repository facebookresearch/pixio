# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


def feat_align_loss(stu_feat, tea_feat, ids_keep, args):
    stu_feat = stu_feat.float()
    tea_feat = tea_feat.float()
    
    n_cls_tokens = args.n_cls_tokens
    n_tokens = stu_feat.shape[1]
    n_patch_tokens = n_tokens - n_cls_tokens
    
    stu_cls = stu_feat[:, :n_cls_tokens]
    stu_patch = stu_feat[:, n_cls_tokens:]
    
    tea_cls = tea_feat[:, :n_cls_tokens]
    tea_patch = tea_feat[:, n_cls_tokens:]
    
    if ids_keep is not None:
        ids_keep = ids_keep.unsqueeze(-1).repeat(1, 1, tea_patch.shape[-1])
        tea_patch = torch.gather(tea_patch, dim=1, index=ids_keep)
    
    # computer alignment loss for class tokens
    if args.loss_fn_cls == 'cosine':
        loss_cls = 1 - F.cosine_similarity(stu_cls, tea_cls, dim=-1).mean()
    elif args.loss_fn_cls == 'l1':
        loss_cls = F.l1_loss(stu_cls, tea_cls)
    elif args.loss_fn_cls == 'mse':
        loss_cls = F.mse_loss(stu_cls, tea_cls)
    else:
        raise NotImplementedError
    
    # computer alignment loss for patch tokens
    if args.loss_fn_patch == 'cosine':
        loss_patch = 1 - F.cosine_similarity(stu_patch, tea_patch, dim=-1).mean()
    elif args.loss_fn_patch == 'l1':
        loss_patch = F.l1_loss(stu_patch, tea_patch)
    elif args.loss_fn_patch == 'mse':
        loss_patch = F.mse_loss(stu_patch, tea_patch)
    else:
        raise NotImplementedError
    
    if args.loss_fuse == 'sum':
        loss = loss_cls * (n_cls_tokens / n_tokens) + loss_patch * (n_patch_tokens / n_tokens)
    elif args.loss_fuse == 'avg':
        loss = (loss_cls + loss_patch) / 2.0
    
    return loss, loss_cls, loss_patch
