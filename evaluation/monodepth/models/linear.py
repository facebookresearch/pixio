# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

import common.misc as misc
import common.pixio as pixio


class LinearDepth(nn.Module):
    def __init__(
        self, 
        encoder, 
        pretrained_ckp
    ):
        super(LinearDepth, self).__init__()
        
        self.encoder = pixio.__dict__[encoder]()
        misc.load_pretrained_ckp(self.encoder, pretrained_ckp)
        
        self.head = nn.Sequential(
            nn.BatchNorm2d(self.encoder.embed_dim * 2),
            nn.Conv2d(self.encoder.embed_dim * 2, 1, 1)
        )
        
        self.lock_encoder()
        
    def lock_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        h, w = x.shape[-2:]
        patch_size = self.encoder.patch_embed.patch_size[0]
        patch_h, patch_w = h // patch_size, w // patch_size
        
        rets = self.encoder(x)
        
        # use the last block features, concatenate patch tokens and averaged class token
        features = torch.cat((
            rets[-1]['patch_tokens_norm'], 
            rets[-1]['cls_tokens_norm'].mean(dim=1, keepdim=True).expand(-1, rets[-1]['patch_tokens_norm'].shape[1], -1)
        ), dim=2)
        
        features = features.permute(0, 2, 1).reshape(features.shape[0], features.shape[-1], patch_h, patch_w)
        
        out = self.head(features)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        out = F.sigmoid(out)
        
        return out.squeeze(1)

