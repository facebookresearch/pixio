# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# UniMatch V2: https://github.com/liheyoung/unimatch-v2
# MiDaS: https://github.com/isl-org/midas
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import common.misc as misc
import common.pixio as pixio
from semseg.models.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        out_channels=[256, 512, 1024, 1024],
        use_bn=False
    ):
        super(DPTHead, self).__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, out_features):
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv(path_1)
        
        return out


class DPTDepth(nn.Module):
    def __init__(
        self, 
        encoder, 
        pretrained_ckp,
        features=256, 
        out_channels=[256, 512, 1024, 1024],
        use_bn=False
    ):
        super(DPTDepth, self).__init__()
        
        self.encoder = pixio.__dict__[encoder]()
        misc.load_pretrained_ckp(self.encoder, pretrained_ckp)
        
        self.head = DPTHead(
            1, self.encoder.embed_dim * 2, features, out_channels, use_bn
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
        
        # use four intermediate features, concatenate patch tokens and averaged class token
        features = [rets[len(rets) // 4 * (stage_i + 1) - 1]  for stage_i in range(4)]
        features = [
            torch.cat((
                feat['patch_tokens_norm'], 
                feat['cls_tokens_norm'].mean(dim=1, keepdim=True).expand(-1, feat['patch_tokens_norm'].shape[1], -1)
            ), dim=2) for feat in features
        ]
        
        features = (
            feat.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[-1], patch_h, patch_w) for feat in features
        )
        
        out = self.head(features)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        out = F.sigmoid(out)
        
        return out.squeeze(1)
