# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from layers.attention import SelfAttentionBlock
from layers.mlp import Mlp
from layers.patch_embed import PatchEmbed


class PixioViT(nn.Module):
    def __init__(
        self, 
        img_size=256, 
        patch_size=16, 
        in_chans=3,
        embed_dim=1280, 
        depth=32, 
        num_heads=16,
        mlp_ratio=4., 
        norm_layer=nn.LayerNorm, 
        n_cls_tokens=8, 
        drop_path=0,
        mask_grid=4,
        output_dim=None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, n_cls_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + n_cls_tokens, embed_dim))
        
        self.n_cls_tokens = n_cls_tokens
        self.mask_grid = mask_grid
        
        drop_path = np.linspace(0, drop_path, depth).tolist()
        
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                embed_dim, 
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                drop_path=drop_path[i], 
                mlp_layer=Mlp
            ) for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        if output_dim is None:
            self.last_proj = nn.Identity()
        else:
            self.last_proj = nn.Sequential(
                nn.Linear(embed_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
            
        trunc_normal_(self.pos_embed, std=0.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def random_masking(self, x, mask_ratio, grid):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        grid: masking granularity, measured in #patches x #patches
        """
        N, L, D = x.shape
        
        H = W = int(L ** 0.5)
        x = x.view(N, H, W, D)
        
        num_patches = (H // grid) * (W // grid)
        len_keep = int(num_patches * (1 - mask_ratio))
        
        noise = torch.rand(N, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        
        patch_grid = torch.arange(H * W, device=x.device).view(1, H, W)
        patch_grid = patch_grid.unfold(1, grid, grid).unfold(2, grid, grid)
        patch_grid = patch_grid.contiguous().view(1, -1, grid, grid)
        
        ids_keep_expanded = patch_grid[:, ids_keep].view(N, -1)
        
        x_masked = torch.gather(
            x.view(N, -1, D), 
            dim=1, 
            index=ids_keep_expanded.unsqueeze(-1).repeat(1, 1, D)
        )
        
        return x_masked, ids_keep_expanded
    
    def forward(self, x, mask_ratio=0, norm=True):
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, self.n_cls_tokens:, :]
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :self.n_cls_tokens, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        
        ids_keep = None
        if mask_ratio > 0:
            x, ids_keep = self.random_masking(x, mask_ratio, self.mask_grid)
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        
        if norm:
            x = self.norm(x)
        
        x = self.last_proj(x)
        
        return x, ids_keep


def pixio_vitb16(**kwargs):
    model = PixioViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def pixio_vitl16(**kwargs):
    model = PixioViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def pixio_vith16(**kwargs):
    model = PixioViT(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def pixio_vit1b16(**kwargs):
    model = PixioViT(
        patch_size=16, embed_dim=1536, depth=48, num_heads=24,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def pixio_vit5b16(**kwargs):
    model = PixioViT(
        patch_size=16, embed_dim=3072, depth=48, num_heads=32,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
