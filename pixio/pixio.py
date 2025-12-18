# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
from typing import Optional, Type

import torch
import torch.nn as nn

from layers.attention import SelfAttentionBlock
from layers.mlp import Mlp
from layers.patch_embed import PatchEmbed


class PixioViT(nn.Module):
    """Pixio ViT encoder"""
    def __init__(
        self,
        img_size: int = 256, 
        patch_size: int = 16, 
        in_chans: int = 3,
        embed_dim: int = 1280, 
        depth: int = 32, 
        num_heads: int = 16,
        mlp_ratio: int = 4., 
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        n_cls_tokens: int = 8
    ):
        super().__init__()
        
        self.n_cls_tokens = n_cls_tokens
        
        self.patch_embed = PatchEmbed(
            img_size, 
            patch_size, 
            in_chans,
            embed_dim
        )
        
        self.cls_token = nn.Parameter(
            torch.zeros(1, n_cls_tokens, embed_dim)
        )
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + n_cls_tokens, embed_dim)
        )
        
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                embed_dim, 
                num_heads, 
                mlp_ratio, 
                qkv_bias=True, 
                norm_layer=norm_layer, 
                mlp_layer=Mlp
            ) for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
    def _interpolate_pos_emb(
        self, 
        x: torch.Tensor
    ):
        """Interpolate the pre-trained positional embeddings to match the input x"""
        assert x.shape[-2] % self.patch_embed.patch_size[0] == 0, \
            f'height {x.shape[-2]} must be divisible by patch size {self.patch_embed.patch_size[0]}'
        assert x.shape[-1] % self.patch_embed.patch_size[1] == 0, \
            f'width {x.shape[-1]} must be divisible by patch size {self.patch_embed.patch_size[1]}'
        
        H = x.shape[-2] // self.patch_embed.patch_size[0]
        W = x.shape[-1] // self.patch_embed.patch_size[1]
        
        cls_pos_embed = self.pos_embed[:, :self.n_cls_tokens]
        patch_pos_embed = self.pos_embed[:, self.n_cls_tokens:]
        
        pt_size = int(patch_pos_embed.shape[1] ** 0.5)
        
        if pt_size == H == W:
            return self.pos_embed
        
        patch_pos_embed = patch_pos_embed.reshape(1, pt_size, pt_size, -1).permute(0, 3, 1, 2)
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed, size=(H, W), mode='bicubic', align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, H * W, -1)
        
        new_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
        
        return new_pos_embed
    
    def forward(
        self, 
        x: torch.Tensor, 
        block_ids: Optional[list[int]] = None
    ):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            block_ids (Optional[List[int]]): Specific blocks indices to extract features from. If None, extracts from all blocks.
        
        Returns:
            List[dict]: List of feature dictionaries, one per specified block.
        """
        pos_embed = self._interpolate_pos_emb(x)
        
        x = self.patch_embed(x)

        x = x + pos_embed[:, self.n_cls_tokens:, :]
        
        cls_token = self.cls_token + pos_embed[:, :self.n_cls_tokens, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        features = []
        if block_ids is None:
            block_ids = list(range(len(self.blocks)))
        
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            
            if i in block_ids:
                features.append({
                    'patch_tokens': x[:, self.n_cls_tokens:],
                    'cls_tokens': x[:, :self.n_cls_tokens],
                    'patch_tokens_norm': self.norm(x)[:, self.n_cls_tokens:],
                    'cls_tokens_norm': self.norm(x)[:, :self.n_cls_tokens]
                })
        
        return features


def pixio_vitb16(pretrained=None):
    model = PixioViT(
        img_size=256,
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12,
        mlp_ratio=4, 
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )

    if pretrained:
        state_dict = torch.load(pretrained, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
    
    return model


def pixio_vitl16(pretrained=None):
    model = PixioViT(
        img_size=256,
        patch_size=16, 
        embed_dim=1024, 
        depth=24, 
        num_heads=16,
        mlp_ratio=4, 
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    if pretrained:
        state_dict = torch.load(pretrained, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
    
    return model


def pixio_vith16(pretrained=None):
    model = PixioViT(
        img_size=256,
        patch_size=16, 
        embed_dim=1280, 
        depth=32, 
        num_heads=16,
        mlp_ratio=4, 
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    if pretrained:
        state_dict = torch.load(pretrained, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
    
    return model


def pixio_vit1b16(pretrained=None):
    model = PixioViT(
        img_size=256,
        patch_size=16, 
        embed_dim=1536, 
        depth=48, 
        num_heads=24,
        mlp_ratio=4, 
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    if pretrained:
        state_dict = torch.load(pretrained, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        
    return model


def pixio_vit5b16(pretrained=None):
    model = PixioViT(
        img_size=256,
        patch_size=16, 
        embed_dim=3072, 
        depth=48,
        num_heads=32,
        mlp_ratio=4, 
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    if pretrained:
        state_dict = torch.load(pretrained, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        
    return model
