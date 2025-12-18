# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from layers.attention import SelfAttentionBlock
from layers.mlp import Mlp
from layers.patch_embed import PatchEmbed


class PixioViT(nn.Module):
    """
    Pixio ViT encoder and decoder for pre-training
    """
    def __init__(
        self, 
        img_size=256, 
        patch_size=16, 
        in_chans=3,
        embed_dim=1280, 
        depth=32, 
        num_heads=16,
        decoder_embed_dim=512, 
        decoder_depth=32, 
        decoder_num_heads=16,
        mlp_ratio=4., 
        norm_layer=nn.LayerNorm, 
        n_cls_tokens=8,
        mask_grid=4, 
        drop_path=0,
        norm_pix_loss=True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_cls_tokens = n_cls_tokens
        self.mask_grid = mask_grid
        self.norm_pix_loss = norm_pix_loss
        
        # --------------------------------------------------------------------------
        # Pixio encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, 
            patch_size,
            in_chans, 
            embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, n_cls_tokens, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + n_cls_tokens, embed_dim)
        )
        
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
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------
        # Pixio decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + n_cls_tokens, decoder_embed_dim)
        )
        
        self.decoder_blocks = nn.ModuleList([
            SelfAttentionBlock(
                decoder_embed_dim, 
                decoder_num_heads,
                mlp_ratio, 
                qkv_bias=True, 
                norm_layer=norm_layer,
                mlp_layer=Mlp
            ) for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        # --------------------------------------------------------------------------
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.decoder_pos_embed, std=0.02)
        
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
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = imgs.shape[-2] // p
        w = imgs.shape[-1] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
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
        ids_masked = ids_shuffle[:, len_keep:]
        
        patch_grid = torch.arange(H * W, device=x.device).view(1, H, W)
        patch_grid = patch_grid.unfold(1, grid, grid).unfold(2, grid, grid)
        patch_grid = patch_grid.contiguous().view(1, -1, grid, grid)
        
        ids_keep_expanded = patch_grid[:, ids_keep].view(N, -1)
        ids_masked_expanded = patch_grid[:, ids_masked].view(N, -1)
        
        x_masked = torch.gather(
            x.view(N, -1, D), 
            dim=1, 
            index=ids_keep_expanded.unsqueeze(-1).repeat(1, 1, D)
        )
        
        ids_restore = torch.cat((ids_keep_expanded, ids_masked_expanded), dim=1)
        ids_restore = torch.argsort(ids_restore, dim=1)
        
        mask = torch.ones([N, H * W], device=x.device)
        mask[:, :len_keep * (grid ** 2)] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        ori_h, ori_w = x.shape[-2:]
        
        H = ori_h // self.patch_embed.patch_size[0]
        W = ori_w // self.patch_embed.patch_size[1]
        
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, self.n_cls_tokens:, :]
        
        cls_token = self.cls_token + self.pos_embed[:, :self.n_cls_tokens, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, self.mask_grid)
        
        # append cls token
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], 
            ids_restore.shape[1] + self.n_cls_tokens - x.shape[1], 
            1
        )
        
        x_ = torch.cat((x[:, self.n_cls_tokens:, :], mask_tokens), dim=1)  # no cls token
        x_ = torch.gather(
            x_, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        
        x = torch.cat((x[:, :self.n_cls_tokens, :], x_), dim=1)  # append cls token
        
        # add pos embed
        x = x + self.decoder_pos_embed
        
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)
        
        x = x[:, self.n_cls_tokens:, :]
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(self, imgs, mask_ratio):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        
        return loss


def pixio_vith16_enc1280x32h16_dec512x32h16(**kwargs):
    model = PixioViT(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=32, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def pixio_vit1b16_enc1536x48h24_dec512x32h16(**kwargs):
    model = PixioViT(
        patch_size=16, embed_dim=1536, depth=48, num_heads=24,
        decoder_embed_dim=512, decoder_depth=32, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def pixio_vit5b16_enc3072x48h32_dec512x32h16(**kwargs):
    model = PixioViT(
        patch_size=16, embed_dim=3072, depth=48, num_heads=32,
        decoder_embed_dim=512, decoder_depth=32, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
