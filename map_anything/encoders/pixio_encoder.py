from functools import partial
from typing import Callable, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pixio.pixio import PixioViT
from map_anything.encoders.base import (
    ViTEncoderInput,
    ViTEncoderOutput,
)

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
print("Enabled torch fsdp", torch.backends.cuda.flash_sdp_enabled())


class PixioEncoder(PixioViT):

    def __init__(
        self,
        data_norm_type: str,
        size: str = "huge",
        patch_size: int = 16,
        img_size: int = 256,
        in_chans: int = 3,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        n_cls_tokens: int | None = None,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        layerscale=None,
        pretrained_checkpoint_path: str = "",
        gradient_checkpointing: bool = True,
        *args,
    ):
        """
        Base class for all Vision Transformer encoders in UniCeption.
        """
        self.patch_size = patch_size

        self.enc_embed_dim = {"base": 1024, "huge": 1280, "1b": 1536}[size]
        depth = {"base": 24, "huge": 32, "1b": 48}[size]
        n_cls_tokens = (
            {"base": 1, "huge": 4, "1b": 4}[size]
            if n_cls_tokens is None
            else n_cls_tokens
        )

        super().__init__(
            img_size=img_size,
            patch_size=patch_size, 
            in_chans=in_chans,
            embed_dim=self.enc_embed_dim, 
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio, 
            n_cls_tokens=n_cls_tokens,
            norm_layer=norm_layer
        )

        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        if self.pretrained_checkpoint_path:
            print(
                f"Loading custom pretrained Pixio Encoder checkpoint from {self.pretrained_checkpoint_path} ..."
            )
            ckpt = torch.load(
                self.pretrained_checkpoint_path,
                map_location='cpu',
                weights_only=False,
            )
            print(self.load_state_dict(ckpt, strict=False))

        if gradient_checkpointing:
            for i in range(len(self.blocks)):
                self.blocks[i] = self.wrap_module_with_gradient_checkpointing(
                    self.blocks[i]
                )

    def wrap_module_with_gradient_checkpointing(self, module: nn.Module):
        # Return a thin wrapper module with checkpoint applied
        class _CheckpointingWrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, *args, **kwargs):
                return checkpoint(
                    self.inner.forward, *args, use_reentrant=False, **kwargs
                )

        return _CheckpointingWrapper(module)

    def forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:

        # Check image normalization type
        # self._check_data_normalization_type(encoder_input.data_norm_type)

        # Check the dtype and shape of the input image
        assert isinstance(
            encoder_input.image, torch.Tensor
        ), "Input must be a torch.Tensor"
        assert encoder_input.image.ndim == 4, "Input must be of shape (B, C, H, W)"
        batch_size, channels, height, width = encoder_input.image.shape
        assert channels == 3, "Input must have 3 channels"
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input shape must be divisible by patch size: {self.patch_size}"

        encoder_output = super().forward(encoder_input.image)
        features = encoder_output[-1]['patch_tokens_norm']

        features = features.permute(0, 2, 1)
        # ddp_gather_and_print("features's shape after permute", shape={features.shape})
        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()
        return ViTEncoderOutput(features=features)


