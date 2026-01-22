"""
Encoder class for Dense Representation Encoder
"""

import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from map_anything.encoders.base import (
    ViTEncoderInput,
    ViTEncoderNonImageInput,
    ViTEncoderOutput,
)


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class ResidualBlock(nn.Module):
    "Redidual block for Dense Representation Encoder"

    def __init__(self, in_channels: int, out_channels: int, act_layer: Type[nn.Module] = nn.GELU):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out += identity

        return self.act(out)


class DenseRepresentationEncoder(nn.Module):
    "UniCeption Dense Representation Encoder"

    def __init__(
        self,
        name: str,
        in_chans: int = 3,
        enc_embed_dim: int = 1024,
        apply_pe: bool = True,
        input_size_for_pe: Union[int, Tuple[int, int]] = 518,
        patch_size: int = 14,
        intermediate_dims: List[int] = [588, 768, 1024],
        data_norm_type: str = "dense_rep_encoder",
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Optional[Callable] = partial(nn.LayerNorm, eps=1e-6),
        post_pe_norm_layer: Optional[Callable] = partial(nn.LayerNorm, eps=1e-6),
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
        pretrained_checkpoint_path: str = None,
        *args,
        **kwargs,
    ):
        """
        Dense Representation Encoder for extracting patch-wise features from a spatial input of size (B, C, H, W).
        Uses a convolution based patchify followed by some residual blocks.
        Also applies positional encoding with interpolation to the patch-wise features if required.

        Args:
            in_chans (int): Number of input channels.
            enc_embed_dim (int): Embedding dimension of the encoder.
            apply_pe (bool): Whether to apply positional encoding.
            input_size_for_pe (Union[int, Tuple[int, int]]): Input size for positional encoding.
            patch_size (int): Patch size of the encoder.
            intermediate_dims (List[int]): Intermediate dimensions of the encoder.
            data_norm_type (str): Data normalization type. (Used for checking if the input images are normalized correctly.)
            act_layer (Type[nn.Module]): Activation layer.
            norm_layer (Optional[Callable]): Normalization layer.
            post_pe_norm_layer (Optional[Callable]): Normalization layer after positional encoding.
            interpolate_antialias (bool): Whether to apply antialiasing in interpolation.
            interpolate_offset (float): Offset for interpolation.
            pretrained_checkpoint_path (str): Path to pretrained checkpoint.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.data_norm_type = data_norm_type
        self.patch_size = patch_size

        # Init the specific attributes
        self.in_chans = in_chans
        self.enc_embed_dim = enc_embed_dim
        self.intermediate_dims = intermediate_dims
        self.apply_pe = apply_pe

        # Initialize the encoder with a pixel unshuffle and conv projection to patchify the input
        self.unshuffle = nn.PixelUnshuffle(self.patch_size)
        self.conv_in = nn.Conv2d(self.in_chans * (self.patch_size**2), self.intermediate_dims[0], 3, 1, 1)

        # Add residual blocks
        layers = []
        for intermediate_idx in range(len(self.intermediate_dims) - 1):
            layers.append(
                ResidualBlock(
                    in_channels=self.intermediate_dims[intermediate_idx],
                    out_channels=self.intermediate_dims[intermediate_idx + 1],
                    act_layer=act_layer,
                )
            )

        # Final projection to match encoder embeddings dim
        layers.append(
            nn.Conv2d(
                in_channels=self.intermediate_dims[-1],
                out_channels=self.enc_embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.encoder = nn.Sequential(*layers)

        # Init norm layer after encoder if required
        self.norm_layer = norm_layer(enc_embed_dim) if norm_layer else nn.Identity()
        if isinstance(self.norm_layer, nn.LayerNorm):
            nn.init.constant_(self.norm_layer.bias, 0)
            nn.init.constant_(self.norm_layer.weight, 1.0)

        if self.apply_pe:
            # Init the patch resolution details required for positional encoding
            patch_HW = make_2tuple(patch_size)
            self.input_size_for_pe = make_2tuple(input_size_for_pe)
            self.patches_resolution = (
                self.input_size_for_pe[0] // patch_HW[0],
                self.input_size_for_pe[1] // patch_HW[1],
            )
            self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

            # Init the sinusodial positional encodings
            self.register_buffer(
                "pos_embed",
                self._get_sinusoid_encoding_table(self.num_patches, self.enc_embed_dim, 70007),
            )
            self.interpolate_antialias = interpolate_antialias
            self.interpolate_offset = interpolate_offset

            # Init the norm layer after positional encoding if required
            self.post_pe_norm = post_pe_norm_layer(enc_embed_dim) if post_pe_norm_layer else nn.Identity()
            if isinstance(self.post_pe_norm, nn.LayerNorm):
                nn.init.constant_(self.post_pe_norm.bias, 0)
                nn.init.constant_(self.post_pe_norm.weight, 1.0)

        # Load the pretrained checkpoint if provided
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        if self.pretrained_checkpoint_path:
            print(
                f"Loading custom pretrained Dense Representation Encoder checkpoint from {self.pretrained_checkpoint_path} ..."
            )
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def _get_sinusoid_encoding_table(self, n_position, d_hid, base):
        "Sinusoid position encoding table"

        def get_position_angle_vec(position):
            return [position / np.power(base, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table)

    def interpolate_pos_encoding(self, features, height, width):
        """
        Interpolate the positional encoding to the expected size.

        Args:
            features (torch.Tensor): Input tensor of shape (B, N, C).
            height (int, float): Height of the input tensor.
            width (int, float): Width of the input tensor.

        Returns:
            torch.Tensor: Interpolated positional encoding tensor of shape (1, N, C).
        """
        previous_dtype = features.dtype
        npatch = features.shape[1]
        N = self.pos_embed.unsqueeze(0).shape[1]
        if npatch == N and height == width:
            return self.pos_embed.unsqueeze(0)
        patch_pos_embed = self.pos_embed.unsqueeze(0).float()
        dim = features.shape[-1]
        height0 = height // self.patch_size
        width0 = width // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sh = float(height0 + self.interpolate_offset) / M
            sw = float(width0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sh, sw)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (height0, width0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (height0, width0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return patch_pos_embed.to(previous_dtype)

    def forward(self, encoder_input: Union[ViTEncoderInput, ViTEncoderNonImageInput]) -> ViTEncoderOutput:
        """
        Dense Representation Encoder Forward Pass

        Args:
            encoder_input (Union[ViTEncoderInput, ViTEncoderNonImageInput]): Input data for the encoder.
                If input type is ViTEncoderInput, input data must contain image normalization type and normalized image tensor.
                If input type is ViTEncoderNonImageInput, input data must contain a tensor of size (B, C, H, W).

        Returns:
            ViTEncoderOutput: Output data from the encoder.
        """
        # Get the input data and verify normalization if the input type is ViTEncoderInput
        if isinstance(encoder_input, ViTEncoderInput):
            self._check_data_normalization_type(encoder_input.data_norm_type)
            input_data = encoder_input.image
        elif isinstance(encoder_input, ViTEncoderNonImageInput):
            input_data = encoder_input.data
        else:
            raise ValueError("Unsupported input type for Dense Representation Encoder.")

        # Check the dtype and shape of the input
        assert isinstance(input_data, torch.Tensor), "Input must be a torch.Tensor"
        assert input_data.ndim == 4, "Input must be of shape (B, C, H, W)"
        assert input_data.shape[1] == self.in_chans, f"Input channels must be {self.in_chans}"
        batch_size, channels, height, width = input_data.shape
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input shape must be divisible by patch size: {self.patch_size}"

        # Encode the dense representation
        features = self.unshuffle(input_data)
        features = self.conv_in(features)
        features = self.encoder(features)
        features = features.flatten(2).transpose(
            1, 2
        )  # (B, E, H / Patch_Size, W / Patch_Size) -> (B, H / Patch_Size * W / Patch_Size, E)
        features = self.norm_layer(features)  # Normalize the features after patch encoding

        # Apply positional encoding if required
        if self.apply_pe:
            features = features + self.interpolate_pos_encoding(
                features, height, width
            )  # (B, H / Patch_Size * W / Patch_Size, E)
            features = self.post_pe_norm(features)  # Normalize the features after positional encoding

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()

        return ViTEncoderOutput(features=features)


if __name__ == "__main__":
    # Init Dense Representation Encoder for images as input
    patch_embedder = DenseRepresentationEncoder(
        name="dense_rep_encoder",
        data_norm_type="dense_rep_encoder",
        input_size_for_pe=518,
        patch_size=14,
        in_chans=3,
        enc_embed_dim=1024,
        apply_pe=False,
    )

    # Test dummy image input
    dummy_image = torch.randn(1, 3, 518, 518)
    patch_embedder_output = patch_embedder(ViTEncoderInput(data_norm_type="dense_rep_encoder", image=dummy_image))
    assert patch_embedder_output.features.shape == (
        1,
        1024,
        37,
        37,
    ), "Output features must have shape (1, 1024, 37, 37)"

    # Init Dense Representation Encoder for non-image data as input
    patch_embedder = DenseRepresentationEncoder(
        name="dense_rep_encoder",
        data_norm_type="dense_rep_encoder",
        input_size_for_pe=518,
        patch_size=14,
        in_chans=6,
        enc_embed_dim=1024,
    )

    # Init Dense Representation Encoder for single channel input
    patch_embedder = DenseRepresentationEncoder(
        name="dense_rep_encoder",
        data_norm_type="dense_rep_encoder",
        input_size_for_pe=518,
        patch_size=14,
        in_chans=1,
        enc_embed_dim=1024,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        apply_pe=True,
    )

    # Test dummy non-image input
    dummy_image = torch.randn(1, 1, 980, 980)
    patch_embedder_output = patch_embedder(ViTEncoderNonImageInput(data=dummy_image))
    assert patch_embedder_output.features.shape == (
        1,
        1024,
        70,
        70,
    ), "Output features must have shape (1, 1024, 70, 70)"

    print("All variants of Dense Representation Encoder have been initialized successfully!")
