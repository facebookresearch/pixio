"""
Base Encoder Class for UniCeption
"""

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.utils.checkpoint import checkpoint


@dataclass
class EncoderInput:
    "Data class for Encoder Input"

    data_norm_type: str
    # Add other fields that are required by the specific implementation of the encoder.


@dataclass
class EncoderOutput:
    "Data class for Encoder Output"

    pass


@dataclass
class EncoderGlobalRepInput:
    "Data class for Encoder Global Representation Input"

    data: Float[Tensor, "batch channel"]


@dataclass
class EncoderGlobalRepOutput:
    "Data class for Encoder Global Representation Output"

    features: Float[Tensor, "batch enc_embed_dim"]


@dataclass
class ViTEncoderInput(EncoderInput):
    "Data class for Vision Transformer Encoder Input"

    image: Float[Tensor, "batch channel height width"]


@dataclass
class ViTEncoderNonImageInput:
    "Data class for Vision (2D-Grid) Transformer Encoder Non-Image Input"

    data: Float[Tensor, "batch channel height width"]


@dataclass
class ViTEncoderOutput(EncoderOutput):
    "Data class for Vision Transformer Encoder Output"

    features: Float[Tensor, "batch enc_embed_dim feat_height feat_width"]
