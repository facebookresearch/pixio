"""
Base Information Sharing Class for UniCeption
"""

from dataclasses import dataclass
from typing import List, Optional

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.utils.checkpoint import checkpoint


@dataclass
class InfoSharingInput:
    pass


@dataclass
class InfoSharingOutput:
    pass


@dataclass
class MultiViewTransformerInput(InfoSharingInput):
    """
    Input class for Multi-View Transformer.
    """

    features: List[Float[Tensor, "batch input_embed_dim feat_height feat_width"]]
    additional_input_tokens: Optional[Float[Tensor, "batch input_embed_dim num_additional_tokens"]] = None


@dataclass
class MultiViewTransformerOutput(InfoSharingOutput):
    """
    Output class for Multi-View Transformer.
    """

    features: List[Float[Tensor, "batch transformer_embed_dim feat_height feat_width"]]
    additional_token_features: Optional[Float[Tensor, "batch transformer_embed_dim num_additional_tokens"]] = None


@dataclass
class MultiSetTransformerInput(InfoSharingInput):
    """
    Input class for Multi-Set Transformer.
    """

    features: List[Float[Tensor, "batch input_embed_dim num_tokens"]]
    additional_input_tokens: Optional[Float[Tensor, "batch input_embed_dim num_additional_tokens"]] = None


@dataclass
class MultiSetTransformerOutput(InfoSharingOutput):
    """
    Output class for Multi-Set Transformer.
    """

    features: List[Float[Tensor, "batch transformer_embed_dim num_tokens"]]
    additional_token_features: Optional[Float[Tensor, "batch transformer_embed_dim num_additional_tokens"]] = None

