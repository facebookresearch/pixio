"""
Encoder class for Global Representation Encoder
"""

from functools import partial
from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn

from map_anything.encoders.base import EncoderGlobalRepInput, EncoderGlobalRepOutput


class GlobalRepresentationEncoder(nn.Module):
    "UniCeption Global Representation Encoder"

    def __init__(
        self,
        name: str,
        in_chans: int = 3,
        enc_embed_dim: int = 1024,
        intermediate_dims: List[int] = [128, 256, 512],
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-6),
        pretrained_checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Global Representation Encoder for projecting a global representation to a desired latent dimension.

        Args:
            name (str): Name of the Encoder.
            in_chans (int): Number of input channels.
            enc_embed_dim (int): Embedding dimension of the encoder.
            intermediate_dims (List[int]): List of intermediate dimensions of the encoder.
            act_layer (Type[nn.Module]): Activation layer to use in the encoder.
            norm_layer (Union[Type[nn.Module], Callable[..., nn.Module]]): Final normalization layer to use in the encoder.
            pretrained_checkpoint_path (Optional[str]): Path to pretrained checkpoint. (default: None)
        """
        super().__init__(*args, **kwargs)

        # Initialize the attributes
        self.name = name
        self.in_chans = in_chans
        self.enc_embed_dim = enc_embed_dim
        self.intermediate_dims = intermediate_dims
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        # Init the activation layer
        self.act_layer = act_layer()

        # Initialize the encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.in_chans, self.intermediate_dims[0]),
            self.act_layer,
        )
        for intermediate_idx in range(1, len(self.intermediate_dims)):
            self.encoder = nn.Sequential(
                self.encoder,
                nn.Linear(self.intermediate_dims[intermediate_idx - 1], self.intermediate_dims[intermediate_idx]),
                self.act_layer,
            )
        self.encoder = nn.Sequential(
            self.encoder,
            nn.Linear(self.intermediate_dims[-1], self.enc_embed_dim),
        )

        # Init weights of the final norm layer
        self.norm_layer = norm_layer(enc_embed_dim) if norm_layer else nn.Identity()
        if isinstance(self.norm_layer, nn.LayerNorm):
            nn.init.constant_(self.norm_layer.bias, 0)
            nn.init.constant_(self.norm_layer.weight, 1.0)

        # Load pretrained weights if provided
        if self.pretrained_checkpoint_path is not None:
            print(
                f"Loading pretrained Global Representation Encoder checkpoint from {self.pretrained_checkpoint_path} ..."
            )
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, encoder_input: EncoderGlobalRepInput) -> EncoderGlobalRepOutput:
        """
        Global Representation Encoder Forward Pass

        Args:
            encoder_input (EncoderGlobalRepInput): Input data for the encoder.
                The provided data must contain a tensor of size (B, C).

        Returns:
            EncoderGlobalRepOutput: Output features from the encoder.
        """
        # Get the input data and verify the shape of the input
        input_data = encoder_input.data
        assert input_data.ndim == 2, "Input data must have shape (B, C)"
        assert input_data.shape[1] == self.in_chans, f"Input data must have {self.in_chans} channels"

        # Encode the global representation
        features = self.encoder(input_data)

        # Normalize the output
        features = self.norm_layer(features)

        return EncoderGlobalRepOutput(features=features)


if __name__ == "__main__":
    dummy_model = GlobalRepresentationEncoder(
        name="dummy", in_chans=3, enc_embed_dim=1024, intermediate_dims=[128, 256, 512]
    )
    dummy_input = EncoderGlobalRepInput(data=torch.randn(4, 3))
    dummy_output = dummy_model(dummy_input)
    assert dummy_output.features.shape == (4, 1024), "Output features must have shape (B, 1024)"
    print("Global Representation Encoder has been initialized successfully!")
