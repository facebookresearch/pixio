"""
MLP head implementation
Downstream heads that coverts a batch of tokens to target representation.
Assumes inputs of size BC (B: batch, C: Channels)
"""

import torch
import torch.nn as nn

from map_anything.prediction_heads.base import PredictionHeadTokenInput, SummaryTaskOutput


class MLPHead(nn.Module):
    """
    MLP head implementation to convert tokens to target representation
    """

    def __init__(
        self,
        input_feature_dim: int,
        output_dim: int,
        num_mlp_layers: int = 2,
        hidden_dim: int = 196,
        pretrained_checkpoint_path: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the MLP head.

        Args:
            input_feature_dim (int): Input feature dimension.
            num_mlp_layers (int): Number of MLP layers.
            pretrained_checkpoint_path (str): Path to a pretrained checkpoint.
        """
        super().__init__()
        self.input_feature_dim = input_feature_dim
        self.num_mlp_layers = num_mlp_layers
        self.hidden_dim = hidden_dim

        # Initialize the input projection layer for the hidden dimension of the mlp head
        self.proj = nn.Linear(self.input_feature_dim, hidden_dim)

        # Initialize the MLP layers
        self.mlp = nn.ModuleList()
        for _ in range(self.num_mlp_layers):
            self.mlp.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()))

        # Initialize the output projection layer for the target representation
        self.output_proj = nn.Linear(self.hidden_dim, output_dim)

        # Load the pretrained checkpoint if provided
        if pretrained_checkpoint_path:
            print(f"Loading pretrained mlp head from {pretrained_checkpoint_path}")
            ckpt = torch.load(pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, feature_input: PredictionHeadTokenInput):
        """
        Forward interface for the mlp head.
        Adapter can be used on output to achieve different types of scaling (linear, log, exp, etc).

        Args:
            feature_input : PredictionHeadTokenInput, the input feature tokens
            - last_feature : torch.Tensor, the last feature tensor

        Returns:
            SummaryTaskOutput, the output of the mlp head
            - decoded_channels : torch.Tensor, the decoded channels
        """
        # Get the token features
        feat = feature_input.last_feature  # (B, C, T)

        # Check the input dimensions
        assert feat.ndim == 3, f"Input feature tensor must have 3 dimensions (B, C, T), got {feat.ndim}"
        assert (
            feat.shape[1] == self.input_feature_dim
        ), f"Input feature dimension {feat.shape[1]} does not match expected dimension {self.input_feature_dim}"

        # Apply the projection layer
        feat = feat.permute(0, 2, 1)  # (B, T, C)
        feat = self.proj(feat)  # (B, hidden_dim)

        # Apply the MLP layers
        for layer in self.mlp:
            feat = layer(feat)

        # Apply the output projection layer
        output = self.output_proj(feat)
        output = output.permute(0, 2, 1)  # (B, C, T)

        return SummaryTaskOutput(decoded_channels=output)


if __name__ == "__main__":
    # Init an example MLP head
    mlp_head = MLPHead(
        input_feature_dim=768,
        output_dim=1,
        num_mlp_layers=2,
        hidden_dim=196,
        pretrained_checkpoint_path=None,
    )

    # Create a dummy input tensor with shape (B, C, T)
    dummy_input = torch.randn(4, 768, 3)  # Example batch of 4 with 768 features

    # Run dummy forward pass
    output = mlp_head(PredictionHeadTokenInput(last_feature=dummy_input))

    # Check the output shape
    assert output.decoded_channels.shape == (4, 1, 3), "Output shape mismatch"

    print("MLP head test passed!")
