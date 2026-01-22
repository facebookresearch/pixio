"""
Pose head implementation
Downstream heads assume inputs of size BCHW (B: batch, C: channels, H: height, W: width);
The Pose head implementation is based on Reloc3r and MaRePo
References:
https://github.com/ffrivera0/reloc3r/blob/main/reloc3r/pose_head.py
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from map_anything.prediction_heads.base import PredictionHeadInput, SummaryTaskOutput


class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block implementation based on Reloc3r & MaRePo
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        )
        self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res


class PoseHead(nn.Module):
    """
    Pose regression head implementation based on Reloc3r & MaRePo
    """

    def __init__(
        self,
        patch_size: int,
        input_feature_dim: int,
        num_resconv_block: int = 2,
        rot_representation_dim: int = 4,
        pretrained_checkpoint_path: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the pose head.

        Args:
            patch_size : int, the patch size of the transformer used to generate the input features
            input_feature_dim : int, the input feature dimension
            num_resconv_block : int, the number of residual convolution blocks
            rot_representation_dim : int, the dimension of the rotation representation
            pretrained_checkpoint_path : str, path to pretrained checkpoint (default: None)
        """
        super().__init__()
        self.patch_size = patch_size
        self.input_feature_dim = input_feature_dim
        self.num_resconv_block = num_resconv_block
        self.rot_representation_dim = rot_representation_dim
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        # Initialize the hidden dimension of the pose head based on the patch size
        self.output_dim = 4 * (self.patch_size**2)

        # Initialize the projection layer for the hidden dimension of the pose head
        self.proj = nn.Conv2d(
            in_channels=self.input_feature_dim,
            out_channels=self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Initialize sequential layers of the pose head
        self.res_conv = nn.ModuleList(
            [copy.deepcopy(ResConvBlock(self.output_dim, self.output_dim)) for _ in range(self.num_resconv_block)]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(self.output_dim, 3)
        self.fc_rot = nn.Linear(self.output_dim, self.rot_representation_dim)

        # Load the pretrained checkpoint if provided
        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained pose head from {self.pretrained_checkpoint_path}")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, feature_input: PredictionHeadInput):
        """
        Forward interface for the pose head.
        The pose head requires an adapter on the final output to get the pose.

        Args:
            feature_input : PredictionHeadInput, the input features
            - last_feature : torch.Tensor, the last feature tensor

        Returns:
            SummaryTaskOutput, the output of the pose head
            - decoded_channels : torch.Tensor, the decoded channels
        """
        # Get the patch-level features from the input
        feat = feature_input.last_feature  # (B, C, H, W)

        # Check the input dimensions
        assert (
            feat.shape[1] == self.input_feature_dim
        ), f"Input feature dimension {feat.shape[1]} does not match expected dimension {self.input_feature_dim}"

        # Apply the projection layer to the patch-level features
        feat = self.proj(feat)  # (B, PC, H, W)

        # Apply the residual convolution blocks to the projected features
        for i in range(self.num_resconv_block):
            feat = self.res_conv[i](feat)

        # Apply the average pooling layer to the residual convolution output
        feat = self.avgpool(feat)  # (B, PC, 1, 1)

        # Flatten the average pooled features
        feat = feat.view(feat.size(0), -1)  # (B, PC)

        # Apply the more MLPs to the flattened features
        feat = self.more_mlps(feat)  # (B, PC)

        # Apply the final linear layers to the more MLPs output
        feat_t = self.fc_t(feat)  # (B, 3)
        feat_rot = self.fc_rot(feat)  # (B, self.rot_representation_dim)

        # Concatenate the translation and rotation features
        output_feat = torch.cat([feat_t, feat_rot], dim=1)  # (B, 3 + self.rot_representation_dim

        return SummaryTaskOutput(decoded_channels=output_feat)


if __name__ == "__main__":
    # Init an example pose head
    pose_head = PoseHead(
        patch_size=16,
        input_feature_dim=768,
        num_resconv_block=2,
        rot_representation_dim=4,
        pretrained_checkpoint_path=None,
    )

    # Create a dummy input tensor with shape (B, C, H, W)
    dummy_input = torch.randn(1, 768, 14, 14)  # Example input

    # Run dummy forward pass
    output = pose_head(PredictionHeadInput(last_feature=dummy_input))

    # Check the output shape
    assert output.decoded_channels.shape == (1, 7), "Output shape mismatch"

    print("Pose head test passed!")
